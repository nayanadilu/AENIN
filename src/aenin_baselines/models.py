"""
Baseline model implementations for comparison against AENIN.

All models share a common input interface so they can be dropped into the
same training / 5-fold and site-wise CV loops as AENIN:

    x : (B, N, F)   node feature matrix   (F = 18 in your pipeline)
    A : (B, N, N)   weighted adjacency / correlation matrix (PLV / Pearson r)
    ts: (B, T, N)   raw BOLD time series  (only used by the dynamic models:
                                            GNN-LSTM, MCDGLN)

Each model's forward() returns raw logits of shape (B, num_classes).

Models implemented (faithful-but-compact re-implementations, not the
authors' original repos, since most of these papers do not release code
that matches your exact feature set):

  1. GroupINN    - Yan et al. 2019
  2. BrainGNN    - Li et al. 2021
  3. EV-GCN      - Huang & Chung 2022
  4. AL-NEGAT    - Chen et al. 2024
  5. Ex-NEGAT    - Bhavna et al. 2025
  6. DeepASD     - Chen et al. 2024
  7. GNN-LSTM    - Dvornek et al. 2017
  8. MCDGLN      - Wang et al. 2025

Install deps:
    pip install torch --break-system-packages
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #

class DenseGCNLayer(nn.Module):
    """Plain dense GCN layer: H' = act(D^-1/2 A D^-1/2 H W)."""

    def __init__(self, in_dim, out_dim, act=F.relu, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act

    @staticmethod
    def normalize_adj(A, eps=1e-6):
        # A: (B, N, N), make symmetric + self loops, then D^-1/2 A D^-1/2
        A = A.clone()
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
        A = A + I
        deg = A.sum(-1).clamp(min=eps)
        d_inv_sqrt = deg.pow(-0.5)
        D = torch.diag_embed(d_inv_sqrt)
        return D @ A @ D

    def forward(self, x, A_norm):
        h = A_norm @ x
        h = self.lin(h)
        return self.act(h) if self.act is not None else h


class DenseGATLayer(nn.Module):
    """Multi-head dense graph attention layer (GAT), edge-masked by A."""

    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, A_mask, edge_feat=None):
        # x: (B, N, Fin), A_mask: (B, N, N) binary / weighted mask
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)          # (B,N,H,D)
        src = (h * self.a_src).sum(-1)                              # (B,N,H)
        dst = (h * self.a_dst).sum(-1)                              # (B,N,H)
        e = self.leaky(src.unsqueeze(2) + dst.unsqueeze(1))         # (B,N,N,H)
        if edge_feat is not None:
            e = e + edge_feat.unsqueeze(-1)                         # optional edge bias
        mask = (A_mask.unsqueeze(-1) > 0)
        e = e.masked_fill(~mask, float('-1e9'))
        alpha = torch.softmax(e, dim=2)                             # over neighbours
        alpha = self.dropout(alpha)
        h = h.permute(0, 2, 1, 3)                                   # (B,H,N,D)
        alpha = alpha.permute(0, 3, 1, 2)                           # (B,H,N,N)
        out = alpha @ h                                             # (B,H,N,D)
        out = out.permute(0, 2, 1, 3)                                # (B,N,H,D)
        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_dim)
        else:
            out = out.mean(dim=2)
        return out, alpha  # return attention for explainability models


def readout(x, mode="mean"):
    """Graph-level readout over node dimension. x: (B, N, F)."""
    if mode == "mean":
        return x.mean(dim=1)
    elif mode == "max":
        return x.max(dim=1).values
    elif mode == "meanmax":
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
    raise ValueError(mode)


def topk_pool(x, A, score, k_ratio=0.5):
    """Simple top-k node pooling (used by BrainGNN's R-pool)."""
    B, N, F_ = x.shape
    k = max(1, int(N * k_ratio))
    topk_idx = score.topk(k, dim=1).indices                         # (B,k)
    gate = torch.sigmoid(score.gather(1, topk_idx)).unsqueeze(-1)   # (B,k,1)
    x_idx = topk_idx.unsqueeze(-1).expand(-1, -1, F_)
    x_pool = x.gather(1, x_idx) * gate
    a_idx_row = topk_idx.unsqueeze(-1).expand(-1, -1, N)
    A_pool = A.gather(1, a_idx_row)
    a_idx_col = topk_idx.unsqueeze(1).expand(-1, k, -1)
    A_pool = A_pool.gather(2, a_idx_col)
    return x_pool, A_pool


# --------------------------------------------------------------------------- #
# 1. GroupINN  (Yan et al., 2019)
# --------------------------------------------------------------------------- #

class GroupINN(nn.Module):
    """
    Group-wise feature reduction GCN: learns a soft clustering / grouping
    matrix S that compresses N ROIs into K "functional groups", then runs
    graph convolutions on the reduced (K x K) graph.
    """

    def __init__(self, n_nodes, in_dim=18, n_groups=16, hidden=32, num_classes=2):
        super().__init__()
        self.n_groups = n_groups
        self.assign = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_groups)
        )
        self.gcn1 = DenseGCNLayer(in_dim, hidden)
        self.gcn2 = DenseGCNLayer(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x, A, ts=None):
        S = torch.softmax(self.assign(x), dim=-1)          # (B, N, K) soft grouping
        A_norm = DenseGCNLayer.normalize_adj(A)
        A_group = S.transpose(1, 2) @ A_norm @ S            # (B, K, K) reduced graph
        x_group = S.transpose(1, 2) @ x                     # (B, K, F) reduced features
        A_group_norm = DenseGCNLayer.normalize_adj(A_group)
        h = self.gcn1(x_group, A_group_norm)
        h = self.gcn2(h, A_group_norm)
        g = readout(h, "mean")
        return self.classifier(g)


# --------------------------------------------------------------------------- #
# 2. BrainGNN  (Li et al., 2021)
# --------------------------------------------------------------------------- #

class ROIAwareConv(nn.Module):
    """Ra-GConv: a separate filter bank conditioned on ROI community id."""

    def __init__(self, in_dim, out_dim, n_communities=8):
        super().__init__()
        self.n_communities = n_communities
        self.community_embed = nn.Embedding(n_communities, in_dim)
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm, community_ids):
        # community_ids: (N,) long tensor, shared across batch
        comm = self.community_embed(community_ids)          # (N, Fin)
        x_mod = x * torch.sigmoid(comm).unsqueeze(0)         # ROI-aware reweight
        h = A_norm @ x_mod
        return F.relu(self.W(h))


class BrainGNN(nn.Module):
    def __init__(self, n_nodes, in_dim=18, hidden=32, n_communities=8,
                 pool_ratio=0.5, num_classes=2):
        super().__init__()
        self.register_buffer(
            "community_ids",
            torch.randint(0, n_communities, (n_nodes,))   # replace with real
        )                                                  # atlas-based community
        self.conv1 = ROIAwareConv(in_dim, hidden, n_communities)
        self.score1 = nn.Linear(hidden, 1)
        self.conv2 = ROIAwareConv(hidden, hidden, n_communities)
        self.pool_ratio = pool_ratio
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x, A, ts=None):
        A_norm = DenseGCNLayer.normalize_adj(A)
        h1 = self.conv1(x, A_norm, self.community_ids)
        s1 = self.score1(h1).squeeze(-1)
        h1_pool, A_pool = topk_pool(h1, A, s1, self.pool_ratio)
        N_pool = h1_pool.shape[1]
        comm_pool = self.community_ids[:N_pool]              # approx after pooling
        A_pool_norm = DenseGCNLayer.normalize_adj(A_pool)
        h2 = self.conv2(h1_pool, A_pool_norm, comm_pool)
        g = readout(h2, "meanmax")
        return self.classifier(g)


# --------------------------------------------------------------------------- #
# 3. EV-GCN  (Huang & Chung, 2022)
# --------------------------------------------------------------------------- #

class EdgeWeightNet(nn.Module):
    """Learns a scalar edge weight from a pair of node features (MLP on |xi-xj|, xi+xj)."""

    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        B, N, Fd = x.shape
        xi = x.unsqueeze(2).expand(B, N, N, Fd)
        xj = x.unsqueeze(1).expand(B, N, N, Fd)
        pair = torch.cat([torch.abs(xi - xj), xi + xj], dim=-1)
        w = torch.sigmoid(self.mlp(pair)).squeeze(-1)        # (B, N, N)
        return w


class EVGCN(nn.Module):
    """
    Edge-Variational GCN: combines the empirical correlation matrix with a
    learned, feature-driven edge-weight matrix (variational edge estimation),
    then performs standard graph convolution on the fused adjacency.
    """

    def __init__(self, n_nodes, in_dim=18, hidden=32, num_classes=2):
        super().__init__()
        self.edge_net = EdgeWeightNet(in_dim, hidden=32)
        self.alpha = nn.Parameter(torch.tensor(0.5))          # fusion weight (learnable)
        self.gcn1 = DenseGCNLayer(in_dim, hidden)
        self.gcn2 = DenseGCNLayer(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x, A, ts=None):
        w_learned = self.edge_net(x)
        a = torch.sigmoid(self.alpha)
        A_fused = a * A.abs() + (1 - a) * w_learned
        A_norm = DenseGCNLayer.normalize_adj(A_fused)
        h = self.gcn1(x, A_norm)
        h = self.gcn2(h, A_norm)
        g = readout(h, "mean")
        return self.classifier(g)


# --------------------------------------------------------------------------- #
# 4. AL-NEGAT  (Chen et al., 2024) - Attention-Learning Node-Edge GAT
# --------------------------------------------------------------------------- #

class AL_NEGAT(nn.Module):
    """
    Node-Edge GAT: attention is computed jointly from node features AND the
    edge weight (correlation strength) feeding an extra bias term into the
    attention logits, with multi-layer attention "learning" via stacked heads.
    """

    def __init__(self, n_nodes, in_dim=18, hidden=16, heads=4, num_classes=2):
        super().__init__()
        self.edge_proj = nn.Linear(1, heads)
        self.gat1 = DenseGATLayer(in_dim, hidden, heads=heads, concat=True)
        self.gat2 = DenseGATLayer(hidden * heads, hidden, heads=heads, concat=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * heads, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x, A, ts=None):
        edge_bias = self.edge_proj(A.unsqueeze(-1))            # (B,N,N,heads)
        h, _ = self.gat1(x, A, edge_feat=edge_bias.mean(-1, keepdim=False))
        h = F.elu(h)
        h, _ = self.gat2(h, A, edge_feat=edge_bias.mean(-1, keepdim=False))
        h = F.elu(h)
        g = readout(h, "mean")
        return self.classifier(g)


# --------------------------------------------------------------------------- #
# 5. Ex-NEGAT  (Bhavna et al., 2025) - Explainable Node-Edge GAT
# --------------------------------------------------------------------------- #

class ExNEGAT(nn.Module):
    """
    Same backbone as AL-NEGAT but exposes per-layer attention maps for
    post-hoc explainability (saliency over edges / ROIs), and adds an
    attention-entropy regularization term you can add to the training loss
    to encourage sparse, interpretable attention.
    """

    def __init__(self, n_nodes, in_dim=18, hidden=16, heads=4, num_classes=2):
        super().__init__()
        self.gat1 = DenseGATLayer(in_dim, hidden, heads=heads, concat=True)
        self.gat2 = DenseGATLayer(hidden * heads, hidden, heads=heads, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )
        self.last_attn = None  # stash for explainability plots

    def forward(self, x, A, ts=None):
        h, alpha1 = self.gat1(x, A)
        h = F.elu(h)
        h, alpha2 = self.gat2(h, A)
        h = F.elu(h)
        self.last_attn = (alpha1.detach(), alpha2.detach())
        g = readout(h, "mean")
        return self.classifier(g)

    @staticmethod
    def attention_entropy_loss(alpha, eps=1e-9):
        # alpha: (B, H, N, N) softmax weights -> encourage low entropy (sparse, interpretable)
        p = alpha.clamp(min=eps)
        ent = -(p * p.log()).sum(-1).mean()
        return ent


# --------------------------------------------------------------------------- #
# 6. DeepASD  (Chen et al., 2024) - CNN over the connectivity matrix
# --------------------------------------------------------------------------- #

class DeepASD(nn.Module):
    """
    Treats the N x N correlation matrix as a single-channel image and the
    node feature matrix as an auxiliary channel-stack, passed through a
    2D-CNN ("connectome-CNN") followed by an MLP classifier.
    """

    def __init__(self, n_nodes, in_dim=18, num_classes=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 32, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, A, ts=None):
        img = A.unsqueeze(1)                                   # (B,1,N,N)
        c = self.conv(img).flatten(1)                          # (B, 64*4*4)
        node_g = self.node_mlp(x).mean(dim=1)                  # (B,32)
        out = torch.cat([c, node_g], dim=-1)
        return self.classifier(out)


# --------------------------------------------------------------------------- #
# 7. GNN-LSTM  (Dvornek et al., 2017)
# --------------------------------------------------------------------------- #

class GNN_LSTM(nn.Module):
    """
    Sliding-window dynamic connectivity: for each window, build a correlation
    graph from the raw time series, run a GCN to get a graph embedding, then
    feed the sequence of window embeddings into an LSTM for temporal modeling.
    Requires `ts` (B, T, N) raw BOLD signals.
    """

    def __init__(self, n_nodes, in_dim=18, hidden=32, window=30, stride=10,
                 lstm_hidden=32, num_classes=2):
        super().__init__()
        self.window = window
        self.stride = stride
        self.gcn = DenseGCNLayer(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    @staticmethod
    def _window_corr(ts_window, eps=1e-8):
        # ts_window: (B, w, N) -> (B, N, N) Pearson correlation
        x = ts_window - ts_window.mean(dim=1, keepdim=True)
        cov = torch.einsum('bwi,bwj->bij', x, x)
        std = x.std(dim=1, unbiased=False).clamp(min=eps)
        denom = std.unsqueeze(-1) * std.unsqueeze(-2) * ts_window.shape[1]
        corr = cov / denom.clamp(min=eps)
        return corr

    def forward(self, x, A, ts):
        # x is the static 18-d node features, reused at every time window
        B, T, N = ts.shape
        embeds = []
        for start in range(0, max(T - self.window, 1), self.stride):
            w_ts = ts[:, start:start + self.window, :]
            if w_ts.shape[1] < 2:
                continue
            A_t = self._window_corr(w_ts)
            A_t_norm = DenseGCNLayer.normalize_adj(A_t)
            h = self.gcn(x, A_t_norm)
            embeds.append(readout(h, "mean"))
        if len(embeds) == 0:                                   # fallback: single static graph
            A_norm = DenseGCNLayer.normalize_adj(A)
            embeds = [readout(self.gcn(x, A_norm), "mean")]
        seq = torch.stack(embeds, dim=1)                        # (B, W, hidden)
        out, (hN, cN) = self.lstm(seq)
        return self.classifier(hN[-1])


# --------------------------------------------------------------------------- #
# 8. MCDGLN  (Wang et al., 2025) - Multi-Channel Dynamic Graph Learning Net
# --------------------------------------------------------------------------- #

class MCDGLN(nn.Module):
    """
    Builds K dynamic graph "channels" from sliding windows of the BOLD time
    series (e.g. correlation, partial correlation-style precision proxy,
    and a thresholded binary channel), runs a separate GCN per channel per
    window, fuses channels with learned attention, then aggregates the
    temporal sequence with a GRU before classification.
    """

    def __init__(self, n_nodes, in_dim=18, hidden=24, window=30, stride=15,
                 n_channels=3, gru_hidden=32, num_classes=2):
        super().__init__()
        self.window = window
        self.stride = stride
        self.n_channels = n_channels
        self.channel_gcns = nn.ModuleList(
            [DenseGCNLayer(in_dim, hidden) for _ in range(n_channels)]
        )
        self.channel_attn = nn.Linear(hidden, 1)
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    @staticmethod
    def _make_channels(A_t, n_channels):
        chans = [A_t]
        if n_channels >= 2:
            chans.append((A_t.abs() > 0.3).float() * A_t)        # sparsified channel
        if n_channels >= 3:
            chans.append(torch.tanh(3.0 * A_t))                  # nonlinear-emphasis channel
        return chans[:n_channels]

    def forward(self, x, A, ts):
        B, T, N = ts.shape
        window_embeds = []
        for start in range(0, max(T - self.window, 1), self.stride):
            w_ts = ts[:, start:start + self.window, :]
            if w_ts.shape[1] < 2:
                continue
            A_t = GNN_LSTM._window_corr(w_ts)
            chans = self._make_channels(A_t, self.n_channels)
            chan_embeds = []
            for c_idx, A_c in enumerate(chans):
                A_norm = DenseGCNLayer.normalize_adj(A_c)
                h = self.channel_gcns[c_idx](x, A_norm)
                chan_embeds.append(readout(h, "mean"))
            chan_stack = torch.stack(chan_embeds, dim=1)          # (B, C, hidden)
            attn_w = torch.softmax(self.channel_attn(chan_stack), dim=1)
            fused = (chan_stack * attn_w).sum(dim=1)               # (B, hidden)
            window_embeds.append(fused)
        if len(window_embeds) == 0:
            A_norm = DenseGCNLayer.normalize_adj(A)
            window_embeds = [readout(self.channel_gcns[0](x, A_norm), "mean")]
        seq = torch.stack(window_embeds, dim=1)                    # (B, W, hidden)
        out, hN = self.gru(seq)
        return self.classifier(hN[-1])


# --------------------------------------------------------------------------- #
# Registry, for convenient lookup in the training script
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "GroupINN": GroupINN,
    "BrainGNN": BrainGNN,
    "EV-GCN": EVGCN,
    "AL-NEGAT": AL_NEGAT,
    "Ex-NEGAT": ExNEGAT,
    "DeepASD": DeepASD,
    "GNN-LSTM": GNN_LSTM,
    "MCDGLN": MCDGLN,
}

# Models that require raw time series (dynamic graph construction)
DYNAMIC_MODELS = {"GNN-LSTM", "MCDGLN"}
