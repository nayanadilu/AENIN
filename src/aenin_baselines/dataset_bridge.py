"""
Dataset wrapper bridging your REAL AENIN data pipeline (PyG `Data` objects
built in feature_pipeline.py / your notebook) to two consumption modes:

  1. Native PyG mode   -> use torch_geometric.loader.DataLoader directly on
                          the list of Data objects (this is what your AENIN
                          model's forward(data) expects).

  2. Dense mode        -> AENINDenseDataset + collate_fn below convert each
                          PyG Data into (x, A, ts) dense tensors so the 8
                          baseline models in models.py (GroupINN, BrainGNN,
                          EV-GCN, AL-NEGAT, Ex-NEGAT, DeepASD, GNN-LSTM,
                          MCDGLN) can train on the EXACT same subjects,
                          splits, and underlying graph/features as AENIN.

Both modes pull from the SAME cached dataset (dataset.pt / dataset_cor.pt),
so the comparison in train_compare.py is apples-to-apples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Config, load_or_build_dataset, balanced_subset already defined above in this notebook


# --------------------------------------------------------------------------- #
# PyG Data -> dense (x, A) conversion
# --------------------------------------------------------------------------- #

def pyg_to_dense_adj(data, n_rois: int) -> torch.Tensor:
    """
    Rebuild the dense (N, N) weighted adjacency actually used by AENIN's
    message passing, from edge_index/edge_attr — excluding the self-loops
    that build_edge_index_and_attr() adds (DenseGCNLayer adds its own).
    """
    A = torch.zeros((n_rois, n_rois), dtype=torch.float32)
    src, dst = data.edge_index
    w = data.edge_attr.squeeze(-1)
    keep = src != dst  # drop self loops; dense layers re-add them
    A[src[keep], dst[keep]] = w[keep]
    return A


def pyg_node_features_18(data, n_rois: int) -> torch.Tensor:
    """
    data.x is (N, 18 + N): your notebook concatenates the 18-dim topology/
    signal features with the N x N correlation matrix. Slice back the first
    18 columns for the baseline models (in_dim=18, matching models.py).
    """
    return data.x[:, :18].clone()


class AENINDenseDataset(Dataset):
    """
    Wraps a list of PyG `Data` objects (from feature_pipeline.load_or_build_dataset)
    and serves dense tensors for the 8 baseline models.
    """

    def __init__(self, pyg_dataset, n_rois: int = 116, max_T: int = None):
        self.data_list = pyg_dataset
        self.n_rois = n_rois
        self.max_T = max_T

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        x = pyg_node_features_18(d, self.n_rois)             # (N, 18)
        A = pyg_to_dense_adj(d, self.n_rois)                  # (N, N) weighted

        if hasattr(d, "time_series") and d.time_series is not None:
            ts = d.time_series.clone()                        # (T, N)
        else:
            # Dynamic baselines (GNN-LSTM, MCDGLN) need raw BOLD signal.
            # Your cached dataset.pt may not include it (only AENIN-ready
            # x/edge_index/edge_attr/plv_matrix are saved) — rebuild with
            # feature_pipeline.build_dataset(..., keep_time_series=True)
            # if you need those two baselines. Fallback: zero placeholder.
            T_fallback = self.max_T or 1
            ts = torch.zeros((T_fallback, self.n_rois), dtype=torch.float32)

        if self.max_T is not None:
            T = ts.shape[0]
            if T >= self.max_T:
                ts = ts[:self.max_T]
            else:
                pad = torch.zeros((self.max_T - T, self.n_rois), dtype=torch.float32)
                ts = torch.cat([ts, pad], dim=0)

        return {
            "x": x,
            "A": A,
            "ts": ts,
            "y": d.y.view(-1)[0].long(),
            "site": getattr(d, "site", "UNKNOWN_SITE"),
        }


def collate_fn(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    A = torch.stack([b["A"] for b in batch], dim=0)
    ts = torch.stack([b["ts"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    sites = [b["site"] for b in batch]
    return {"x": x, "A": A, "ts": ts, "y": y, "site": sites}


# --------------------------------------------------------------------------- #
# Convenience: site / label arrays for sklearn CV splitters
# --------------------------------------------------------------------------- #

def get_labels_and_sites(pyg_dataset):
    labels = np.array([d.y.item() for d in pyg_dataset])
    sites = np.array([getattr(d, "site", "UNKNOWN_SITE") for d in pyg_dataset])
    return labels, sites
