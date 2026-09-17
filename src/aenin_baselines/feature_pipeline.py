"""
feature_pipeline.py

This is the REAL data pipeline, lifted from AENIN_lasso.ipynb, so the
8 baseline models and AENIN are trained/evaluated on the exact same
subjects, graphs, and features.

Pipeline:
    .nii / .nii.gz  -->  AAL ROI time series  -->  PLV matrix
        -->  adaptive threshold (A, A_w)  -->  18-dim node features
        -->  concat with N x N correlation matrix  -->  PyG Data object

Expected folder structure (same as your notebook):
    data_root/
        ASD/  sub-xxxx_..._.nii.gz   (label = 1)
        TD/   sub-xxxx_..._.nii.gz   (label = 0)

Site id is parsed from the filename prefix before the first "_"
(same convention as your notebook: site = Path(nii_path).name.split("_")[0]).

Usage:
    from feature_pipeline import load_or_build_dataset, Config

    cfg = Config()
    cfg.DATA_ROOT = "/path/to/Dataset_ABIDE_sorted"
    dataset = load_or_build_dataset(cfg, cache_path="dataset_cor.pt")
    # dataset: list[torch_geometric.data.Data], each with
    #   .x            (N, 18 + N)   node features (18 topo/signal + N-dim corr row)
    #   .edge_index   (2, E)
    #   .edge_attr    (E, 1)        PLV weight (incl. self loops)
    #   .y            (1,)          0 = TD, 1 = ASD
    #   .plv_matrix   (N, N)        full (unthresholded) PLV matrix
    #   .time_series  (T, N)        raw ROI BOLD signal  (added for dynamic baselines)
    #   .site         str           acquisition site id parsed from filename
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import hilbert
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix

import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

from torch_geometric.data import Data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("AENIN-data")


# --------------------------------------------------------------------------- #
# Config (subset relevant to data building; mirrors your notebook's Config)
# --------------------------------------------------------------------------- #

class Config:
    DATA_ROOT: str = "./data"
    ATLAS: str = "aal"
    N_ROIS: int = 116
    APS_ALPHA: float = 0.5
    MIN_EDGES: int = 30
    WAVELET_SCALES: list = [0.5, 1.0, 2.0]
    SEED: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 1. Atlas loading
# --------------------------------------------------------------------------- #

def load_atlas(atlas_name: str):
    log.info(f"Loading atlas: {atlas_name.upper()}")
    if atlas_name.lower() == "aal":
        atlas = datasets.fetch_atlas_aal()
        atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
        roi_labels = list(atlas.labels)
        n_rois = 116
    elif atlas_name.lower() in ("ho", "harvard-oxford"):
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
        roi_labels = list(atlas.labels)
        n_rois = len(roi_labels)
    else:
        raise ValueError(f"Unknown atlas: {atlas_name}. Use 'aal' or 'ho'.")

    atlas_data = atlas_img.get_fdata()
    atlas_values = sorted([int(v) for v in np.unique(atlas_data.astype(np.int32)) if int(v) > 0])
    atlas_values = atlas_values[:n_rois]
    value_to_pos = {v: i for i, v in enumerate(atlas_values)}
    if len(roi_labels) > n_rois:
        roi_labels = roi_labels[:n_rois]
    log.info(f"  Atlas loaded: {n_rois} ROIs")
    return atlas_img, n_rois, roi_labels, value_to_pos


# --------------------------------------------------------------------------- #
# 2. ROI time-series extraction (fixed AAL ordering)
# --------------------------------------------------------------------------- #

def extract_roi_timeseries_fixed_aal(nii_path, atlas_img, t_r=2.0, n_rois=116):
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize=True, detrend=True,
        low_pass=0.1, high_pass=0.01, t_r=t_r, verbose=0,
    )
    ts_partial = masker.fit_transform(nii_path)
    T = ts_partial.shape[0]

    atlas_nii = nib.load(atlas_img) if isinstance(atlas_img, str) else atlas_img
    atlas_data = atlas_nii.get_fdata()
    atlas_values = sorted([int(v) for v in np.unique(atlas_data.astype(np.int32)) if int(v) > 0])
    atlas_values = atlas_values[:n_rois]
    value_to_pos = {v: i for i, v in enumerate(atlas_values)}

    ts_full = np.zeros((T, n_rois), dtype=np.float32)
    extracted_labels = getattr(masker, "labels_", None)
    if extracted_labels is not None:
        extracted_labels = [int(v) for v in extracted_labels if int(v) > 0]
        for k, atlas_value in enumerate(extracted_labels):
            if k >= ts_partial.shape[1]:
                break
            if atlas_value in value_to_pos:
                ts_full[:, value_to_pos[atlas_value]] = ts_partial[:, k]
    else:
        K = min(ts_partial.shape[1], n_rois)
        ts_full[:, :K] = ts_partial[:, :K]

    return np.nan_to_num(ts_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Adaptive Phase Synchronization (APS) graph construction
# --------------------------------------------------------------------------- #

def compute_plv_matrix(time_series: np.ndarray) -> np.ndarray:
    T, N = time_series.shape
    analytic = hilbert(time_series, axis=0)
    phases = np.angle(analytic)
    plv = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        diff = phases[:, i:i + 1] - phases
        plv[i, :] = np.abs(np.mean(np.exp(1j * diff), axis=0))
    np.fill_diagonal(plv, 0.0)
    return plv


def adaptive_threshold(plv_matrix: np.ndarray, alpha: float = 0.5):
    vals = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
    mu, sigma = vals.mean(), vals.std()
    theta = mu + alpha * sigma
    A_w = plv_matrix.copy()
    A_w[A_w < theta] = 0.0
    A = (A_w > 0).astype(np.float32)
    return A, A_w, float(theta)


def build_edge_index_and_attr(A: np.ndarray, A_w: np.ndarray):
    src, dst = np.where(A > 0)
    weights = A_w[src, dst]
    N = A.shape[0]
    self_src = np.arange(N)
    self_dst = np.arange(N)
    self_w = np.ones(N, dtype=np.float32)
    src = np.concatenate([src, self_src])
    dst = np.concatenate([dst, self_dst])
    weights = np.concatenate([weights, self_w])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(weights[:, None], dtype=torch.float32)
    return edge_index, edge_attr


# --------------------------------------------------------------------------- #
# 4. 18-dim node feature extraction (verbatim from your notebook)
# --------------------------------------------------------------------------- #

def _personalized_pagerank(A, alpha=0.85, max_iter=100):
    N = A.shape[0]
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg
    r = np.ones(N, dtype=np.float64) / N
    teleport = np.ones(N, dtype=np.float64) / N
    for _ in range(max_iter):
        r_new = alpha * P.T @ r + (1 - alpha) * teleport
        if np.linalg.norm(r_new - r, 1) < 1e-6:
            break
        r = r_new
    return r.astype(np.float32)


def _harmonic_centrality(A):
    N = A.shape[0]
    sp = shortest_path(csr_matrix(A), directed=False, unweighted=True)
    sp[sp == 0] = np.inf
    hc = (1.0 / sp)
    np.fill_diagonal(hc, 0.0)
    return hc.sum(axis=1).astype(np.float32) / (N - 1)


def _k_core_numbers(A):
    N = A.shape[0]
    adj = (A > 0).astype(int)
    core = np.zeros(N, dtype=np.float32)
    remaining = np.ones(N, dtype=bool)
    k = 1
    while remaining.any():
        changed = True
        while changed:
            changed = False
            for i in np.where(remaining)[0]:
                deg = adj[i][remaining].sum() - adj[i, i]
                if deg < k:
                    remaining[i] = False
                    core[i] = k - 1
                    changed = True
        k += 1
        if k > N:
            break
    core[remaining] = k - 1
    return core


def _wavelet_energy(ts, scales):
    T, N = ts.shape
    energies = np.zeros((N, len(scales)), dtype=np.float32)
    for s_idx, scale in enumerate(scales):
        step = max(1, int(scale * 2))
        downsampled = ts[::step, :]
        energies[:, s_idx] = np.mean(downsampled ** 2, axis=0)
    return energies


def _hub_score(A):
    eigvals, eigvecs = np.linalg.eig(A @ A.T)
    idx = np.argmax(np.real(eigvals))
    hub = np.abs(np.real(eigvecs[:, idx]))
    hub = hub / (hub.max() + 1e-8)
    return hub.astype(np.float32)


def _core_periphery_score(A):
    core = _k_core_numbers(A)
    return (core / (core.max() + 1e-8)).astype(np.float32)


def _participation_score(A):
    n_components, labels = connected_components(A.astype(np.int32), directed=False)
    N = A.shape[0]
    deg = A.sum(axis=1)
    P = np.zeros(N)
    for i in range(N):
        if deg[i] == 0:
            continue
        s = 0
        for c in range(n_components):
            nodes = np.where(labels == c)[0]
            k_im = A[i, nodes].sum()
            s += (k_im / deg[i]) ** 2
        P[i] = 1 - s
    return P.astype(np.float32)


def _flow_betweenness(A):
    sp = shortest_path(A, directed=False)
    N = A.shape[0]
    score = np.zeros(N)
    for i in range(N):
        score[i] = np.isfinite(sp[i]).sum()
    score /= score.max() + 1e-8
    return score.astype(np.float32)


def extract_node_features(time_series, A, A_w, cfg):
    """18-dim per-ROI feature vector. Same indices as your notebook."""
    present_mask = (time_series.std(axis=0) > 1e-8).astype(np.float32)
    T, N = time_series.shape
    feats = np.zeros((N, 18), dtype=np.float32)

    deg = A.sum(axis=1)
    feats[:, 0] = deg / (N - 1 + 1e-8)
    feats[:, 1] = A_w.sum(axis=1)
    feats[:, 2] = _personalized_pagerank(A)
    feats[:, 3] = _harmonic_centrality(A)
    k_core = _k_core_numbers(A)
    feats[:, 4] = k_core / (k_core.max() + 1e-8)

    A_2 = A @ A
    np.fill_diagonal(A_2, 0)
    feats[:, 5] = A_2.sum(axis=1) / (N * (N - 1) + 1e-8)

    avg_nb_deg = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        avg_nb_deg[i] = deg[nb].mean() if len(nb) > 0 else 0.0
    feats[:, 6] = avg_nb_deg / (N + 1e-8)

    feats[:, 7] = time_series.mean(axis=0)
    feats[:, 8] = time_series.std(axis=0)

    wav = _wavelet_energy(time_series, cfg.WAVELET_SCALES)
    feats[:, 9:12] = wav
    feats[:, 13] = present_mask

    plv_nb = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        plv_nb[i] = A_w[i, nb].mean() if len(nb) > 0 else 0.0
    feats[:, 12] = plv_nb

    feats[:, 14] = _hub_score(A)
    feats[:, 15] = _core_periphery_score(A)
    feats[:, 16] = _participation_score(A)
    feats[:, 17] = _flow_betweenness(A)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
# 5. Subject scanning + per-subject processing + dataset building
# --------------------------------------------------------------------------- #

def load_subjects(data_root: str):
    """Scan data_root/ASD and data_root/TD for .nii/.nii.gz. ASD=1, TD=0."""
    root = Path(data_root)
    subjects = []
    for label_name, label_val in [("ASD", 1), ("TD", 0)]:
        folder = root / label_name
        if not folder.exists():
            log.warning(f"  Folder not found: {folder}")
            continue
        nii_files = sorted(list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz")))
        log.info(f"  {label_name}: {len(nii_files)} subjects found")
        for f in nii_files:
            subjects.append((str(f), label_val))
    if len(subjects) == 0:
        raise FileNotFoundError(f"No .nii/.nii.gz files found under {data_root}/ASD/ or {data_root}/TD/")
    return subjects


def process_subject(nii_path: str, label: int, atlas_img, cfg: Config,
                     keep_time_series: bool = True) -> Data:
    """fMRI -> time series -> APS graph -> 18-d node feats -> PyG Data."""
    ts = extract_roi_timeseries_fixed_aal(nii_path, atlas_img, n_rois=cfg.N_ROIS)
    site = Path(nii_path).name.split("_")[0]

    plv = compute_plv_matrix(ts)
    corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)

    A, A_w, theta = adaptive_threshold(plv, alpha=cfg.APS_ALPHA)
    if A.sum() < cfg.MIN_EDGES:
        for alpha_relax in [0.3, 0.1, 0.0]:
            A, A_w, theta = adaptive_threshold(plv, alpha=alpha_relax)
            if A.sum() >= cfg.MIN_EDGES:
                break

    x18 = extract_node_features(ts, A, A_w, cfg)          # (N, 18)
    corr_feat = corr.astype(np.float32)                    # (N, N)
    x = np.concatenate([x18, corr_feat], axis=1).astype(np.float32)  # (N, 18+N)

    edge_index, edge_attr = build_edge_index_and_attr(A, A_w)

    n_edges = int(A.sum())
    density = n_edges / (x.shape[0] * (x.shape[0] - 1) + 1e-8)
    plv_full = plv.copy()
    np.fill_diagonal(plv_full, 0)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        plv_matrix=torch.tensor(plv_full, dtype=torch.float32),
        n_edges=n_edges,
        density=density,
    )
    data.site = site
    if keep_time_series:
        # needed by the dynamic baselines (GNN-LSTM, MCDGLN); not in your
        # original cache, so re-build with keep_time_series=True if you
        # want those two baselines to use real sliding-window dynamics.
        data.time_series = torch.tensor(ts, dtype=torch.float32)
    return data


def build_dataset(subjects: list, atlas_img, cfg: Config, keep_time_series: bool = True) -> list:
    dataset = []
    n = len(subjects)
    failed = 0
    for idx, (nii_path, label) in enumerate(subjects):
        try:
            data = process_subject(nii_path, label, atlas_img, cfg, keep_time_series)
            dataset.append(data)
            if (idx + 1) % 10 == 0 or (idx + 1) == n:
                log.info(f"  Processed {idx + 1}/{n} subjects")
        except Exception as e:
            log.warning(f"  Failed [{Path(nii_path).name}]: {e}")
            failed += 1
    log.info(f"  Dataset built: {len(dataset)} subjects ({failed} failed)")
    return dataset


def load_or_build_dataset(cfg: Config, cache_path: Optional[str] = "dataset_cor.pt",
                           force_rebuild: bool = False, keep_time_series: bool = True):
    """
    Loads dataset.pt / dataset_cor.pt if it already exists (same as your
    `torch.load("dataset.pt", ...)` cells), otherwise builds it from
    cfg.DATA_ROOT and caches it.
    """
    if cache_path is not None and Path(cache_path).exists() and not force_rebuild:
        log.info(f"Loading cached dataset: {cache_path}")
        dataset = torch.load(cache_path, map_location="cpu", weights_only=False)
        log.info(f"  Loaded {len(dataset)} subjects")
        return dataset

    log.info(f"Building dataset from: {cfg.DATA_ROOT}")
    atlas_img, n_rois, roi_labels, value_to_pos = load_atlas(cfg.ATLAS)
    cfg.N_ROIS = n_rois
    subjects = load_subjects(cfg.DATA_ROOT)
    dataset = build_dataset(subjects, atlas_img, cfg, keep_time_series=keep_time_series)

    if cache_path is not None:
        torch.save(dataset, cache_path)
        log.info(f"  Cached dataset to: {cache_path}")
    return dataset


def balanced_subset(dataset, n_per_class: int = 400, seed: int = 42):
    """Matches your notebook's 'keep only N ASD + N TD' balancing step."""
    asd = [d for d in dataset if d.y.item() == 1]
    td = [d for d in dataset if d.y.item() == 0]
    rng = np.random.RandomState(seed)
    rng.shuffle(asd)
    rng.shuffle(td)
    out = asd[:n_per_class] + td[:n_per_class]
    rng.shuffle(out)
    return out
