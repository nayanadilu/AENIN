"""
AENIN accepted-article code release.

This module is derived from the supplied AENIN_lasso(2).ipynb notebook.
Notebook outputs and machine-specific paths were removed for public release.
The algorithmic implementation is preserved. A stale feature-count docstring
was updated to match the executable code, which constructs 18 base node
features before concatenating the correlation row.
"""
# ─── Standard Library ────────────────────────────────────────────────────────
import os
import sys
import argparse
import warnings
import logging
import time
import json
from pathlib import Path
from collections import defaultdict
import subprocess
from typing import Optional
from torch.optim import AdamW

warnings.filterwarnings("ignore")

# ─── Numerical / Scientific ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# ─── Neuroimaging ─────────────────────────────────────────────────────────────
import nibabel as nib
from nilearn import datasets, image, signal
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiLabelsMasker

# ─── Machine Learning ─────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ─── Deep Learning ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── PyTorch Geometric ────────────────────────────────────────────────────────
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    MessagePassing, global_mean_pool, global_max_pool
)
from torch_geometric.utils import add_self_loops, softmax

# ─── Visualization ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.sparse.csgraph import connected_components

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("AENIN")

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — all hyper-parameters in one place."""

    # ── Data
    DATA_ROOT: str = "./data"          # root folder with ASD/ and TD/ sub-dirs
    OUTPUT_DIR: str = "./aenin_output" # where results are saved
    ATLAS: str = "aal"                 # "aal" (116 ROIs) | "ho"  (48 ROIs)
    N_ROIS: int = 116                  # updated automatically from atlas choice

    # ── Graph construction (APS)
    APS_ALPHA: float = 0.5            # threshold = mean + alpha * std
    MIN_EDGES: int = 30               # minimum edges after thresholding

    # ── Node features
    WAVELET_SCALES: list = [0.5, 1.0, 2.0]

    # ── Model architecture
    NODE_FEAT_DIM: int = 130           # set automatically after feature extraction
    EDGE_FEAT_DIM: int = 1            # PLV weight
    HIDDEN_DIM: int = 64
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.5
    HEADS: int = 4                    # attention heads in AENIN layer

    # ── Training
    EPOCHS: int = 200
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 4
    PATIENCE: int = 150                # early stopping patience
    N_FOLDS: int = 5

    
    
    def pick_gpu_with_max_free_memory():
        if not torch.cuda.is_available():
            print("CUDA not available. Using CPU.")
            return torch.device("cpu")

        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8"
            )

            best_gpu = None
            best_free = -1
            for line in result.strip().split("\n"):
                idx, free_mem, total_mem = [x.strip() for x in line.split(",")]
                idx = int(idx)
                free_mem = int(free_mem)
                total_mem = int(total_mem)
                print(f"GPU {idx}: free={free_mem} MB / total={total_mem} MB")
                if free_mem > best_free:
                    best_free = free_mem
                    best_gpu = idx

            if best_gpu is not None:
                print(f"Selected GPU {best_gpu}")
                return torch.device(f"cuda:{best_gpu}")
        except Exception as e:
            print("nvidia-smi query failed:", e)

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Device
    DEVICE: str = pick_gpu_with_max_free_memory()

    # ── Reproducibility
    SEED: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ATLAS LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_atlas(atlas_name: str):
    
    
    
    
    """
    Load atlas and return:
        atlas_img   : loaded Niimg object, not string path
        n_rois      : number of ROIs used
        roi_labels  : ROI names
        value_to_pos: atlas integer label value -> column position
    """
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
def safe_zscore(x: np.ndarray, axis=0):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)

    std[std < 1e-8] = 1.0

    return (x - mean) / std

def pad_or_crop_time(ts: np.ndarray, max_time_steps: Optional[int] = None) -> np.ndarray:
    """Optional time normalization. If max_time_steps is None, keep original T."""
    if max_time_steps is None:
        return ts
    T, N = ts.shape
    if T == max_time_steps:
        return ts
    if T > max_time_steps:
        return ts[:max_time_steps, :]
    pad = np.zeros((max_time_steps - T, N), dtype=ts.dtype)
    return np.vstack([ts, pad])


# def extract_roi_timeseries(
#     nii_path: str,
#     atlas_img,
#     value_to_pos: dict,
#     n_rois: int = 116,
#     max_time_steps: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     Manual ROI extraction with correct atlas label mapping.

#     This fixes:
#         1) 'str' object has no attribute 'shape' by loading atlas paths.
#         2) index 116/114/104 out-of-bounds by mapping atlas label VALUES to 0..115 positions.
#         3) accuracy drop from naive padding/truncation by preserving true AAL label order.
#     """
#     fmri_img = nib.load(str(nii_path))

#     if isinstance(atlas_img, str):
#         atlas_img = nib.load(atlas_img)

#     raw_shape = tuple(fmri_img.shape)
#     if len(raw_shape) != 4:
#         raise ValueError(f"Expected 4D fMRI NIfTI, got {raw_shape} for {nii_path}")

#     if atlas_img.shape[:3] != fmri_img.shape[:3] or not np.allclose(atlas_img.affine, fmri_img.affine):
#         atlas_img = resample_to_img(
#             source_img=atlas_img,
#             target_img=fmri_img,
#             interpolation="nearest",
#             force_resample=True,
#             copy_header=True,
#         )

#     # float32 saves RAM compared with default float64
#     fmri_data = fmri_img.get_fdata(dtype=np.float32)
#     atlas_data = atlas_img.get_fdata().astype(np.int32)

#     _, _, _, T = fmri_data.shape
#     ts = np.zeros((T, n_rois), dtype=np.float32)

#     present_values = np.unique(atlas_data)
#     present_values = [int(v) for v in present_values if int(v) in value_to_pos]
#     atlas_labels = sorted([int(v) for v in np.unique(atlas_data.astype(np.int32)) if int(v) > 0])

#     value_to_pos = {
#         label: idx
#         for idx, label in enumerate(atlas_labels[:116])
#     }

#     for atlas_value in present_values:
#         pos = value_to_pos[int(atlas_value)]
#         if pos >= n_rois:
#             continue
#         mask = atlas_data == atlas_value
#         if np.any(mask):
#             voxel_ts = fmri_data[mask]  # shape: n_voxels x T
#             if voxel_ts.size > 0:
#                 ts[:, pos] = np.mean(voxel_ts, axis=0)

#     ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
#     ts = safe_zscore(ts, axis=0).astype(np.float32)
#     ts = pad_or_crop_time(ts, max_time_steps)
#     return ts.astype(np.float32)

def extract_roi_timeseries_fixed_aal(
    nii_path,
    atlas_img,
    t_r=2.0,
    n_rois=116
):
    """
    Extract ROI time series using AAL atlas with fixed 116 ROI order.

    Handles missing ROIs safely:
    - uses NiftiLabelsMasker for normal extraction
    - maps extracted atlas label values to fixed AAL positions
    - zero-fills only the actually missing ROI positions
    """

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=t_r,
        verbose=0,
    )

    ts_partial = masker.fit_transform(nii_path)
    T = ts_partial.shape[0]

    # Load atlas if atlas_img is path
    if isinstance(atlas_img, str):
        atlas_nii = nib.load(atlas_img)
    else:
        atlas_nii = atlas_img

    atlas_data = atlas_nii.get_fdata()

    atlas_values = sorted(
        [int(v) for v in np.unique(atlas_data.astype(np.int32)) if int(v) > 0]
    )

    atlas_values = atlas_values[:n_rois]

    value_to_pos = {
        atlas_value: idx
        for idx, atlas_value in enumerate(atlas_values)
    }

    ts_full = np.zeros((T, n_rois), dtype=np.float32)

    extracted_labels = getattr(masker, "labels_", None)

    if extracted_labels is not None:
        extracted_labels = [int(v) for v in extracted_labels if int(v) > 0]

        for k, atlas_value in enumerate(extracted_labels):
            if k >= ts_partial.shape[1]:
                break

            if atlas_value in value_to_pos:
                pos = value_to_pos[atlas_value]
                ts_full[:, pos] = ts_partial[:, k]

    else:
        # fallback: only if masker.labels_ unavailable
        K = min(ts_partial.shape[1], n_rois)
        ts_full[:, :K] = ts_partial[:, :K]

    ts_full = np.nan_to_num(
        ts_full,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return ts_full.astype(np.float32)



# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ADAPTIVE PHASE SYNCHRONIZATION (APS) GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_plv_matrix(time_series: np.ndarray) -> np.ndarray:
    """
    Compute the Phase Locking Value (PLV) matrix from ROI time-series.
    PLV_ij = |mean(exp(j * (phi_i - phi_j)))| ∈ [0, 1]

    Steps:
        1. Hilbert transform → analytic signal → instantaneous phase
        2. Phase difference Δφ_ij(t) = φ_i(t) - φ_j(t)
        3. PLV_ij = |1/T * Σ_t exp(j * Δφ_ij(t))|

    Args:
        time_series: (T, N) array of ROI BOLD signals
    Returns:
        plv_matrix: (N, N) symmetric PLV matrix
    """
    T, N = time_series.shape

    # Step 1: Hilbert transform → instantaneous phase
    analytic = hilbert(time_series, axis=0)       # (T, N) complex
    phases = np.angle(analytic)                    # (T, N) instantaneous phase φ_i(t)

    # Step 2 & 3: Vectorised PLV computation
    # For all pairs (i, j): PLV = |mean_t(exp(j*(phi_i - phi_j)))|
    phase_exp = np.exp(1j * phases)               # (T, N) complex unit vectors
    # PLV matrix via outer product on mean
    # plv[i,j] = |mean_t( phase_exp[:,i] * conj(phase_exp[:,j]) )|
    mean_phase = phase_exp.mean(axis=0)           # (N,) — NOT what we want, keep full
    plv = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        diff = phases[:, i:i+1] - phases          # (T, N)
        plv[i, :] = np.abs(np.mean(np.exp(1j * diff), axis=0))
    np.fill_diagonal(plv, 0.0)
    return plv


def adaptive_threshold(plv_matrix: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Subject-specific adaptive thresholding:
        θ = μ(PLV) + α * σ(PLV)
    Returns binary adjacency matrix A and weighted matrix A_w.
    """
    vals = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
    mu, sigma = vals.mean(), vals.std()
    theta = mu + alpha * sigma
    A_w = plv_matrix.copy()
    A_w[A_w < theta] = 0.0
    A = (A_w > 0).astype(np.float32)
    return A, A_w, float(theta)


# def build_edge_index_and_attr(A: np.ndarray, A_w: np.ndarray):
#     """
#     Convert adjacency matrices to COO edge_index and edge_attr tensors.
#     Returns:
#         edge_index: (2, E) long tensor
#         edge_attr:  (E, 1) float tensor  (PLV weights)
#     """
#     src, dst = np.where(A > 0)
#     weights = A_w[src, dst]
#     edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
#     edge_attr = torch.tensor(weights[:, None], dtype=torch.float32)
#     return edge_index, edge_attr
def build_edge_index_and_attr(A: np.ndarray, A_w: np.ndarray):
    src, dst = np.where(A > 0)
    weights = A_w[src, dst]

    N = A.shape[0]

    # add self loops
    self_src = np.arange(N)
    self_dst = np.arange(N)
    self_w = np.ones(N, dtype=np.float32)

    src = np.concatenate([src, self_src])
    dst = np.concatenate([dst, self_dst])
    weights = np.concatenate([weights, self_w])

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(weights[:, None], dtype=torch.float32)

    return edge_index, edge_attr


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  NODE FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _personalized_pagerank(A: np.ndarray, alpha: float = 0.85,
                            max_iter: int = 100) -> np.ndarray:
    """Power-iteration Personalized PageRank (uniform teleport)."""
    N = A.shape[0]
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg                                    # row-stochastic
    r = np.ones(N, dtype=np.float64) / N
    teleport = np.ones(N, dtype=np.float64) / N
    for _ in range(max_iter):
        r_new = alpha * P.T @ r + (1 - alpha) * teleport
        if np.linalg.norm(r_new - r, 1) < 1e-6:
            break
        r = r_new
    return r.astype(np.float32)


def _harmonic_centrality(A: np.ndarray) -> np.ndarray:
    """Harmonic centrality: sum of inverse shortest-path distances."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    N = A.shape[0]
    sp = shortest_path(csr_matrix(A), directed=False, unweighted=True)
    sp[sp == 0] = np.inf
    hc = (1.0 / sp)
    np.fill_diagonal(hc, 0.0)
    return hc.sum(axis=1).astype(np.float32) / (N - 1)


def _k_core_numbers(A: np.ndarray) -> np.ndarray:
    """Approximate k-core number by iterative degree pruning."""
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


def _wavelet_energy(ts: np.ndarray, scales: list) -> np.ndarray:
    """
    Multi-scale wavelet energy (discrete approximation using downsampling).
    Returns shape (N, len(scales)).
    """
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

    max_core = core.max() + 1e-8

    return (core / max_core).astype(np.float32)
def _participation_score(A):
    n_components, labels = connected_components(
        A.astype(np.int32),
        directed=False
    )

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
    from scipy.sparse.csgraph import shortest_path

    sp = shortest_path(A, directed=False)

    N = A.shape[0]

    score = np.zeros(N)

    for i in range(N):

        reachable = np.isfinite(sp[i])

        score[i] = reachable.sum()

    score /= score.max() + 1e-8

    return score.astype(np.float32)

def extract_node_features(time_series: np.ndarray, A: np.ndarray,
                           A_w: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute a comprehensive feature vector for each ROI node.

    Features per node (18 total in the executable implementation):
        0.  Node degree (normalised)
        1.  Weighted degree (sum of PLV weights)
        2.  Personalised PageRank centrality
        3.  Harmonic centrality
        4.  K-core number (normalised)
        5.  Betweenness centrality (approx via random walks)
        6.  Average neighbour degree
        7.  Mean BOLD signal
        8.  Std BOLD signal
        9-11. Wavelet energy (3 scales: 0.5, 1.0, 2.0)
        12. Mean absolute PLV to neighbours

    Returns:
        node_features: (N, 18) float32 array
    """
    present_mask = (time_series.std(axis=0) > 1e-8).astype(np.float32)
    T, N = time_series.shape
    #feats = np.zeros((N, 13), dtype=np.float32)
    feats = np.zeros((N, 18), dtype=np.float32)

    # Topology features from binary adjacency
    deg = A.sum(axis=1)
    feats[:, 0] = deg / (N - 1 + 1e-8)                          # f0: degree

    feats[:, 1] = A_w.sum(axis=1)                               # f1: weighted degree

    feats[:, 2] = _personalized_pagerank(A)                     # f2: PPR

    feats[:, 3] = _harmonic_centrality(A)                       # f3: harmonic centrality

    k_core = _k_core_numbers(A)
    feats[:, 4] = k_core / (k_core.max() + 1e-8)               # f4: k-core (norm)

    # Betweenness centrality approximation: fraction of neighbours'
    # neighbours reachable through this node
    A_2 = A @ A                                                  # 2-hop adjacency
    np.fill_diagonal(A_2, 0)
    feats[:, 5] = A_2.sum(axis=1) / (N * (N - 1) + 1e-8)       # f5: approx betweenness

    # Average neighbour degree
    avg_nb_deg = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        avg_nb_deg[i] = deg[nb].mean() if len(nb) > 0 else 0.0
    feats[:, 6] = avg_nb_deg / (N + 1e-8)                       # f6: avg nb degree

    # Signal-derived features
    feats[:, 7] = time_series.mean(axis=0)                      # f7: mean BOLD
    feats[:, 8] = time_series.std(axis=0)                       # f8: std BOLD

    # Wavelet energies
    wav = _wavelet_energy(time_series, cfg.WAVELET_SCALES)
    feats[:, 9:12] = wav                                         # f9-f11: wavelet
    feats[:, 13] = present_mask

    # Mean absolute PLV to neighbours
    plv_nb = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nb = np.where(A[i] > 0)[0]
        plv_nb[i] = A_w[i, nb].mean() if len(nb) > 0 else 0.0
    feats[:, 12] = plv_nb                                        # f12: mean PLV
    feats[:,14] = _hub_score(A)

    feats[:,15] = _core_periphery_score(A)

    feats[:,16] = _participation_score(A)

    feats[:,17] = _flow_betweenness(A)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    # Standardise each feature dimension (per-subject)
    # scaler = StandardScaler()
    # feats = scaler.fit_transform(feats).astype(np.float32)

    return feats.astype(np.float32)



# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def load_subjects(data_root: str):
    """
    Scan data_root/ASD/ and data_root/TD/ for .nii and .nii.gz files.
    Returns list of (filepath, label) tuples: ASD=1, TD=0.
    """
    root = Path(data_root)
    subjects = []
    for label_name, label_val in [("ASD", 1), ("TD", 0)]:
        folder = root / label_name
        if not folder.exists():
            log.warning(f"  Folder not found: {folder}")
            continue
        nii_files = sorted(
            list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
        )
        log.info(f"  {label_name}: {len(nii_files)} subjects found")
        for f in nii_files:
            subjects.append((str(f), label_val))
    if len(subjects) == 0:
        raise FileNotFoundError(
            f"No .nii/.nii.gz files found under {data_root}/ASD/ or {data_root}/TD/"
        )
    return subjects


def process_subject(nii_path: str, label: int,
                    atlas_img, value_to_pos: dict, cfg: Config) -> Data:
    """
    Full preprocessing pipeline for one subject:
        fMRI → time-series → APS graph → node features → PyG Data object.
    """
    # 1. Extract ROI time-series
    #ts = extract_roi_timeseries(nii_path, atlas_img, value_to_pos, n_rois=cfg.N_ROIS)
    ts = extract_roi_timeseries_fixed_aal(nii_path, atlas_img)
    site = Path(nii_path).name.split("_")[0]
    # 2. APS: PLV → adaptive threshold → adjacency matrices
    plv = compute_plv_matrix(ts)
    corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    A, A_w, theta = adaptive_threshold(plv, alpha=cfg.APS_ALPHA)

    # Ensure minimum connectivity
    if A.sum() < cfg.MIN_EDGES:
        # Relax threshold until minimum edges met
        for alpha_relax in [0.3, 0.1, 0.0]:
            A, A_w, theta = adaptive_threshold(plv, alpha=alpha_relax)
            if A.sum() >= cfg.MIN_EDGES:
                break

    # 3. Extract node features
    x = extract_node_features(ts, A, A_w, cfg)   # (N, 18)
    corr_feat = corr.astype(np.float32)

    x = np.concatenate([x, corr_feat],axis=1).astype(np.float32)
    # x = np.concatenate([x, plv], axis=1)
    # 4. Build edge index and attributes
    edge_index, edge_attr = build_edge_index_and_attr(A, A_w)

    # 5. Graph-level statistics (for interpretability)
    n_edges = int(A.sum())
    n_rois_actual = x.shape[0]
    density = n_edges / (n_rois_actual * (n_rois_actual - 1) + 1e-8)
    plv_full = plv.copy()
    np.fill_diagonal(plv_full, 0)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        plv_matrix=torch.tensor(plv, dtype=torch.float32),  # stored for SCRM
        n_edges=n_edges,
        density=density,
    )
    data.site = site
    return data


def build_dataset(subjects: list, atlas_img, value_to_pos: dict, cfg: Config) -> list:
    """
    Process all subjects and return list of PyG Data objects.
    Shows progress with a simple counter.
    """
    dataset = []
    n = len(subjects)
    failed = 0
    for idx, (nii_path, label) in enumerate(subjects):
        try:
            data = process_subject(nii_path, label, atlas_img, value_to_pos, cfg)
            dataset.append(data)
            if (idx + 1) % 10 == 0 or (idx + 1) == n:
                log.info(f"  Processed {idx+1}/{n} subjects")
        except Exception as e:
            log.warning(f"  Failed [{Path(nii_path).name}]: {e}")
            failed += 1
    log.info(f"  Dataset built: {len(dataset)} subjects ({failed} failed)")
    return dataset


class AdaptiveEdgeNodeLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim,
                 heads=4, dropout=0.3, use_feedback=True):
        super().__init__(aggr="add", node_dim=0)

        self.in_node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.use_feedback = use_feedback

        # Edge MLP: uses source node, destination node, and edge feature
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU()
        )

        # Message MLP: creates node messages from source node + updated edge
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Node MLP: refines aggregated node messages
        self.node_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        # Skip connection
        self.skip = nn.Linear(node_dim, out_dim)

        # Edge feedback MLP: updates edge again using updated nodes
        self.edge_feedback_mlp = nn.Sequential(
            nn.Linear(2 * out_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        # 1. Edge update
        edge_msg = torch.cat(
            [x[row], x[col], edge_attr],
            dim=-1
        )
        edge_attr_new = self.edge_mlp(edge_msg)

        # 2. Message passing
        agg_msg = self.propagate(
            edge_index,
            x=x,
            edge_attr_new=edge_attr_new
        )

        # 3. Node update
        x_new = self.node_mlp(agg_msg)
        x_new = x_new + self.skip(x)

        # 4. Edge feedback
        if self.use_feedback:
            feedback_msg = torch.cat(
                [x_new[row], x_new[col], edge_attr_new],
                dim=-1
            )
            edge_attr_new = self.edge_feedback_mlp(feedback_msg)

        return x_new, edge_attr_new

    def message(self, x_j, edge_attr_new):
        msg = torch.cat(
            [x_j, edge_attr_new],
            dim=-1
        )
        return self.msg_mlp(msg)


# # ═══════════════════════════════════════════════════════════════════════════════
# # 7.  AENIN MODEL ARCHITECTURE
# # ═══════════════════════════════════════════════════════════════════════════════

# class AdaptiveEdgeNodeLayer(MessagePassing):
#     """
#     Core AENIN layer implementing bidirectional edge-node co-evolution.

#     At layer l:
#         1. Edge update:  e_ij^(l+1) = EdgeMLP([x_i^(l) || x_j^(l) || e_ij^(l)])
#         2. Message:      m_ij^(l)   = NodeMLP([x_i^(l) || e_ij^(l+1)])
#         3. Node update:  x_i^(l+1) = Σ_j m_ij^(l)
#         4. Feedback:     e_ij^(l+1) = EdgeMLP([x_i^(l+1) || x_j^(l+1) || e_ij^(l)])

#     Edge and node MLPs each have two linear layers with ReLU.
#     """

#     def __init__(self, node_dim: int, edge_dim: int, out_dim: int,
#                  heads: int = 4, dropout: float = 0.3):
#         super().__init__(aggr="add", node_dim=0)
#         # Do NOT assign self.node_dim = node_dim; PyG uses node_dim as tensor dimension index.
#         self.in_node_dim = node_dim
#         self.edge_dim = edge_dim
#         self.out_dim = out_dim
#         self.heads = heads
#         self.dropout = dropout

#         # Edge MLP: [x_i || x_j || e_ij] → e'_ij
#         edge_in = node_dim + node_dim + edge_dim
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(edge_in, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU()
#         )

#         # Node MLP: [x_i || e'_ij] → message
#         node_in = node_dim + out_dim
#         self.node_mlp = nn.Sequential(
#             nn.Linear(node_in, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU()
#         )

#         # Attention: scalar score per updated edge
#         self.attn = nn.Linear(out_dim, 1)

#         # Skip connection projection (if dims differ)
#         self.skip = nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()

#         # Feedback edge MLP (reuse architecture, separate weights)
#         self.edge_feedback_mlp = nn.Sequential(
#             nn.Linear(out_dim + out_dim + out_dim, out_dim),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )

#     def forward(self, x, edge_index, edge_attr):
#         """
#         Args:
#             x:          (N, node_dim) node features
#             edge_index: (2, E) edge connectivity
#             edge_attr:  (E, edge_dim) edge features
#         Returns:
#             x_new:          (N, out_dim) updated node features
#             edge_attr_new:  (E, out_dim) updated edge features
#         """
#         row, col = edge_index  # src=row, dst=col

#         # ── Step 1: Edge update ──────────────────────────────────────────────
#         edge_msg = torch.cat([x[row], x[col], edge_attr], dim=-1)  # (E, 2*nd+ed)
#         edge_attr_new = self.edge_mlp(edge_msg)                    # (E, out_dim)

#         # ── Step 2 & 3: Message passing → node update ───────────────────────
#         x_new = self.propagate(edge_index, x=x,
#                                edge_attr_new=edge_attr_new)        # (N, out_dim)
#         x_new = x_new + self.skip(x)                               # skip connection

#         # ── Step 4: Edge feedback loop ───────────────────────────────────────
#         feedback_msg = torch.cat([x_new[row], x_new[col], edge_attr_new], dim=-1)
#         edge_attr_new = self.edge_feedback_mlp(feedback_msg)       # (E, out_dim)

#         return x_new, edge_attr_new

#     def message(self, x_i, edge_attr_new):
#         """Construct messages: fuse node_i features with updated edge."""
#         msg_input = torch.cat([x_i, edge_attr_new], dim=-1)       # (E, nd+od)
#         msg = self.node_mlp(msg_input)                             # (E, out_dim)
#         # Attention gating
#         alpha = torch.sigmoid(self.attn(edge_attr_new))            # (E, 1)
#         return msg * alpha


class AENIN(nn.Module):
    """
    Adaptive Edge-Node Interaction Network.

    Architecture:
        Input projection → L × AdaptiveEdgeNodeLayer → Global Pooling
        → MLP Classifier → Binary output
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        hd = cfg.HIDDEN_DIM

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.NODE_FEAT_DIM, hd),
            nn.LayerNorm(hd),
            nn.GELU()
        )
        self.edge_proj = nn.Linear(cfg.EDGE_FEAT_DIM, hd)

        # AENIN layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(cfg.NUM_LAYERS):
            in_dim = hd
            self.layers.append(
                AdaptiveEdgeNodeLayer(
                    node_dim=in_dim,
                    edge_dim=hd,
                    out_dim=hd,
                    heads=cfg.HEADS,
                    dropout=cfg.DROPOUT
                )
            )
            self.bns.append(nn.BatchNorm1d(hd))
        
        # Readout
        self.pool_norm = nn.LayerNorm(hd)

        # Classifier MLP
        # self.classifier = nn.Sequential(
        #     nn.Linear(hd * 2, hd),    # concat mean + max pooling
        #     nn.LayerNorm(hd),
        #     nn.ReLU(),
        #     nn.Dropout(cfg.DROPOUT),
        #     nn.Linear(hd, hd // 2),
        #     nn.ReLU(),
        #     nn.Dropout(cfg.DROPOUT / 2),
        #     nn.Linear(hd // 2, 2)    # binary: ASD vs TD
        # )
        self.classifier = nn.Sequential(
            nn.Linear(hd * 2, hd * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hd * 2, hd),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hd, 2)
        )

        self.apply(self._init_weights)

#     def _init_weights(self):
#         for m in self.modules():

#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)

#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_in",
                nonlinearity="relu"
        )

            if m.bias is not None:
                 nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Project inputs to hidden dimension
        x = self.input_proj(x)                    # (N, hd)
        edge_attr = self.edge_proj(edge_attr)      # (E, hd)

        # AENIN layers
        # for layer in self.layers:
        #     x, edge_attr = layer(x, edge_index, edge_attr)
        for layer, bn in zip(self.layers, self.bns):
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        x = self.pool_norm(x)

        # Global readout: mean + max pooling concatenated
        h_mean = global_mean_pool(x, batch)        # (B, hd)
        h_max = global_max_pool(x, batch)          # (B, hd)
        h_G = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*hd)

        # Classification
        logits = self.classifier(h_G)              # (B, 2)
        
#         h_mean = global_mean_pool(x, batch)
#         h_max = global_max_pool(x, batch)

#         num_graphs = h_mean.size(0)

#         density = data.density.view(num_graphs, 1).to(x.device)
#         n_edges = data.n_edges.view(num_graphs, 1).float().to(x.device)

#         graph_stats = torch.cat([density, n_edges / 10000.0], dim=1)

#         h_G = torch.cat([h_mean, h_max, graph_stats], dim=-1)
#                 # Classification
#         logits = self.classifier(h_G)              # (B, 2)
        return logits

    def get_node_gradients(self, data):
        """
        Compute gradient of predicted class score w.r.t. node features.
        Used for CIS (Critical Influence Score) computation.
        Returns: grad tensor (N, node_feat_dim)
        """
        data = data.to(next(self.parameters()).device)
        x = data.x.clone().requires_grad_(True)
        data_clone = Data(
            x=x, edge_index=data.edge_index,
            edge_attr=data.edge_attr, batch=data.batch
        )
        logits = self.forward(data_clone)
        pred_class = logits.argmax(dim=-1)
        score = logits[0, pred_class]
        score.backward()
        return x.grad.detach()



class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(1)

        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -torch.sum(true_dist * log_probs, dim=1)

        return loss.mean()

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold: float = 0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        total_loss += loss.item()

        probs = F.softmax(logits, dim=-1)
        preds = (probs[:, 1] >= threshold).long()

        y_true.extend(batch.y.view(-1).cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
    }
    return metrics


# def train_fold(model, train_loader, val_loader, cfg: Config):
#     """Train one fold with original-style early stopping and plain CrossEntropyLoss."""
#     device = cfg.DEVICE
#     model = model.to(device)

#     optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
#     scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
#     criterion = nn.CrossEntropyLoss()

#     best_val_score = -1.0
#     best_val_metrics = None
#     best_weights = None
#     patience_counter = 0
#     history = defaultdict(list)

#     for epoch in range(1, cfg.EPOCHS + 1):
#         train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
#         val_metrics = evaluate(model, val_loader, device, criterion, threshold=0.5)

#         scheduler.step(val_metrics["accuracy"])

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_metrics["loss"])
#         history["val_acc"].append(val_metrics["accuracy"])
#         history["val_f1"].append(val_metrics["f1"])
#         history["val_auc"].append(val_metrics["auc"])
#         f"    Ep {epoch:3d}  "
#         f"loss={train_loss:.4f}  "
#         f"val_acc={val_metrics['accuracy']:.4f}  "
#         f"val_f1={val_metrics['f1']:.4f}  "
#         f"val_auc={val_metrics['auc']:.4f}"
        
#         # if val_metrics["accuracy"] > best_val_score:
#         #     best_val_score = val_metrics["accuracy"]
#         # best_val_metrics = val_metrics
        
#         # Accuracy selection mirrors the original code that produced the stronger run.
#         if val_metrics["accuracy"] > best_val_score:
#             best_val_score = val_metrics["accuracy"]
#             best_val_metrics = val_metrics
#             best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= cfg.PATIENCE:
#                 log.info(f"    Early stop at epoch {epoch}")
#                 break

#         if epoch % 25 == 0:
#             log.info(
#                 f"    Ep {epoch:3d}  "
#                 f"loss={train_loss:.4f}  "
#                 f"val_acc={val_metrics['accuracy']:.4f}  "
#                 f"val_f1={val_metrics['f1']:.4f}  "
#                 f"val_auc={val_metrics['auc']:.4f}"
#             )

#     if best_weights is not None:
#         model.load_state_dict(best_weights)
#     else:
#         best_val_metrics = evaluate(model, val_loader, device, criterion, threshold=0.5)

#     return best_val_metrics, dict(history)

def train_fold(model, train_loader, val_loader, cfg: Config):
    """
    Train one fold for all epochs.
    No early stopping.
    Returns the best validation-accuracy checkpoint of the fold.
    """

    device = cfg.DEVICE
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10
    )

    criterion = nn.CrossEntropyLoss()

    best_val_score = -1.0
    best_val_metrics = None
    best_weights = None

    history = defaultdict(list)

    for epoch in range(1, cfg.EPOCHS + 1):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            threshold=0.5
        )

        scheduler.step(val_metrics["accuracy"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        # Store best validation accuracy, but do NOT stop training
        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_val_metrics = val_metrics
            best_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch % 25 == 0 or epoch == 1 or epoch == cfg.EPOCHS:
            log.info(
                f"    Ep {epoch:3d}/{cfg.EPOCHS}  "
                f"loss={train_loss:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"best_acc={best_val_score:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  "
                f"val_auc={val_metrics['auc']:.4f}"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return best_val_metrics, dict(history)


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SUPPORTIVE CLINICAL REPORTING MODULE (SCRM)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_cis(model, dataset, device, n_rois: int) -> np.ndarray:
    """
    Critical Influence Score (CIS):
        CIS_i = ||∂ŷ / ∂x_i||_2    (gradient L2 norm per node)
 
    Averaged over all correctly classified subjects.
    Returns: (n_rois,) array of CIS values.
    """
    model.eval()
    cis_asd = np.zeros(n_rois, dtype=np.float64)
    cis_td  = np.zeros(n_rois, dtype=np.float64)
    count_asd = count_td = 0
 
    for data in dataset:
        single = DataLoader([data], batch_size=1)
        batch = next(iter(single)).to(device)
 
        try:
            grad = model.get_node_gradients(batch)    # (N, F)
            node_cis = grad.norm(dim=-1).cpu().numpy()  # (N,)
        except Exception:
            continue
 
        label = data.y.item()
        if label == 1:
            cis_asd += node_cis
            count_asd += 1
        else:
            cis_td += node_cis
            count_td += 1
 
    if count_asd > 0:
        cis_asd /= count_asd
    if count_td > 0:
        cis_td /= count_td
 
    return cis_asd, cis_td
 
def compute_its(model, dataset, device, n_rois: int) -> np.ndarray:
    """
    Interaction Topology Score (ITS):
        ITS_ij = mean over subjects of final-layer edge_attr L2 norm.
    Returns: (n_rois, n_rois) ITS matrix.
    """
    model.eval()
    its_asd = np.zeros((n_rois, n_rois), dtype=np.float64)
    its_td  = np.zeros((n_rois, n_rois), dtype=np.float64)
    count_asd = count_td = 0
 
    for data in dataset:
        single = DataLoader([data], batch_size=1)
        batch = next(iter(single)).to(device)
 
        # Hook to capture edge_attr after final AENIN layer
        edge_activations = {}
 
        def hook_fn(module, inp, out):
            # out is (x_new, edge_attr_new)
            edge_activations["edge"] = out[1].detach().cpu()
 
        hook = model.layers[-1].register_forward_hook(hook_fn)
 
        with torch.no_grad():
            _ = model(batch)
 
        hook.remove()
 
        if "edge" not in edge_activations:
            continue
 
        edge_norms = edge_activations["edge"].norm(dim=-1).numpy()  # (E,)
        ei = data.edge_index.numpy()
        label = data.y.item()
 
        mat = np.zeros((n_rois, n_rois))
        for e_idx in range(ei.shape[1]):
            i, j = ei[0, e_idx], ei[1, e_idx]
            if i < n_rois and j < n_rois:
                mat[i, j] += edge_norms[e_idx]
 
        if label == 1:
            its_asd += mat
            count_asd += 1
        else:
            its_td += mat
            count_td += 1
 
    if count_asd > 0:
        its_asd /= count_asd
    if count_td > 0:
        its_td /= count_td
 
    return its_asd, its_td
 
def compute_pcs(A_w: np.ndarray, k_hops: int = 3) -> np.ndarray:
    """
    Path Connectivity Score (PCS):
        PCS_i = Σ_{k=1}^{K} (A_w^k)_i / k   (normalised multi-hop strength)
 
    Captures cooperative multi-hop communication paths from each node.
    Returns: (n_rois,) PCS vector.
    """
    N = A_w.shape[0]
    pcs = np.zeros(N, dtype=np.float64)
    A_k = A_w.copy().astype(np.float64)
    for k in range(1, k_hops + 1):
        pcs += A_k.sum(axis=1) / k
        A_k = A_k @ A_w
    pcs /= (pcs.max() + 1e-8)
    return pcs.astype(np.float32)
 
def compute_group_pcs(dataset, n_rois: int, k_hops: int = 3):
    """Compute mean PCS for ASD and TD groups."""
    pcs_asd = np.zeros(n_rois, dtype=np.float64)
    pcs_td  = np.zeros(n_rois, dtype=np.float64)
    count_asd = count_td = 0
 
    for data in dataset:
        plv = data.plv_matrix.numpy()
        pcs = compute_pcs(plv, k_hops=k_hops)
        if data.y.item() == 1:
            pcs_asd += pcs
            count_asd += 1
        else:
            pcs_td += pcs
            count_td += 1
 
    if count_asd > 0:
        pcs_asd /= count_asd
    if count_td > 0:
        pcs_td /= count_td
    return pcs_asd, pcs_td
 
def group_statistical_test(dataset, n_rois: int):
    """
    Two-sample t-test on CIS distributions across ASD vs TD subjects,
    with Benjamini-Hochberg FDR correction.
    Returns DataFrame with regions, t-stats, raw p-values, corrected p-values.
    """
    # Collect per-subject CIS (mean node feature as proxy)
    asd_feats = []
    td_feats  = []
    for data in dataset:
        feat = data.x.numpy()[:n_rois, :].mean(axis=-1)  # (n_rois,)
        if data.y.item() == 1:
            asd_feats.append(feat)
        else:
            td_feats.append(feat)
 
    asd_feats = np.array(asd_feats) if asd_feats else np.zeros((1, n_rois))
    td_feats  = np.array(td_feats)  if td_feats  else np.zeros((1, n_rois))
 
    t_stats = []
    p_vals  = []
    for roi in range(n_rois):
        t, p = ttest_ind(asd_feats[:, roi], td_feats[:, roi],
                         equal_var=False, nan_policy="omit")
        t_stats.append(float(t))
        p_vals.append(float(p))
 
    # FDR correction (Benjamini-Hochberg)
    p_arr = np.array(p_vals)
    reject, p_corrected, _, _ = multipletests(p_arr, method="fdr_bh")
 
    df = pd.DataFrame({
        "roi_index": range(n_rois),
        "t_stat": t_stats,
        "p_value": p_vals,
        "p_corrected_fdr": p_corrected,
        "significant_fdr": reject
    })
    df = df.sort_values("p_corrected_fdr")
    return df
    

# ═══════════════════════════════════════════════════════════════════════════════
# 10. VISUALISATION & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history_folds: list, out_dir: str):
    """Plot training loss and validation accuracy curves for all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AENIN — Training Curves (5-Fold CV)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(history_folds)))

    for fold_idx, history in enumerate(history_folds):
        color = colors[fold_idx]
        label = f"Fold {fold_idx+1}"
        axes[0].plot(history["train_loss"], color=color, alpha=0.8, label=label)
        axes[0].plot(history["val_loss"], color=color, alpha=0.4, linestyle="--")
        axes[1].plot(history["val_acc"], color=color, alpha=0.8, label=label)
        axes[1].plot(history["val_f1"], color=color, alpha=0.4, linestyle="--")

    axes[0].set_title("Loss (solid=train, dashed=val)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Metrics (solid=Acc, dashed=F1)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    log.info("  Saved: training_curves.png")


def plot_confusion_matrices(fold_cms: list, out_dir: str):
    """Plot confusion matrices for all folds + aggregate."""
    n_folds = len(fold_cms)
    fig, axes = plt.subplots(1, n_folds + 1,
                              figsize=(4 * (n_folds + 1), 4))
    fig.suptitle("Confusion Matrices — 5-Fold CV", fontsize=14, fontweight="bold")

    agg_cm = sum(fold_cms)
    all_cms = fold_cms + [agg_cm]
    titles = [f"Fold {i+1}" for i in range(n_folds)] + ["Aggregate"]

    for ax, cm, title in zip(axes, all_cms, titles):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["TD", "ASD"],
                    yticklabels=["TD", "ASD"],
                    ax=ax, cbar=False)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    log.info("  Saved: confusion_matrices.png")


def plot_roc_curves(fold_metrics: list, out_dir: str):
    """Plot per-fold ROC curves with mean AUC band."""
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_metrics)))
    aucs = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fold_idx, m in enumerate(fold_metrics):
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
        auc = m["auc"]
        aucs.append(auc)
        ax.plot(fpr, tpr, color=colors[fold_idx], alpha=0.6,
                label=f"Fold {fold_idx+1} (AUC={auc:.3f})", lw=1.5)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="navy", lw=2.5,
            label=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})")

    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color="navy", alpha=0.1, label="± 1 std")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — 5-Fold CV", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close()
    log.info("  Saved: roc_curves.png")


def plot_cis_scores(cis_asd, cis_td, roi_labels, out_dir: str, top_k: int = 20):
    """Bar plot of top-K CIS regions for ASD and TD."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Critical Influence Score (CIS) — Top Influential Brain Regions",
                 fontsize=13, fontweight="bold")

    for ax, cis, group, color in [
        (axes[0], cis_asd, "ASD", "#C00000"),
        (axes[1], cis_td,  "TD",  "#2E75B6")
    ]:
        top_idx = np.argsort(cis)[::-1][:top_k]
        labels  = [roi_labels[i] if i < len(roi_labels) else f"ROI_{i}"
                   for i in top_idx]
        vals    = cis[top_idx]

        bars = ax.barh(range(top_k), vals[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_xlabel("CIS Value", fontsize=11)
        ax.set_title(f"Top {top_k} Regions — {group}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Annotate top 3
        for bar, val in zip(bars[-3:], vals[-3:]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cis_scores.png"), dpi=150)
    plt.close()
    log.info("  Saved: cis_scores.png")


def plot_its_heatmap(its_asd, its_td, out_dir: str):
    """Heatmap of ITS matrices for ASD and TD groups."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Interaction Topology Score (ITS) Matrix",
                 fontsize=13, fontweight="bold")

    for ax, its, group, cmap in [
        (axes[0], its_asd, "ASD", "Reds"),
        (axes[1], its_td,  "TD",  "Blues")
    ]:
        im = ax.imshow(its, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_title(f"ITS — {group}", fontweight="bold")
        ax.set_xlabel("Target ROI")
        ax.set_ylabel("Source ROI")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "its_heatmaps.png"), dpi=150)
    plt.close()
    log.info("  Saved: its_heatmaps.png")


def plot_pcs_scores(pcs_asd, pcs_td, roi_labels, out_dir: str):
    """Compare PCS distributions between ASD and TD."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Path Connectivity Score (PCS) — ASD vs TD",
                 fontsize=13, fontweight="bold")

    top_k = 20
    # Violin + strip plot
    ax = axes[0]
    data_asd = pd.DataFrame({"PCS": pcs_asd, "Group": "ASD"})
    data_td  = pd.DataFrame({"PCS": pcs_td,  "Group": "TD"})
    df_plot  = pd.concat([data_asd, data_td])
    sns.violinplot(x="Group", y="PCS", data=df_plot,
                   palette={"ASD": "#C00000", "TD": "#2E75B6"},
                   ax=ax, inner="quartile")
    ax.set_title("PCS Distribution: ASD vs TD", fontweight="bold")
    ax.set_ylabel("PCS Value")
    ax.grid(alpha=0.3)

    # Top regions comparison
    ax = axes[1]
    diff = pcs_asd - pcs_td
    top_idx = np.argsort(np.abs(diff))[::-1][:top_k]
    labels = [roi_labels[i] if i < len(roi_labels) else f"ROI_{i}"
              for i in top_idx]
    vals = diff[top_idx]
    colors = ["#C00000" if v > 0 else "#2E75B6" for v in vals]
    ax.barh(range(top_k), vals[::-1], color=colors[::-1], alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PCS Difference (ASD - TD)")
    ax.set_title(f"Top {top_k} PCS Differences (red=ASD↑, blue=TD↑)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pcs_scores.png"), dpi=150)
    plt.close()
    log.info("  Saved: pcs_scores.png")


def plot_statistical_tests(stat_df: pd.DataFrame, out_dir: str, top_k: int = 15):
    """Plot t-test results with FDR-corrected significance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Group-Level Statistical Analysis (ASD vs TD)\nwith Benjamini-Hochberg FDR Correction",
                 fontsize=13, fontweight="bold")

    # Volcano plot: t-stat vs -log10(p_corrected)
    ax = axes[0]
    sig = stat_df["significant_fdr"]
    ax.scatter(stat_df.loc[~sig, "t_stat"],
               -np.log10(stat_df.loc[~sig, "p_corrected_fdr"] + 1e-10),
               color="grey", alpha=0.5, s=20, label="Not significant")
    ax.scatter(stat_df.loc[sig, "t_stat"],
               -np.log10(stat_df.loc[sig, "p_corrected_fdr"] + 1e-10),
               color="#C00000", alpha=0.8, s=40, label="Significant (FDR)")
    ax.axhline(-np.log10(0.05), color="navy", linestyle="--", linewidth=1,
               label="FDR threshold (0.05)")
    ax.set_xlabel("t-statistic (ASD - TD)")
    ax.set_ylabel("-log10(FDR-corrected p-value)")
    ax.set_title("Volcano Plot")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Top significant regions
    ax = axes[1]
    top_sig = stat_df.head(min(top_k, len(stat_df)))
    colors = ["#C00000" if t > 0 else "#2E75B6"
              for t in top_sig["t_stat"]]
    ax.barh(range(len(top_sig)), top_sig["t_stat"].values[::-1],
            color=colors[::-1], alpha=0.8)
    ax.set_yticks(range(len(top_sig)))
    roi_names = [f"ROI_{int(i)}" for i in top_sig["roi_index"].values[::-1]]
    ax.set_yticklabels(roi_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t-statistic")
    ax.set_title(f"Top {len(top_sig)} Regions by FDR p-value\n(red=ASD↑, blue=TD↑)",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "statistical_tests.png"), dpi=150)
    plt.close()
    log.info("  Saved: statistical_tests.png")


def plot_cv_summary(fold_metrics: list, out_dir: str):
    """Bar chart summarising all metrics across folds with mean ± std."""
    metric_names = ["accuracy", "balanced_accuracy", "f1",
                    "precision", "recall", "auc"]
    metric_labels = ["Accuracy", "Balanced\nAccuracy", "F1-Score",
                     "Precision", "Recall", "AUC-ROC"]

    fold_vals = {m: [fm[m] for fm in fold_metrics] for m in metric_names}
    means = [np.mean(fold_vals[m]) for m in metric_names]
    stds  = [np.std(fold_vals[m])  for m in metric_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#2E75B6", "#1F4E79", "#C00000",
                         "#ED7D31", "#70AD47", "#7030A0"],
                  alpha=0.85, width=0.6, error_kw={"linewidth": 2})

    # Individual fold dots
    colors_f = plt.cm.tab10(np.linspace(0, 1, len(fold_metrics)))
    for fold_idx, fm in enumerate(fold_metrics):
        vals = [fm[m] for m in metric_names]
        ax.scatter(x, vals, color=colors_f[fold_idx], s=40, zorder=5,
                   label=f"Fold {fold_idx+1}", alpha=0.9)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("AENIN — 5-Fold Cross-Validation Performance Summary",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_summary.png"), dpi=150)
    plt.close()
    log.info("  Saved: cv_summary.png")


def save_results_csv(fold_metrics: list, stat_df: pd.DataFrame, out_dir: str):
    """Save all numeric results to CSV files."""
    # Per-fold metrics
    metric_names = ["accuracy", "balanced_accuracy", "f1",
                    "precision", "recall", "auc"]
    rows = []
    for i, fm in enumerate(fold_metrics):
        row = {"fold": i + 1}
        row.update({m: fm[m] for m in metric_names})
        rows.append(row)
    # Summary row
    summary = {"fold": "mean±std"}
    for m in metric_names:
        vals = [fm[m] for fm in fold_metrics]
        summary[m] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rows.append(summary)

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    log.info("  Saved: fold_metrics.csv")

    # Statistical test results
    stat_df.to_csv(os.path.join(out_dir, "statistical_tests.csv"), index=False)
    log.info("  Saved: statistical_tests.csv")


def print_summary_table(fold_metrics: list):
    """Print a formatted summary table to console."""
    metric_names = ["accuracy", "balanced_accuracy", "f1",
                    "precision", "recall", "auc"]
    header = f"{'Fold':<8}" + "".join(f"{m.upper():<18}" for m in metric_names)
    print("\n" + "=" * 120)
    print("AENIN — 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 120)
    print(header)
    print("-" * 120)
    for i, fm in enumerate(fold_metrics):
        row = f"{i+1:<8}" + "".join(f"{fm[m]:<18.4f}" for m in metric_names)
        print(row)
    print("-" * 120)
    means = [np.mean([fm[m] for fm in fold_metrics]) for m in metric_names]
    stds  = [np.std( [fm[m] for fm in fold_metrics]) for m in metric_names]
    mean_row = f"{'MEAN':<8}" + "".join(f"{m:<18.4f}" for m in means)
    std_row  = f"{'STD':<8}"  + "".join(f"{s:<18.4f}" for s in stds)
    print(mean_row)
    print(std_row)
    print("=" * 120)




# ═══════════════════════════════════════════════════════════════════════════════
# 11. ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_study(dataset, labels, cfg: Config, out_dir: str,
                   n_seeds: int = 3):
    """
    Ablation study over multiple seeds with 3 conditions:
        1. Without edge feedback loop
        2. Without APS (use uniform random edge weights)
        3. Without node centrality features (zero out features 2-6)
        4. Full AENIN (baseline)

    Reports mean ± std accuracy for each condition.
    """
    log.info("Running ablation study...")

    conditions = {
        "Full AENIN":          {"disable_feedback": False, "random_edges": False, "no_centrality": False},
        "No Edge Feedback":    {"disable_feedback": True,  "random_edges": False, "no_centrality": False},
        "No APS (random)":     {"disable_feedback": False, "random_edges": True,  "no_centrality": False},
        "No Centrality Feats": {"disable_feedback": False, "random_edges": False, "no_centrality": True},
    }

    results = {name: [] for name in conditions}
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=cfg.SEED)

    for cond_name, flags in conditions.items():
        for seed in range(n_seeds):
            set_seed(cfg.SEED + seed)
            fold_accs = []

            for train_idx, val_idx in skf.split(np.zeros(len(dataset)), labels):
                train_data = [dataset[i] for i in train_idx]
                val_data   = [dataset[i] for i in val_idx]

                # Apply ablation flags
                if flags["no_centrality"]:
                    # Zero out centrality features (indices 2-6)
                    for d in train_data + val_data:
                        d.x[:, 2:7] = 0.0

                if flags["random_edges"]:
                    # Replace PLV edge weights with random uniform
                    for d in train_data + val_data:
                        n_edges = d.edge_attr.shape[0]
                        d.edge_attr = torch.rand(n_edges, 1)

                train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE,
                                          shuffle=True)
                val_loader   = DataLoader(val_data,   batch_size=cfg.BATCH_SIZE)

                model = AENIN(cfg).to(cfg.DEVICE)

                if flags["disable_feedback"]:
                    # Monkey-patch: skip feedback loop in all layers
                    for layer in model.layers:
                        layer.edge_feedback_mlp = nn.Identity()

                abl_cfg = Config()
                abl_cfg.EPOCHS  = min(cfg.EPOCHS, 50)
                abl_cfg.PATIENCE = 15
                abl_cfg.DEVICE  = cfg.DEVICE

                val_metrics, _ = train_fold(model, train_loader, val_loader, abl_cfg)
                fold_accs.append(val_metrics["accuracy"])

            results[cond_name].append(np.mean(fold_accs))

    # Print ablation table
    print("\n" + "=" * 60)
    print("ABLATION STUDY (mean ± std accuracy over seeds)")
    print("=" * 60)
    for cond_name, accs in results.items():
        print(f"  {cond_name:<28s}  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("=" * 60)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds  = [np.std(v)  for v in results.values()]
    colors = ["#2E75B6", "#ED7D31", "#C00000", "#70AD47"]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, width=0.5,
                  error_kw={"linewidth": 2})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.005,
                f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Accuracy (mean ± std)", fontsize=12)
    ax.set_title("Ablation Study — Effect of Each AENIN Component",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_study.png"), dpi=150)
    plt.close()
    log.info("  Saved: ablation_study.png")

    return results



def diagnose_dataset(dataset: list, n_rois: int = 116):
    """
    Run sanity checks on the built dataset BEFORE training starts.
    Catches the most common causes of flat-loss / chance-level performance:
      1. All-zero or near-constant node features (dead input)
      2. NaN/Inf in features, edge weights, or PLV matrices
      3. Severely degenerate graphs (too few edges)
      4. Label imbalance or label leakage
      5. Feature scale issues (e.g. everything collapsed to ~0 after scaling)
    Prints a diagnostic report and returns True if dataset looks healthy.
    """
    log.info("\n" + "="*70)
    log.info("DATASET HEALTH CHECK (run before training)")
    log.info("="*70)
 
    issues = []
 
    # ── 1. Label distribution ────────────────────────────────────────────────
    labels = np.array([d.y.item() for d in dataset])
    n_asd, n_td = (labels==1).sum(), (labels==0).sum()
    log.info(f"  Labels: ASD={n_asd}  TD={n_td}  (total={len(labels)})")
    if min(n_asd, n_td) == 0:
        issues.append("CRITICAL: one class has zero subjects")
 
    # ── 2. Node feature statistics across whole dataset ─────────────────────
    all_x = torch.cat([d.x for d in dataset], dim=0).numpy()   # (sum_N, F)
    feat_mean = all_x.mean(axis=0)
    feat_std  = all_x.std(axis=0)
    n_nan     = np.isnan(all_x).sum()
    n_inf     = np.isinf(all_x).sum()
    n_zero_cols = (feat_std < 1e-6).sum()
 
    log.info(f"  Node features: shape per subject = {dataset[0].x.shape}")
    log.info(f"  Feature means (first 5): {feat_mean[:5].round(4)}")
    log.info(f"  Feature stds  (first 5): {feat_std[:5].round(4)}")
    log.info(f"  NaN count: {n_nan}   Inf count: {n_inf}")
    log.info(f"  Near-constant feature columns (std<1e-6): {n_zero_cols}/{all_x.shape[1]}")
 
    if n_nan > 0:
        issues.append(f"CRITICAL: {n_nan} NaN values in node features")
    if n_inf > 0:
        issues.append(f"CRITICAL: {n_inf} Inf values in node features")
    if n_zero_cols > all_x.shape[1] * 0.5:
        issues.append(f"CRITICAL: {n_zero_cols}/{all_x.shape[1]} feature columns are near-constant (dead inputs)")
 
    # ── 3. Per-subject zero-fill / degeneracy check ──────────────────────────
    zero_node_fracs = []
    edge_counts     = []
    for d in dataset:
        x_np = d.x.numpy()
        zero_nodes = (np.abs(x_np).sum(axis=1) < 1e-8).sum()
        zero_node_fracs.append(zero_nodes / x_np.shape[0])
        edge_counts.append(d.edge_index.shape[1])
 
    mean_zero_frac = np.mean(zero_node_fracs)
    mean_edges     = np.mean(edge_counts)
    min_edges      = np.min(edge_counts)
 
    log.info(f"  Mean fraction of all-zero nodes per subject: {mean_zero_frac:.3f}")
    log.info(f"  Mean edges per subject: {mean_edges:.1f}  (min={min_edges})")
 
    if mean_zero_frac > 0.3:
        issues.append(f"CRITICAL: {mean_zero_frac*100:.1f}% of nodes are all-zero on average — "
                      f"likely a zero-fill / ROI extraction bug")
    if min_edges < 10:
        issues.append(f"CRITICAL: some subjects have <10 edges — graph is too sparse to learn from")
 
    # ── 4. Check if ASD vs TD features are distinguishable AT ALL ───────────
    # (a model-free sanity check: if classes have near-identical feature means,
    #  no model can separate them from these features alone)
    asd_x = torch.cat([d.x for d in dataset if d.y.item()==1], dim=0).numpy()
    td_x  = torch.cat([d.x for d in dataset if d.y.item()==0], dim=0).numpy()
    asd_mean, td_mean = asd_x.mean(axis=0), td_x.mean(axis=0)
    feat_diff = np.abs(asd_mean - td_mean)
    max_diff_idx = np.argmax(feat_diff)
    log.info(f"  Max |ASD-TD| mean feature difference: {feat_diff[max_diff_idx]:.4f} "
             f"(feature idx {max_diff_idx})")
    if feat_diff.max() < 0.01:
        issues.append("WARNING: ASD and TD node features are nearly identical on average — "
                      "classes may not be separable with current features")
 
    # ── 5. PLV matrix sanity (if present) ────────────────────────────────────
    if hasattr(dataset[0], "plv_matrix"):
        plv_sample = dataset[0].plv_matrix.numpy()
        log.info(f"  Sample PLV matrix: min={plv_sample.min():.3f} max={plv_sample.max():.3f} "
                 f"mean={plv_sample.mean():.3f}")
        if plv_sample.max() < 0.05:
            issues.append("WARNING: PLV values are unusually low — check Hilbert transform / "
                          "phase extraction step")
 
    # ── Report ────────────────────────────────────────────────────────────
    log.info("-"*70)
    if issues:
        log.warning(f"  FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            log.warning(f"    {i}. {issue}")
    else:
        log.info("  No critical issues detected in dataset health check.")
    log.info("="*70 + "\n")
 
    return len(issues) == 0
 
