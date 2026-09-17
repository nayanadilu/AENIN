"""
Run 5-fold CV and Site-wise CV for ALL baselines (+ your AENIN model) on
the SAME data, SAME splits, SAME features -> produces a fair, reviewer-proof
comparison table (accuracy, sensitivity, specificity, F1, AUC per model).

Usage:
    python train_compare.py --mode 5fold
    python train_compare.py --mode site

Plug in:
    1. `load_subjects()`      -> build your list-of-dict subjects (see dataset.py)
    2. `build_node_features`  -> your feature-extraction function (18-d feats)
    3. `AENIN`                -> import your own model class and register it
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import subprocess

from .models import MODEL_REGISTRY, DYNAMIC_MODELS
from .dataset_bridge import AENINDenseDataset, collate_fn, get_labels_and_sites


# >>> plug in your own AENIN model here (PyG-based; needs its own training
#     loop with torch_geometric.loader.DataLoader since it expects forward(data),
#     not forward(x, A) like the dense baselines). See note in run_5fold(). >>>
# from your_aenin_module import AENIN
# MODEL_REGISTRY["AENIN"] = AENIN
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



    # # ── Reproducibility
    # SEED: int = 42



DEVICE = pick_gpu_with_max_free_memory()


def train_one_fold(model_name, n_nodes, in_dim, num_classes,
                    train_ds, val_ds, epochs=60, lr=1e-3, batch_size=8):
    ModelCls = MODEL_REGISTRY[model_name]
    model = ModelCls(n_nodes=n_nodes, in_dim=in_dim, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    best_val_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x, A, ts, y = (batch["x"].to(DEVICE), batch["A"].to(DEVICE),
                            batch["ts"].to(DEVICE), batch["y"].to(DEVICE))
            opt.zero_grad()
            logits = model(x, A, ts) if model_name in DYNAMIC_MODELS else model(x, A)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # quick val check each epoch, keep best checkpoint
        val_acc, *_ = evaluate(model, model_name, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, val_loader


@torch.no_grad()
def evaluate(model, model_name, loader):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for batch in loader:
        x, A, ts, y = (batch["x"].to(DEVICE), batch["A"].to(DEVICE),
                        batch["ts"].to(DEVICE), batch["y"].to(DEVICE))
        logits = model(x, A, ts) if model_name in DYNAMIC_MODELS else model(x, A)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    sens = recall_score(all_y, all_pred, zero_division=0)               # recall = sensitivity
    spec = recall_score(all_y, all_pred, pos_label=0, zero_division=0)  # specificity
    try:
        auc = roc_auc_score(all_y, all_prob)
    except ValueError:
        auc = float("nan")
    return acc, f1, sens, spec, auc


def run_5fold(pyg_dataset, n_nodes, in_dim=18, num_classes=2, n_splits=5, epochs=60):
    labels, _sites = get_labels_and_sites(pyg_dataset)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    full_ds = AENINDenseDataset(pyg_dataset, n_rois=n_nodes)

    results = {name: [] for name in MODEL_REGISTRY}
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n===== Fold {fold + 1}/{n_splits} "
              f"(train={len(tr_idx)}, val={len(va_idx)}) =====")
        train_ds = Subset(full_ds, tr_idx)
        val_ds = Subset(full_ds, va_idx)
        for name in MODEL_REGISTRY:
            model, val_loader = train_one_fold(name, n_nodes, in_dim, num_classes,
                                                train_ds, val_ds, epochs=epochs)
            acc, f1, sens, spec, auc = evaluate(model, name, val_loader)
            results[name].append((acc, f1, sens, spec, auc))
            print(f"  {name:10s} | acc={acc:.4f} f1={f1:.4f} "
                  f"sens={sens:.4f} spec={spec:.4f} auc={auc:.4f}")

    print("\n===== 5-fold CV summary (mean ± std) =====")
    for name, vals in results.items():
        arr = np.array(vals)
        means, stds = arr.mean(axis=0), arr.std(axis=0)
        print(f"{name:10s} | "
              f"acc={means[0]:.4f}±{stds[0]:.4f} "
              f"f1={means[1]:.4f}±{stds[1]:.4f} "
              f"sens={means[2]:.4f}±{stds[2]:.4f} "
              f"spec={means[3]:.4f}±{stds[3]:.4f} "
              f"auc={means[4]:.4f}±{stds[4]:.4f}")
    return results


def run_sitewise(pyg_dataset, n_nodes, in_dim=18, num_classes=2, epochs=60):
    labels, sites = get_labels_and_sites(pyg_dataset)
    logo = LeaveOneGroupOut()
    full_ds = AENINDenseDataset(pyg_dataset, n_rois=n_nodes)

    results = {name: [] for name in MODEL_REGISTRY}
    for fold, (tr_idx, va_idx) in enumerate(logo.split(np.zeros(len(labels)), labels, sites)):
        held_out_site = sites[va_idx[0]]
        print(f"\n===== Held-out site: {held_out_site} "
              f"(train={len(tr_idx)}, val={len(va_idx)}) =====")
        train_ds = Subset(full_ds, tr_idx)
        val_ds = Subset(full_ds, va_idx)
        for name in MODEL_REGISTRY:
            model, val_loader = train_one_fold(name, n_nodes, in_dim, num_classes,
                                                train_ds, val_ds, epochs=epochs)
            acc, f1, sens, spec, auc = evaluate(model, name, val_loader)
            results[name].append((held_out_site, acc, f1, sens, spec, auc))
            print(f"  {name:10s} | acc={acc:.4f} f1={f1:.4f} "
                  f"sens={sens:.4f} spec={spec:.4f} auc={auc:.4f}")

    print("\n===== Site-wise CV summary (mean ± std across sites) =====")
    for name, vals in results.items():
        arr = np.array([v[1:] for v in vals], dtype=np.float32)
        means, stds = arr.mean(axis=0), arr.std(axis=0)
        print(f"{name:10s} | "
              f"acc={means[0]:.4f}±{stds[0]:.4f} "
              f"f1={means[1]:.4f}±{stds[1]:.4f} "
              f"sens={means[2]:.4f}±{stds[2]:.4f} "
              f"spec={means[3]:.4f}±{stds[3]:.4f} "
              f"auc={means[4]:.4f}±{stds[4]:.4f}")
    return results
