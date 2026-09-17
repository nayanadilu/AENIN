"""Command-line runner for the accepted AENIN implementation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from aenin import (
    AENIN,
    Config,
    ablation_study,
    build_dataset,
    compute_cis,
    compute_group_pcs,
    compute_its,
    diagnose_dataset,
    group_statistical_test,
    load_atlas,
    load_subjects,
    plot_cis_scores,
    plot_confusion_matrices,
    plot_cv_summary,
    plot_its_heatmap,
    plot_pcs_scores,
    plot_roc_curves,
    plot_statistical_tests,
    plot_training_curves,
    print_summary_table,
    save_results_csv,
    set_seed,
    train_fold,
)

log = logging.getLogger("AENIN-run")


def _balanced_subset(dataset, n_per_class: int, seed: int):
    asd = [d for d in dataset if int(d.y.item()) == 1]
    td = [d for d in dataset if int(d.y.item()) == 0]

    if len(asd) < n_per_class or len(td) < n_per_class:
        raise ValueError(
            f"Requested {n_per_class} subjects/class but found "
            f"ASD={len(asd)}, TD={len(td)}."
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(asd)
    rng.shuffle(td)
    out = asd[:n_per_class] + td[:n_per_class]
    rng.shuffle(out)
    return out


def _load_dataset(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def run(args):
    set_seed(Config.SEED)

    cfg = Config()
    cfg.DATA_ROOT = args.data_root
    cfg.OUTPUT_DIR = args.output_dir
    cfg.ATLAS = args.atlas
    cfg.EPOCHS = args.epochs
    cfg.HIDDEN_DIM = args.hidden
    cfg.BATCH_SIZE = args.batch_size
    cfg.APS_ALPHA = args.aps_alpha
    cfg.DEVICE = Config.pick_gpu_with_max_free_memory()

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_img, n_rois, roi_labels, value_to_pos = load_atlas(cfg.ATLAS)
    cfg.N_ROIS = n_rois
    roi_labels = list(roi_labels)

    if args.dataset_pt:
        log.info("Loading processed graph dataset: %s", args.dataset_pt)
        dataset = _load_dataset(args.dataset_pt)
    else:
        log.info("Loading NIfTI subjects from: %s", cfg.DATA_ROOT)
        subjects = load_subjects(cfg.DATA_ROOT)
        dataset = build_dataset(
            subjects,
            atlas_img,
            value_to_pos,
            cfg,
        )

    if not dataset:
        raise RuntimeError("Dataset is empty.")

    if args.balance_per_class:
        dataset = _balanced_subset(
            dataset,
            n_per_class=args.balance_per_class,
            seed=cfg.SEED,
        )

    if args.save_dataset:
        dataset_out = output_dir / "dataset.pt"
        torch.save(dataset, dataset_out)
        log.info("Saved processed dataset: %s", dataset_out)

    # The accepted notebook infers these dimensions from the graph objects.
    cfg.NODE_FEAT_DIM = int(dataset[0].x.shape[1])
    cfg.EDGE_FEAT_DIM = int(dataset[0].edge_attr.shape[1])

    labels = np.asarray([int(d.y.item()) for d in dataset], dtype=np.int64)
    n_asd = int((labels == 1).sum())
    n_td = int((labels == 0).sum())

    log.info(
        "Subjects=%d | ASD=%d | TD=%d | node_dim=%d | edge_dim=%d",
        len(dataset),
        n_asd,
        n_td,
        cfg.NODE_FEAT_DIM,
        cfg.EDGE_FEAT_DIM,
    )

    diagnose_dataset(dataset, n_rois=cfg.N_ROIS)

    skf = StratifiedKFold(
        n_splits=cfg.N_FOLDS,
        shuffle=True,
        random_state=cfg.SEED,
    )

    fold_metrics = []
    history_folds = []
    best_model_states = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(dataset)), labels),
        start=1,
    ):
        log.info(
            "Fold %d/%d | train=%d | val=%d",
            fold_idx,
            cfg.N_FOLDS,
            len(train_idx),
            len(val_idx),
        )

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        train_loader = DataLoader(
            train_data,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )

        model = AENIN(cfg).to(cfg.DEVICE)
        val_metrics, history = train_fold(
            model,
            train_loader,
            val_loader,
            cfg,
        )

        fold_metrics.append(val_metrics)
        history_folds.append(history)
        best_model_states.append(
            {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        )

        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary_table(fold_metrics)

    plot_training_curves(history_folds, cfg.OUTPUT_DIR)
    plot_confusion_matrices(
        [fm["confusion_matrix"] for fm in fold_metrics],
        cfg.OUTPUT_DIR,
    )
    plot_roc_curves(fold_metrics, cfg.OUTPUT_DIR)
    plot_cv_summary(fold_metrics, cfg.OUTPUT_DIR)

    # The accepted-study notebook selects the best fold by validation accuracy
    # before the SCRM analysis.
    best_fold_idx = int(
        np.argmax([fm["accuracy"] for fm in fold_metrics])
    )
    best_model = AENIN(cfg).to(cfg.DEVICE)
    best_model.load_state_dict(best_model_states[best_fold_idx])
    best_model.eval()

    log.info(
        "Running SCRM using best-accuracy model from fold %d",
        best_fold_idx + 1,
    )

    cis_asd, cis_td = compute_cis(
        best_model,
        dataset,
        cfg.DEVICE,
        n_rois,
    )
    plot_cis_scores(
        cis_asd,
        cis_td,
        roi_labels,
        cfg.OUTPUT_DIR,
    )

    its_asd, its_td = compute_its(
        best_model,
        dataset,
        cfg.DEVICE,
        n_rois,
    )
    plot_its_heatmap(
        its_asd,
        its_td,
        cfg.OUTPUT_DIR,
    )

    pcs_asd, pcs_td = compute_group_pcs(
        dataset,
        n_rois,
    )
    plot_pcs_scores(
        pcs_asd,
        pcs_td,
        roi_labels,
        cfg.OUTPUT_DIR,
    )

    stat_df = group_statistical_test(
        dataset,
        n_rois,
    )
    plot_statistical_tests(
        stat_df,
        cfg.OUTPUT_DIR,
    )

    if not args.skip_ablation:
        ablation_study(
            dataset,
            labels,
            cfg,
            cfg.OUTPUT_DIR,
            n_seeds=args.ablation_seeds,
        )

    save_results_csv(
        fold_metrics,
        stat_df,
        cfg.OUTPUT_DIR,
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    summary = {
        "model": "AENIN",
        "n_subjects": len(dataset),
        "n_asd": n_asd,
        "n_td": n_td,
        "atlas": cfg.ATLAS,
        "n_rois": int(cfg.N_ROIS),
        "node_feature_dim_after_correlation_concat": int(cfg.NODE_FEAT_DIM),
        "edge_feature_dim": int(cfg.EDGE_FEAT_DIM),
        "n_folds": int(cfg.N_FOLDS),
        "epochs": int(cfg.EPOCHS),
        "hidden_dim": int(cfg.HIDDEN_DIM),
        "batch_size": int(cfg.BATCH_SIZE),
        "aps_alpha": float(cfg.APS_ALPHA),
        "seed": int(cfg.SEED),
        "best_fold_for_scrm": best_fold_idx + 1,
        "metrics": {
            name: {
                "mean": float(np.mean([f[name] for f in fold_metrics])),
                "std": float(np.std([f[name] for f in fold_metrics])),
            }
            for name in metric_names
        },
    }

    with (output_dir / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    log.info("Completed AENIN run. Results: %s", output_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the accepted AENIN implementation."
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Folder containing ASD/ and TD/ NIfTI subfolders.",
    )
    parser.add_argument(
        "--dataset-pt",
        default=None,
        help="Optional prebuilt PyG dataset; skips NIfTI graph construction.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/aenin",
    )
    parser.add_argument(
        "--atlas",
        default="aal",
        choices=["aal", "ho"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--aps-alpha", type=float, default=0.5)
    parser.add_argument(
        "--balance-per-class",
        type=int,
        default=None,
        help="Optional balanced subset size per class.",
    )
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-seeds", type=int, default=3)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
