"""Run the eight baseline models from the supplied comparison notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aenin_baselines.feature_pipeline import (
    Config,
    balanced_subset,
    load_or_build_dataset,
)
from aenin_baselines.train_compare import (
    run_5fold,
    run_sitewise,
)


def _jsonable(results):
    out = {}
    for model_name, values in results.items():
        converted = []
        for row in values:
            converted.append(
                [
                    x.item() if isinstance(x, np.generic) else x
                    for x in row
                ]
            )
        out[model_name] = converted
    return out


def main(args):
    cfg = Config()
    cfg.DATA_ROOT = args.data_root
    cfg.ATLAS = args.atlas
    cfg.APS_ALPHA = args.aps_alpha

    dataset = load_or_build_dataset(
        cfg,
        cache_path=args.cache,
        force_rebuild=args.force_rebuild,
        keep_time_series=not args.no_time_series,
    )

    if args.balance_per_class:
        dataset = balanced_subset(
            dataset,
            n_per_class=args.balance_per_class,
            seed=cfg.SEED,
        )

    if not dataset:
        raise RuntimeError("No graph data available.")

    n_nodes = int(dataset[0].x.shape[0])

    if args.mode == "5fold":
        results = run_5fold(
            dataset,
            n_nodes,
            in_dim=18,
            n_splits=args.n_splits,
            epochs=args.epochs,
        )
    else:
        results = run_sitewise(
            dataset,
            n_nodes,
            in_dim=18,
            epochs=args.epochs,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)

    print(f"Saved baseline results to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run AENIN baseline-comparison models."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--cache",
        default="./data/dataset_baselines.pt",
    )
    parser.add_argument(
        "--mode",
        choices=["5fold", "site"],
        default="5fold",
    )
    parser.add_argument("--atlas", choices=["aal", "ho"], default="aal")
    parser.add_argument("--aps-alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--balance-per-class",
        type=int,
        default=400,
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--no-time-series",
        action="store_true",
        help=(
            "Do not store ROI time series. Dynamic baselines then receive "
            "the notebook's zero fallback rather than real temporal data."
        ),
    )
    parser.add_argument(
        "--output",
        default="./results/baselines.json",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
