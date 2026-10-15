#!/usr/bin/env python3
"""Run the frozen train-label-selected production configuration.

This entrypoint keeps formal 20-seed evaluation separate from model selection:
the settings live in ``config/production_config.json`` and must not be changed
after reading an oracle/test-label table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from run_compliant import OFFICIAL_SEEDS, parse_seed_values, run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only-selected official configurations")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset", choices=["all", "Suzuki", "Buchwald_sub4"], default="all")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-values", help="Optional explicit official seed subset for an array job")
    args = parser.parse_args()
    if args.rounds != 40 or not 1 <= args.seeds <= 20:
        raise ValueError("Formal run requires 40 rounds and 1..20 official seeds")
    config = json.loads((CODE_DIR / "config" / "production_config.json").read_text(encoding="utf-8"))
    selected_seeds = parse_seed_values(args.seed_values) if args.seed_values else OFFICIAL_SEEDS[:args.seeds]
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    root = Path(args.data_root)
    output_root = Path(args.results_root)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for dataset in datasets:
        settings = config["datasets"][dataset]
        structure_map = settings.get("structure_map")
        if structure_map is not None:
            structure_map = CODE_DIR / structure_map
        preference_prior = settings.get("preference_prior")
        if preference_prior is not None:
            preference_prior = CODE_DIR / preference_prior
        dft_features = settings.get("dft_features")
        if dft_features is not None:
            dft_features = CODE_DIR / dft_features
        output_dir = output_root / dataset
        summary = run_dataset(
            dataset_dir=root / dataset,
            dataset_name=dataset,
            output_dir=output_dir,
            rounds=args.rounds,
            seeds=selected_seeds,
            epsilon_start=settings["epsilon_start"],
            epsilon_end=settings["epsilon_end"],
            use_chemistry=settings["use_chemistry"],
            strategy=settings["strategy"],
            kernel=settings["kernel"],
            acquisition=settings["acquisition"],
            surrogate=settings["surrogate"],
            forest_min_samples_leaf=settings.get("forest_min_samples_leaf", 2),
            hybrid_forest_weight=settings.get("hybrid_forest_weight", 0.5),
            hybrid_forest_weight_end=settings.get("hybrid_forest_weight_end"),
            gp_restarts=settings.get("gp_restarts", 3),
            categorical_rho=settings.get("categorical_rho", 0.3),
            optimize_categorical_kernel=settings.get(
                "optimize_categorical_kernel", True
            ),
            categorical_optimize_once=settings.get(
                "categorical_optimize_once", False
            ),
            kernel_switch_rounds=settings.get("kernel_switch_rounds", 0),
            kernel_after_switch=settings.get("kernel_after_switch"),
            categorical_kernel_columns=settings.get("categorical_kernel_columns"),
            use_structure=settings.get("use_structure", False),
            structure_map=structure_map,
            structure_fingerprint_bits=settings.get("structure_fingerprint_bits", 32),
            include_categorical_identity=settings.get("include_categorical_identity", True),
            uncertainty_scale=settings.get("uncertainty_scale", 1.0),
            warmup_random_rounds=settings.get("warmup_random_rounds", 0),
            greedy_rounds=settings.get("greedy_rounds", 0),
            top_probability_rounds=settings.get("top_probability_rounds", 0),
            tail_probability_weight=settings.get("tail_probability_weight", 0.0),
            tail_quantile=settings.get("tail_quantile", 0.75),
            adaptive_models=settings.get("adaptive_models"),
            adaptive_temperature=settings.get("adaptive_temperature", 0.50),
            adaptive_forgetting=settings.get("adaptive_forgetting", 0.85),
            preference_prior=preference_prior,
            preference_strength=settings.get("preference_strength", 1.0),
            svr_c=settings.get("svr_c", 10.0),
            svr_epsilon=settings.get("svr_epsilon", 0.10),
            lasso_alpha=settings.get("lasso_alpha", 0.50),
            boosting_estimators=settings.get("boosting_estimators", 60),
            boosting_learning_rate=settings.get("boosting_learning_rate", 0.05),
            boosting_max_depth=settings.get("boosting_max_depth", 2),
            use_dft=settings.get("use_dft", False),
            dft_features=dft_features,
            dft_profile=settings.get("dft_profile", "all"),
            xgboost_estimators=settings.get("xgboost_estimators", 100),
            xgboost_learning_rate=settings.get("xgboost_learning_rate", 0.05),
            xgboost_max_depth=settings.get("xgboost_max_depth", 2),
            xgboost_subsample=settings.get("xgboost_subsample", 0.8),
            xgboost_colsample_bytree=settings.get("xgboost_colsample_bytree", 0.8),
            xgboost_reg_alpha=settings.get("xgboost_reg_alpha", 0.0),
            xgboost_reg_lambda=settings.get("xgboost_reg_lambda", 5.0),
            catboost_members=settings.get("catboost_members", 12),
            catboost_iterations=settings.get("catboost_iterations", 80),
            catboost_learning_rate=settings.get("catboost_learning_rate", 0.05),
            catboost_depth=settings.get("catboost_depth", 3),
            catboost_l2_leaf_reg=settings.get("catboost_l2_leaf_reg", 5.0),
        )
        print(f"\n=== {dataset}: frozen production configuration ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
