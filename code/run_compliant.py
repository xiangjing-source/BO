"""Run leakage-safe sequential BO on one public competition dataset.

This local runner is intentionally split into a public-data optimizer and a
private offline oracle.  The same optimizer can later be connected to a remote
experimental service without changing its decision logic.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from compliant_bo import CategoricalBayesOptimizer
from chemical_knowledge import ChemicalKnowledgeFeaturizer
from dft_features import DFTElectronicFeaturizer, MechanisticDFTFeaturizer
from molecular_features import CombinedFeaturizer, MolecularStructureFeaturizer
from preference_guidance import MechanisticPreferencePrior
from protocol import OfflineOracle, official_metrics, load_public_dataset, save_trajectory


OFFICIAL_SEEDS = list(range(100, 2001, 100))


def parse_seed_values(value: str) -> List[int]:
    """Parse an explicit, non-empty subset of the published fixed seeds."""
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in OFFICIAL_SEEDS for seed in seeds):
        raise ValueError("--seed-values must be unique members of 100,200,...,2000")
    return seeds


def summarize(metrics: List[Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for key in (
        "initial_round_found_best",
        "best_found",
        "round_to_95_global_best",
        "AUC_best_so_far",
    ):
        values = np.asarray([entry[key] for entry in metrics], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append({"metric": key, "mean": mean, "std": std,
                     "ci95_lower": mean - ci, "ci95_upper": mean + ci})
    return pd.DataFrame(rows)


def run_dataset(
    dataset_dir: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    rounds: int = 40,
    seeds: List[int] | None = None,
    epsilon_start: float = 0.15,
    epsilon_end: float = 0.01,
    knowledge_cards: str | Path | None = None,
    use_chemistry: bool = True,
    strategy: str = "expected_improvement",
    kernel: str = "dot",
    acquisition: str = "expected_improvement",
    ucb_beta: float = 1.0,
    surrogate: str = "gaussian_process",
    forest_min_samples_leaf: int = 2,
    ridge_alpha: float = 10.0,
    ensemble_size: int = 100,
    hybrid_forest_weight: float = 0.5,
    hybrid_ridge_weight: float = 0.5,
    feature_mode: str = "main_effects",
    diversity_weight: float = 0.0,
    knn_neighbors: int = 5,
    uncertainty_scale: float = 1.0,
    warmup_random_rounds: int = 0,
    greedy_rounds: int = 0,
    top_probability_rounds: int = 0,
    tail_probability_weight: float = 0.0,
    tail_quantile: float = 0.75,
    gp_bootstrap: bool = False,
    gp_alpha: float = 1e-6,
    gp_noise: float = 0.05,
    product_descriptor_only: bool = False,
    gp_restarts: int = 3,
    hybrid_forest_weight_end: float | None = None,
    product_transfer_weight: float = 0.5,
    use_structure: bool = False,
    structure_map: str | Path | None = None,
    structure_fingerprint_bits: int = 32,
    include_categorical_identity: bool = True,
    adaptive_models: List[str] | None = None,
    adaptive_temperature: float = 0.50,
    adaptive_forgetting: float = 0.85,
    preference_prior: str | Path | None = None,
    preference_strength: float = 1.0,
    svr_c: float = 10.0,
    svr_epsilon: float = 0.10,
    lasso_alpha: float = 0.50,
    boosting_estimators: int = 60,
    boosting_learning_rate: float = 0.05,
    boosting_max_depth: int = 2,
    use_dft: bool = False,
    dft_features: str | Path | None = None,
    dft_profile: str = "all",
    use_mechanistic_dft: bool = False,
    mechanistic_dft_features: str | Path | None = None,
    mechanistic_dft_profile: str = "compact",
    xgboost_estimators: int = 100,
    xgboost_learning_rate: float = 0.05,
    xgboost_max_depth: int = 2,
    xgboost_subsample: float = 0.8,
    xgboost_colsample_bytree: float = 0.8,
    xgboost_reg_alpha: float = 0.0,
    xgboost_reg_lambda: float = 5.0,
    catboost_members: int = 12,
    catboost_iterations: int = 80,
    catboost_learning_rate: float = 0.05,
    catboost_depth: int = 3,
    catboost_l2_leaf_reg: float = 5.0,
    categorical_rho: float = 0.3,
    optimize_categorical_kernel: bool = True,
    categorical_optimize_once: bool = False,
    kernel_switch_rounds: int = 0,
    kernel_after_switch: str | None = None,
    categorical_kernel_columns: List[str] | None = None,
) -> pd.DataFrame:
    public = load_public_dataset(dataset_dir, dataset_name)
    xgboost_version = None
    for distribution in ("xgboost-cpu", "xgboost"):
        try:
            xgboost_version = package_version(distribution)
            break
        except PackageNotFoundError:
            continue
    all_metrics: List[Dict[str, float]] = []
    all_records: Dict[str, List[Dict[str, object]]] = {}
    selected_seeds = OFFICIAL_SEEDS if seeds is None else seeds

    for seed in selected_seeds:
        # The optimizer gets no oracle table and no global optimum.
        knowledge = ChemicalKnowledgeFeaturizer(public.variable_columns, knowledge_cards)
        feature_blocks: list[object] = []
        dft_featurizer: DFTElectronicFeaturizer | None = None
        mechanistic_dft_featurizer: MechanisticDFTFeaturizer | None = None
        if use_chemistry:
            feature_blocks.append(knowledge)
        if use_structure:
            if structure_map is None:
                raise ValueError("use_structure requires a fixed structure_map")
            feature_blocks.append(
                MolecularStructureFeaturizer(
                    public.variable_columns,
                    structure_map,
                    fingerprint_bits=structure_fingerprint_bits,
                    radius=2,
                    include_fingerprint=True,
                    include_physicochemical=True,
                )
            )
        if use_dft:
            if dft_features is None:
                raise ValueError("use_dft requires an outcome-free dft_features table")
            dft_profiles = {
                "all": None,
                "compact": (
                    "homo_ev",
                    "lumo_ev",
                    "homo_lumo_gap_ev",
                    "dipole_debye",
                    "hirshfeld_abs_charge_mean",
                    "mayer_bond_density",
                ),
                "frontier": (
                    "homo_ev",
                    "lumo_ev",
                    "homo_lumo_gap_ev",
                    "dipole_debye",
                ),
                "extended": DFTElectronicFeaturizer.EXTENDED_FEATURE_COLUMNS,
            }
            if dft_profile not in dft_profiles:
                raise ValueError(f"Unknown DFT profile: {dft_profile}")
            dft_featurizer = DFTElectronicFeaturizer(
                dataset_name,
                public.variable_columns,
                dft_features,
                feature_columns=dft_profiles[dft_profile],
            )
            feature_blocks.append(dft_featurizer)
        if use_mechanistic_dft:
            if mechanistic_dft_features is None:
                raise ValueError(
                    "use_mechanistic_dft requires an outcome-free feature table"
                )
            mechanistic_dft_featurizer = MechanisticDFTFeaturizer(
                dataset_name,
                public.variable_columns,
                mechanistic_dft_features,
                profile=mechanistic_dft_profile,
            )
            feature_blocks.append(mechanistic_dft_featurizer)
        auxiliary_featurizer = (
            None
            if not feature_blocks
            else feature_blocks[0]
            if len(feature_blocks) == 1
            else CombinedFeaturizer(*feature_blocks)
        )
        preference_provider = (
            None
            if preference_prior is None
            else MechanisticPreferencePrior(
                dataset_name,
                public.variable_columns,
                preference_prior,
            )
        )
        optimizer = CategoricalBayesOptimizer(
            variable_columns=public.variable_columns,
            categories=public.categories,
            seed=seed,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            chemical_featurizer=auxiliary_featurizer,
            strategy=strategy,
            kernel=kernel,
            acquisition=acquisition,
            ucb_beta=ucb_beta,
            surrogate=surrogate,
            forest_min_samples_leaf=forest_min_samples_leaf,
            ridge_alpha=ridge_alpha,
            ensemble_size=ensemble_size,
            hybrid_forest_weight=hybrid_forest_weight,
            hybrid_ridge_weight=hybrid_ridge_weight,
            feature_mode=feature_mode,
            diversity_weight=diversity_weight,
            knn_neighbors=knn_neighbors,
            uncertainty_scale=uncertainty_scale,
            warmup_random_rounds=warmup_random_rounds,
            greedy_rounds=greedy_rounds,
            top_probability_rounds=top_probability_rounds,
            tail_probability_weight=tail_probability_weight,
            tail_quantile=tail_quantile,
            gp_bootstrap=gp_bootstrap,
            gp_alpha=gp_alpha,
            gp_noise=gp_noise,
            product_descriptor_only=product_descriptor_only,
            n_restarts_optimizer=gp_restarts,
            hybrid_forest_weight_end=hybrid_forest_weight_end,
            product_transfer_weight=product_transfer_weight,
            include_categorical_identity=include_categorical_identity,
            adaptive_models=adaptive_models,
            adaptive_temperature=adaptive_temperature,
            adaptive_forgetting=adaptive_forgetting,
            preference_provider=preference_provider,
            preference_strength=preference_strength,
            svr_c=svr_c,
            svr_epsilon=svr_epsilon,
            lasso_alpha=lasso_alpha,
            boosting_estimators=boosting_estimators,
            boosting_learning_rate=boosting_learning_rate,
            boosting_max_depth=boosting_max_depth,
            xgboost_estimators=xgboost_estimators,
            xgboost_learning_rate=xgboost_learning_rate,
            xgboost_max_depth=xgboost_max_depth,
            xgboost_subsample=xgboost_subsample,
            xgboost_colsample_bytree=xgboost_colsample_bytree,
            xgboost_reg_alpha=xgboost_reg_alpha,
            xgboost_reg_lambda=xgboost_reg_lambda,
            catboost_members=catboost_members,
            catboost_iterations=catboost_iterations,
            catboost_learning_rate=catboost_learning_rate,
            catboost_depth=catboost_depth,
            catboost_l2_leaf_reg=catboost_l2_leaf_reg,
            categorical_rho=categorical_rho,
            optimize_categorical_kernel=optimize_categorical_kernel,
            categorical_optimize_once=categorical_optimize_once,
            kernel_switch_rounds=kernel_switch_rounds,
            kernel_after_switch=kernel_after_switch,
            categorical_kernel_columns=categorical_kernel_columns,
        )
        optimizer.add_initial_training(public.train)
        # This object is the sole local owner of test labels.
        oracle = OfflineOracle(dataset_dir, public)

        queried: List[int] = []
        best_so_far = float(public.target_train["Yield"].max())
        records: List[Dict[str, object]] = []
        for round_index in range(rounds):
            recommendation = optimizer.recommend(
                candidates=public.candidates,
                queried_indices=queried,
                target_incumbent=best_so_far,
                round_index=round_index,
                total_rounds=rounds,
            )
            index = recommendation.query_index
            condition = public.candidates.iloc[[index]]
            observed_yield = oracle.query(index)
            best_so_far = max(best_so_far, observed_yield)
            optimizer.add_feedback(condition, observed_yield)
            queried.append(index)
            records.append({
                "step": round_index + 1,
                "query_index": index,
                "condition": {column: str(condition.iloc[0][column]) for column in public.variable_columns},
                "observed_yield": observed_yield,
                "best_so_far": best_so_far,
                "predicted_yield": recommendation.predicted_yield,
                "uncertainty": recommendation.uncertainty,
                "acquisition_value": recommendation.acquisition_value,
                "selection_mode": recommendation.selection_mode,
                "preference_confidence": recommendation.preference_confidence,
                "preference_shift": recommendation.preference_shift,
                "ensemble_weights": (
                    dict(recommendation.ensemble_weights)
                    if recommendation.ensemble_weights is not None
                    else None
                ),
                "dft_components_available": (
                    int(dft_featurizer.coverage(condition)["mean_components_available"])
                    if dft_featurizer is not None
                    else None
                ),
                "mechanistic_dft_components_available": (
                    float(
                        mechanistic_dft_featurizer.coverage(condition)[
                            "mean_components_available"
                        ]
                    )
                    if mechanistic_dft_featurizer is not None
                    else None
                ),
            })

        global_best = oracle.final_global_best()  # evaluation only, after all decisions
        metric = official_metrics(records, global_best)
        metric["seed"] = int(seed)
        all_metrics.append(metric)
        all_records[str(seed)] = records
        save_trajectory(output_dir, public, seed, records, expected_rounds=rounds)

    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(all_metrics)
    summary.insert(0, "dataset", dataset_name)
    summary.to_csv(result_dir / f"{dataset_name}_summary_metrics.csv", index=False)
    with (result_dir / f"{dataset_name}_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "dataset": dataset_name,
            "rounds": rounds,
            "seeds": selected_seeds,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "knowledge_cards": str(knowledge_cards) if knowledge_cards else None,
            "auxiliary_feature_names": (
                auxiliary_featurizer.feature_names if auxiliary_featurizer is not None else []
            ),
            "use_chemistry": use_chemistry,
            "use_structure": use_structure,
            "structure_map": str(structure_map) if structure_map else None,
            "structure_fingerprint_bits": structure_fingerprint_bits,
            "include_categorical_identity": include_categorical_identity,
            "adaptive_models": adaptive_models,
            "adaptive_temperature": adaptive_temperature,
            "adaptive_forgetting": adaptive_forgetting,
            "preference_prior": str(preference_prior) if preference_prior else None,
            "preference_strength": preference_strength,
            "svr_c": svr_c,
            "svr_epsilon": svr_epsilon,
            "lasso_alpha": lasso_alpha,
            "boosting_estimators": boosting_estimators,
            "boosting_learning_rate": boosting_learning_rate,
            "boosting_max_depth": boosting_max_depth,
            "use_dft": use_dft,
            "dft_features": str(dft_features) if dft_features else None,
            "dft_profile": dft_profile,
            "dft_public_coverage": (
                dft_featurizer.coverage(
                    pd.concat(
                        [
                            public.train.loc[:, public.variable_columns],
                            public.candidates.loc[:, public.variable_columns],
                        ],
                        ignore_index=True,
                    )
                )
                if dft_featurizer is not None
                else None
            ),
            "use_mechanistic_dft": use_mechanistic_dft,
            "mechanistic_dft_features": (
                str(mechanistic_dft_features)
                if mechanistic_dft_features
                else None
            ),
            "mechanistic_dft_profile": mechanistic_dft_profile,
            "mechanistic_dft_public_coverage": (
                mechanistic_dft_featurizer.coverage(
                    pd.concat(
                        [
                            public.train.loc[:, public.variable_columns],
                            public.candidates.loc[:, public.variable_columns],
                        ],
                        ignore_index=True,
                    )
                )
                if mechanistic_dft_featurizer is not None
                else None
            ),
            "xgboost_estimators": xgboost_estimators,
            "xgboost_learning_rate": xgboost_learning_rate,
            "xgboost_max_depth": xgboost_max_depth,
            "xgboost_subsample": xgboost_subsample,
            "xgboost_colsample_bytree": xgboost_colsample_bytree,
            "xgboost_reg_alpha": xgboost_reg_alpha,
            "xgboost_reg_lambda": xgboost_reg_lambda,
            "xgboost_version": xgboost_version,
            "catboost_members": catboost_members,
            "catboost_iterations": catboost_iterations,
            "catboost_learning_rate": catboost_learning_rate,
            "catboost_depth": catboost_depth,
            "catboost_l2_leaf_reg": catboost_l2_leaf_reg,
            "categorical_rho": categorical_rho,
            "optimize_categorical_kernel": optimize_categorical_kernel,
            "categorical_optimize_once": categorical_optimize_once,
            "kernel_switch_rounds": kernel_switch_rounds,
            "kernel_after_switch": kernel_after_switch,
            "categorical_kernel_columns": categorical_kernel_columns,
            "strategy": strategy,
            "kernel": kernel,
            "acquisition": acquisition,
            "ucb_beta": ucb_beta,
            "surrogate": surrogate,
            "forest_min_samples_leaf": forest_min_samples_leaf,
            "ridge_alpha": ridge_alpha,
            "ensemble_size": ensemble_size,
            "hybrid_forest_weight": hybrid_forest_weight,
            "hybrid_ridge_weight": hybrid_ridge_weight,
            "feature_mode": feature_mode,
            "diversity_weight": diversity_weight,
            "knn_neighbors": knn_neighbors,
            "uncertainty_scale": uncertainty_scale,
            "warmup_random_rounds": warmup_random_rounds,
            "greedy_rounds": greedy_rounds,
            "top_probability_rounds": top_probability_rounds,
            "tail_probability_weight": tail_probability_weight,
            "tail_quantile": tail_quantile,
            "gp_bootstrap": gp_bootstrap,
            "gp_alpha": gp_alpha,
            "gp_noise": gp_noise,
            "product_descriptor_only": product_descriptor_only,
            "gp_restarts": gp_restarts,
            "hybrid_forest_weight_end": hybrid_forest_weight_end,
            "product_transfer_weight": product_transfer_weight,
            "metrics": all_metrics,
        }, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe reaction-condition BO runner")
    parser.add_argument("--dataset-dir", required=True, help="One dataset directory, e.g. .../Suzuki")
    parser.add_argument("--dataset-name", required=True, choices=["Suzuki", "Buchwald_sub4"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-values", help="Comma-separated explicit official seed subset; enables batch/array execution")
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--knowledge-cards", help="Optional human-reviewed JSON chemistry card file")
    parser.add_argument("--without-chemistry", action="store_true", help="Run categorical-only ablation")
    parser.add_argument("--strategy", choices=["expected_improvement", "random", "tpe", "thompson"], default="expected_improvement")
    parser.add_argument(
        "--kernel",
        choices=["rbf", "dot", "categorical_product", "rbf_categorical_sum"],
        default="dot",
        help="GP covariance used by the surrogate",
    )
    parser.add_argument("--acquisition", choices=["expected_improvement", "upper_confidence_bound"], default="expected_improvement")
    parser.add_argument("--ucb-beta", type=float, default=1.0, help="Uncertainty weight when using upper confidence bound")
    parser.add_argument(
        "--surrogate",
        choices=[
            "gaussian_process", "random_forest", "extra_trees", "svr_ensemble",
            "lasso_ensemble", "gradient_boosting_ensemble", "adaptive_ensemble",
            "xgboost_ensemble",
            "catboost_categorical_ensemble",
            "dual_forest_product_transfer", "categorical_knn", "ridge_ensemble",
            "ridge_pair_ensemble", "hybrid_gp_forest", "hybrid_gp_ridge",
            "hybrid_identity_gp_forest",
        ],
        default="gaussian_process",
    )
    parser.add_argument("--forest-min-samples-leaf", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--ensemble-size", type=int, default=100)
    parser.add_argument("--hybrid-forest-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-ridge-weight", type=float, default=0.5)
    parser.add_argument("--feature-mode", choices=["main_effects", "pairwise"], default="main_effects")
    parser.add_argument("--diversity-weight", type=float, default=0.0, help="Label-free categorical diversity bonus for EI/UCB")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Number of categorical neighbours when using categorical_knn")
    parser.add_argument("--uncertainty-scale", type=float, default=1.0, help="Multiplicative calibration for acquisition uncertainty")
    parser.add_argument("--warmup-random-rounds", type=int, default=0, help="Initial label-free random exploration rounds")
    parser.add_argument("--greedy-rounds", type=int, default=0, help="Opening rounds ranked by posterior mean")
    parser.add_argument("--top-probability-rounds", type=int, default=0, help="Opening rounds ranked by train-only high-yield-tail probability")
    parser.add_argument("--tail-probability-weight", type=float, default=0.0, help="Additive high-yield-tail probability weight after the opening")
    parser.add_argument("--tail-quantile", type=float, default=0.75, help="Observed-yield quantile defining the high-yield class")
    parser.add_argument("--gp-bootstrap", action="store_true", help="Bootstrap GP observations for small-sample uncertainty")
    parser.add_argument("--gp-alpha", type=float, default=1e-6, help="GP diagonal regularization")
    parser.add_argument("--gp-noise", type=float, default=0.05, help="Initial GP white-noise level")
    parser.add_argument("--product-descriptor-only", action="store_true", help="For transfer tasks, replace Product one-hot with chemistry descriptors")
    parser.add_argument("--gp-restarts", type=int, default=3, help="GP kernel-optimizer restarts")
    parser.add_argument("--hybrid-forest-weight-end", type=float, help="Optional final forest weight for a linear hybrid schedule")
    parser.add_argument("--product-transfer-weight", type=float, default=0.5, help="Transfer-forest weight for dual_forest_product_transfer")
    parser.add_argument("--use-structure", action="store_true", help="Append fixed molecular structure features")
    parser.add_argument("--structure-map", help="Reviewed component-name to structure JSON")
    parser.add_argument("--structure-fingerprint-bits", type=int, default=32)
    parser.add_argument("--without-categorical-identity", action="store_true")
    parser.add_argument("--adaptive-models", help="Comma-separated adaptive ensemble components")
    parser.add_argument("--adaptive-temperature", type=float, default=0.50)
    parser.add_argument("--adaptive-forgetting", type=float, default=0.85)
    parser.add_argument("--preference-prior", help="Fixed label-independent preference-prior JSON")
    parser.add_argument("--preference-strength", type=float, default=1.0)
    parser.add_argument("--svr-c", type=float, default=10.0)
    parser.add_argument("--svr-epsilon", type=float, default=0.10)
    parser.add_argument("--lasso-alpha", type=float, default=0.50)
    parser.add_argument("--boosting-estimators", type=int, default=60)
    parser.add_argument("--boosting-learning-rate", type=float, default=0.05)
    parser.add_argument("--boosting-max-depth", type=int, default=2)
    parser.add_argument("--use-dft", action="store_true", help="Append frozen missing-aware DFT features")
    parser.add_argument("--dft-features", help="Outcome-free frozen component DFT CSV")
    parser.add_argument(
        "--dft-profile",
        choices=["all", "compact", "frontier", "extended"],
        default="all",
    )
    parser.add_argument(
        "--use-mechanistic-dft",
        action="store_true",
        help="Append frozen reaction-site and mechanism-interaction DFT features",
    )
    parser.add_argument(
        "--mechanistic-dft-features",
        help="Outcome-free frozen mechanism-aware component DFT CSV",
    )
    parser.add_argument(
        "--mechanistic-dft-profile",
        choices=["full", "compact", "interactions", "poisoning"],
        default="compact",
    )
    parser.add_argument("--xgboost-estimators", type=int, default=100)
    parser.add_argument("--xgboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgboost-max-depth", type=int, default=2)
    parser.add_argument("--xgboost-subsample", type=float, default=0.8)
    parser.add_argument("--xgboost-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgboost-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgboost-reg-lambda", type=float, default=5.0)
    parser.add_argument("--catboost-members", type=int, default=12)
    parser.add_argument("--catboost-iterations", type=int, default=80)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--catboost-depth", type=int, default=3)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--categorical-rho", type=float, default=0.3)
    parser.add_argument(
        "--no-optimize-categorical-kernel",
        action="store_true",
        help="Freeze categorical rho/amplitude/noise instead of optimizing each fit",
    )
    parser.add_argument(
        "--categorical-optimize-once",
        action="store_true",
        help="Optimize the categorical kernel on initial labels, then freeze it",
    )
    parser.add_argument("--kernel-switch-rounds", type=int, default=0)
    parser.add_argument(
        "--kernel-after-switch",
        choices=["rbf", "dot", "categorical_product"],
    )
    parser.add_argument(
        "--categorical-kernel-columns",
        help="Comma-separated identity columns used by categorical-product kernels",
    )
    args = parser.parse_args()
    # Hyperparameter-bound messages are diagnostic during development, but do
    # not change a fitted trajectory and would otherwise flood 20-seed logs.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if not 1 <= args.seeds <= 20:
        raise ValueError("seeds must be between 1 and 20")
    selected_seeds = parse_seed_values(args.seed_values) if args.seed_values else OFFICIAL_SEEDS[:args.seeds]
    summary = run_dataset(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        rounds=args.rounds,
        seeds=selected_seeds,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        knowledge_cards=args.knowledge_cards,
        use_chemistry=not args.without_chemistry,
        strategy=args.strategy,
        kernel=args.kernel,
        acquisition=args.acquisition,
        ucb_beta=args.ucb_beta,
        surrogate=args.surrogate,
        forest_min_samples_leaf=args.forest_min_samples_leaf,
        ridge_alpha=args.ridge_alpha,
        ensemble_size=args.ensemble_size,
        hybrid_forest_weight=args.hybrid_forest_weight,
        hybrid_ridge_weight=args.hybrid_ridge_weight,
        feature_mode=args.feature_mode,
        diversity_weight=args.diversity_weight,
        knn_neighbors=args.knn_neighbors,
        uncertainty_scale=args.uncertainty_scale,
        warmup_random_rounds=args.warmup_random_rounds,
        greedy_rounds=args.greedy_rounds,
        top_probability_rounds=args.top_probability_rounds,
        tail_probability_weight=args.tail_probability_weight,
        tail_quantile=args.tail_quantile,
        gp_bootstrap=args.gp_bootstrap,
        gp_alpha=args.gp_alpha,
        gp_noise=args.gp_noise,
        product_descriptor_only=args.product_descriptor_only,
        gp_restarts=args.gp_restarts,
        hybrid_forest_weight_end=args.hybrid_forest_weight_end,
        product_transfer_weight=args.product_transfer_weight,
        use_structure=args.use_structure,
        structure_map=args.structure_map,
        structure_fingerprint_bits=args.structure_fingerprint_bits,
        include_categorical_identity=not args.without_categorical_identity,
        adaptive_models=args.adaptive_models.split(",") if args.adaptive_models else None,
        adaptive_temperature=args.adaptive_temperature,
        adaptive_forgetting=args.adaptive_forgetting,
        preference_prior=args.preference_prior,
        preference_strength=args.preference_strength,
        svr_c=args.svr_c,
        svr_epsilon=args.svr_epsilon,
        lasso_alpha=args.lasso_alpha,
        boosting_estimators=args.boosting_estimators,
        boosting_learning_rate=args.boosting_learning_rate,
        boosting_max_depth=args.boosting_max_depth,
        use_dft=args.use_dft,
        dft_features=args.dft_features,
        dft_profile=args.dft_profile,
        use_mechanistic_dft=args.use_mechanistic_dft,
        mechanistic_dft_features=args.mechanistic_dft_features,
        mechanistic_dft_profile=args.mechanistic_dft_profile,
        xgboost_estimators=args.xgboost_estimators,
        xgboost_learning_rate=args.xgboost_learning_rate,
        xgboost_max_depth=args.xgboost_max_depth,
        xgboost_subsample=args.xgboost_subsample,
        xgboost_colsample_bytree=args.xgboost_colsample_bytree,
        xgboost_reg_alpha=args.xgboost_reg_alpha,
        xgboost_reg_lambda=args.xgboost_reg_lambda,
        catboost_members=args.catboost_members,
        catboost_iterations=args.catboost_iterations,
        catboost_learning_rate=args.catboost_learning_rate,
        catboost_depth=args.catboost_depth,
        catboost_l2_leaf_reg=args.catboost_l2_leaf_reg,
        categorical_rho=args.categorical_rho,
        optimize_categorical_kernel=not args.no_optimize_categorical_kernel,
        categorical_optimize_once=args.categorical_optimize_once,
        kernel_switch_rounds=args.kernel_switch_rounds,
        kernel_after_switch=args.kernel_after_switch,
        categorical_kernel_columns=(
            args.categorical_kernel_columns.split(",")
            if args.categorical_kernel_columns
            else None
        ),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
