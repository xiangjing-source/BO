"""Regression tests for the optimizer's information boundary and new BO layer."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from compliant_bo import CategoricalBayesOptimizer
from preference_guidance import MechanisticPreferencePrior
from protocol import OfflineOracle, load_public_dataset


VARIABLES = ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"]


def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for index in range(12):
        rows.append({
            "Electrophile": ["1-bromobenzene", "1-chlorobenzene"][index % 2],
            "Nucleophile": ["phenylboronic acid", "phenylboronate"][index % 2],
            "Ligand": ["triphenylphosphane", "tricyclohexylphosphane"][(index // 2) % 2],
            "Base": ["potassium carbonate", "potassium phosphate"][(index // 3) % 2],
            "Solvent": ["methanol", "oxolane"][(index // 4) % 2],
            "Yield": float(5 + 4 * index),
        })
    train = pd.DataFrame(rows)
    candidates = pd.DataFrame([
        {
            "Electrophile": "1-bromobenzene",
            "Nucleophile": "phenylboronic acid",
            "Ligand": ligand,
            "Base": base,
            "Solvent": solvent,
        }
        for ligand, base, solvent in (
            ("triphenylphosphane", "potassium carbonate", "methanol"),
            ("tricyclohexylphosphane", "potassium carbonate", "oxolane"),
            ("triphenylphosphane", "potassium phosphate", "oxolane"),
            ("tricyclohexylphosphane", "potassium phosphate", "methanol"),
        )
    ])
    oracle = candidates.copy()
    oracle["Yield"] = [30.0, 55.0, 40.0, 65.0]
    return train, candidates, oracle


class ProtocolAndPreferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        train, candidates, oracle = synthetic_tables()
        train.to_csv(root / "suzuki_train.csv", index=False)
        candidates.to_csv(root / "suzuki_test_features.csv", index=False)
        oracle.to_csv(root / "suzuki_test.csv", index=False)
        self.root = root
        self.prior_path = CODE_DIR / "config" / "mechanistic_preference_priors.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_public_loader_never_exposes_candidate_yield(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        self.assertNotIn("Yield", public.candidates)
        oracle = OfflineOracle(self.root, public)
        self.assertEqual(oracle.query(1), 55.0)
        with self.assertRaises(ValueError):
            oracle.query(1)

    def test_labelled_candidate_file_is_rejected(self) -> None:
        _, _, oracle = synthetic_tables()
        oracle.to_csv(self.root / "suzuki_test_features.csv", index=False)
        with self.assertRaisesRegex(ValueError, "must not contain Yield"):
            load_public_dataset(self.root, "Suzuki")

    def test_preference_score_does_not_depend_on_a_yield_column(self) -> None:
        _, candidates, _ = synthetic_tables()
        prior = MechanisticPreferencePrior("Suzuki", VARIABLES, self.prior_path)
        baseline = prior.score(candidates)
        labelled_copy = candidates.copy()
        labelled_copy["Yield"] = np.linspace(-1e9, 1e9, len(labelled_copy))
        np.testing.assert_allclose(baseline, prior.score(labelled_copy))

    def test_misaligned_preference_is_automatically_disabled(self) -> None:
        train, _, _ = synthetic_tables()
        prior = MechanisticPreferencePrior("Suzuki", VARIABLES, self.prior_path)
        scores = prior.score(train)
        confidence, tau = prior.confidence(train[VARIABLES], -scores)
        self.assertLess(tau, 0.0)
        self.assertEqual(confidence, 0.0)

    def test_adaptive_weights_and_region_lift_are_auditable(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        prior = MechanisticPreferencePrior("Suzuki", VARIABLES, self.prior_path)
        optimizer = CategoricalBayesOptimizer(
            variable_columns=VARIABLES,
            categories=public.categories,
            seed=123,
            surrogate="adaptive_ensemble",
            adaptive_models=("random_forest", "extra_trees", "svr", "lasso"),
            ensemble_size=12,
            preference_provider=prior,
        )
        optimizer.add_initial_training(public.train)
        recommendation = optimizer.recommend(
            public.candidates,
            queried_indices=[],
            target_incumbent=float(public.train["Yield"].max()),
            round_index=0,
            total_rounds=4,
        )
        self.assertIn(recommendation.query_index, range(len(public.candidates)))
        self.assertIsNotNone(recommendation.ensemble_weights)
        self.assertAlmostEqual(sum(recommendation.ensemble_weights.values()), 1.0)
        self.assertGreaterEqual(recommendation.preference_confidence, 0.0)
        self.assertLessEqual(recommendation.preference_confidence, 1.0)

    def test_opening_tail_ranker_is_candidate_label_blind(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")

        def recommend(candidates: pd.DataFrame):
            optimizer = CategoricalBayesOptimizer(
                variable_columns=VARIABLES,
                categories=public.categories,
                seed=321,
                surrogate="random_forest",
                ensemble_size=24,
                top_probability_rounds=1,
                tail_quantile=0.75,
                epsilon_start=0.0,
                epsilon_end=0.0,
            )
            optimizer.add_initial_training(public.train)
            return optimizer.recommend(
                candidates,
                queried_indices=[],
                target_incumbent=float(public.train["Yield"].max()),
                round_index=0,
                total_rounds=4,
            )

        baseline = recommend(public.candidates)
        labelled = public.candidates.copy()
        labelled["Yield"] = np.linspace(-1e9, 1e9, len(labelled))
        perturbed = recommend(labelled)
        self.assertEqual(baseline.query_index, perturbed.query_index)
        self.assertEqual(baseline.selection_mode, "top_tail_probability")
        self.assertEqual(baseline.selection_mode, perturbed.selection_mode)

    def test_fixed_categorical_kernel_is_fast_and_candidate_label_blind(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")

        def recommend(candidates: pd.DataFrame) -> tuple[int, float]:
            optimizer = CategoricalBayesOptimizer(
                variable_columns=VARIABLES,
                categories=public.categories,
                seed=913,
                surrogate="hybrid_gp_forest",
                kernel="categorical_product",
                categorical_rho=0.30,
                optimize_categorical_kernel=False,
                ensemble_size=12,
                n_restarts_optimizer=0,
                epsilon_start=0.0,
                epsilon_end=0.0,
            )
            optimizer.add_initial_training(public.train)
            result = optimizer.recommend(
                candidates,
                queried_indices=[],
                target_incumbent=float(public.train["Yield"].max()),
                round_index=0,
                total_rounds=4,
            )
            self.assertIsNone(optimizer._model["gp"].optimizer)
            return result.query_index, result.acquisition_value

        baseline = recommend(public.candidates)
        labelled = public.candidates.copy()
        labelled["Yield"] = np.linspace(-1e9, 1e9, len(labelled))
        self.assertEqual(baseline, recommend(labelled))

    def test_categorical_opening_switches_to_rbf(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        optimizer = CategoricalBayesOptimizer(
            variable_columns=VARIABLES,
            categories=public.categories,
            seed=914,
            surrogate="hybrid_gp_forest",
            kernel="categorical_product",
            categorical_rho=0.30,
            optimize_categorical_kernel=False,
            kernel_switch_rounds=1,
            kernel_after_switch="rbf",
            ensemble_size=12,
            n_restarts_optimizer=0,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        optimizer.add_initial_training(public.train)
        first = optimizer.recommend(
            public.candidates,
            queried_indices=[],
            target_incumbent=float(public.train["Yield"].max()),
            round_index=0,
            total_rounds=4,
        )
        self.assertEqual(optimizer._active_kernel, "categorical_product")
        optimizer.add_feedback(
            public.candidates.iloc[[first.query_index]],
            observed_yield=50.0,
        )
        optimizer.recommend(
            public.candidates,
            queried_indices=[first.query_index],
            target_incumbent=float(public.train["Yield"].max()),
            round_index=1,
            total_rounds=4,
        )
        self.assertEqual(optimizer._active_kernel, "rbf")

    def test_categorical_kernel_can_optimize_once_then_freeze(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        optimizer = CategoricalBayesOptimizer(
            variable_columns=VARIABLES,
            categories=public.categories,
            seed=915,
            surrogate="hybrid_gp_forest",
            kernel="categorical_product",
            categorical_optimize_once=True,
            ensemble_size=12,
            n_restarts_optimizer=0,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        optimizer.add_initial_training(public.train)
        first = optimizer.recommend(
            public.candidates,
            queried_indices=[],
            target_incumbent=float(public.train["Yield"].max()),
            round_index=0,
            total_rounds=4,
        )
        self.assertIsNotNone(optimizer._cached_categorical_kernel)
        optimizer.add_feedback(
            public.candidates.iloc[[first.query_index]],
            observed_yield=50.0,
        )
        optimizer.predict(
            public.candidates.drop(index=first.query_index).reset_index(drop=True)
        )
        self.assertIsNone(optimizer._model["gp"].optimizer)

    def test_categorical_kernel_can_target_transferable_condition_columns(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        optimizer = CategoricalBayesOptimizer(
            variable_columns=VARIABLES,
            categories=public.categories,
            seed=916,
            surrogate="hybrid_identity_gp_forest",
            kernel="categorical_product",
            categorical_kernel_columns=("Ligand", "Base"),
            ensemble_size=12,
            n_restarts_optimizer=0,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        # Constant amplitude + two categorical rho values + white noise.
        self.assertEqual(len(optimizer._make_gp_kernel().theta), 4)
        with self.assertRaisesRegex(ValueError, "unknown/non-identity"):
            CategoricalBayesOptimizer(
                variable_columns=VARIABLES,
                categories=public.categories,
                seed=917,
                kernel="categorical_product",
                categorical_kernel_columns=("not_a_variable",),
            )

    def test_dual_forest_novelty_gate_preserves_known_level_local_prediction(self) -> None:
        class TinyFeaturizer:
            def transform(self, frame: pd.DataFrame) -> np.ndarray:
                return np.asarray(
                    [[len(str(row["Product"])), len(str(row["Additive"]))] for _, row in frame.iterrows()],
                    dtype=float,
                )

        variables = ["Product", "Ligand", "Additive"]
        categories = {
            "Product": ["P1", "P2"],
            "Ligand": ["L1", "L2"],
            "Additive": ["A1", "A2"],
        }
        train = pd.DataFrame(
            [
                {"Product": "P1", "Ligand": "L1", "Additive": "A1", "Yield": 20.0},
                {"Product": "P1", "Ligand": "L2", "Additive": "A1", "Yield": 30.0},
                {"Product": "P2", "Ligand": "L1", "Additive": "A1", "Yield": 40.0},
                {"Product": "P2", "Ligand": "L2", "Additive": "A1", "Yield": 50.0},
            ]
        )
        candidates = pd.DataFrame(
            [
                {"Product": "P1", "Ligand": "L1", "Additive": "A1"},
                {"Product": "P1", "Ligand": "L1", "Additive": "A2"},
            ]
        )
        optimizer = CategoricalBayesOptimizer(
            variable_columns=variables,
            categories=categories,
            seed=918,
            surrogate="dual_forest_product_transfer",
            chemical_featurizer=TinyFeaturizer(),
            transfer_chemical_featurizer=TinyFeaturizer(),
            product_transfer_weight=0.5,
            product_transfer_novelty_columns=("Additive",),
            ensemble_size=12,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        optimizer.add_initial_training(train)
        mean, _ = optimizer.predict(candidates)
        local_encoded = optimizer._encode(candidates)
        local_values = np.asarray(
            [
                tree.predict(local_encoded)
                for tree in optimizer._model["local"].estimators_
            ]
        )
        self.assertAlmostEqual(mean[0], float(local_values[:, 0].mean()))

    def test_opening_policies_are_mutually_exclusive(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            CategoricalBayesOptimizer(
                variable_columns=VARIABLES,
                categories=public.categories,
                seed=1,
                greedy_rounds=1,
                top_probability_rounds=1,
            )

    def test_additive_rbf_categorical_kernel_is_finite_and_label_blind(self) -> None:
        public = load_public_dataset(self.root, "Suzuki")

        def predict(candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            optimizer = CategoricalBayesOptimizer(
                variable_columns=VARIABLES,
                categories=public.categories,
                seed=91,
                surrogate="hybrid_gp_forest",
                kernel="rbf_categorical_sum",
                ensemble_size=12,
                n_restarts_optimizer=0,
            )
            optimizer.add_initial_training(public.train)
            return optimizer.predict(candidates)

        baseline_mean, baseline_std = predict(public.candidates)
        labelled = public.candidates.copy()
        labelled["Yield"] = np.linspace(-1e9, 1e9, len(labelled))
        labelled_mean, labelled_std = predict(labelled)
        self.assertTrue(np.isfinite(baseline_mean).all())
        self.assertTrue(np.isfinite(baseline_std).all())
        np.testing.assert_allclose(baseline_mean, labelled_mean)
        np.testing.assert_allclose(baseline_std, labelled_std)


if __name__ == "__main__":
    unittest.main()
