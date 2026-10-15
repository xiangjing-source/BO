from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from dft_features import (
    DFTElectronicFeaturizer,
    DFTObservationPreferenceGuide,
    MechanisticDFTFeaturizer,
)


def _write_table(path: Path, include_yield: bool = False) -> None:
    rows = []
    for variable, name, offset in (
        ("Ligand", "L1", 0.0),
        ("Ligand", "L2", 1.0),
        ("Base", "L1", 2.0),
    ):
        row = {
            "dataset": "demo",
            "original_dataset_column": variable,
            "chemical_name": name,
            "calculation_id": f"C{len(rows)}",
            "homo_ev": -5.0 + offset,
            "lumo_ev": -1.0 + offset,
            "homo_lumo_gap_ev": 4.0,
            "dipole_debye": 1.0 + offset,
            "atom_count": 10 + offset,
            "hirshfeld_charge_min": -0.2,
            "hirshfeld_charge_max": 0.2,
            "hirshfeld_abs_charge_mean": 0.05,
            "mayer_bond_density": 2.0,
            "hirshfeld_charge_std": 0.08 + 0.01 * offset,
            "mayer_atomic_charge_abs_mean": 0.20 + 0.01 * offset,
            "mayer_total_valence_mean": 3.0 + 0.10 * offset,
            "mayer_total_valence_std": 1.2 + 0.05 * offset,
            "mayer_bond_order_mean": 0.6 + 0.02 * offset,
            "mayer_bond_order_max": 1.8 + 0.10 * offset,
        }
        if include_yield:
            row["Yield"] = 99.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_mechanistic_table(
    path: Path, dataset: str = "Suzuki", include_yield: bool = False
) -> None:
    fields = (
        MechanisticDFTFeaturizer.SUZUKI_FIELDS
        if dataset == "Suzuki"
        else MechanisticDFTFeaturizer.BUCHWALD_FIELDS
    )
    all_fields = sorted({field for role_fields in fields.values() for field in role_fields})
    rows = []
    for role_index, (role, role_fields) in enumerate(fields.items()):
        for material_index in range(2):
            row = {
                "dataset": dataset,
                "calculation_id": f"{dataset}_{role_index}_{material_index}",
                "original_dataset_column": role,
                "chemical_name": f"{role}_{material_index}",
            }
            row.update({field: np.nan for field in all_fields})
            for field_index, field in enumerate(role_fields):
                row[field] = float(10 * role_index + material_index + field_index / 10)
            if include_yield:
                row["Yield"] = 99.0
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


class DFTFeatureTests(unittest.TestCase):
    def test_role_aware_lookup_and_missing_indicator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "dft.csv"
            _write_table(feature_path)
            featurizer = DFTElectronicFeaturizer(
                "demo", ["Ligand", "Base"], feature_path
            )
            frame = pd.DataFrame(
                [
                    {"Ligand": "L1", "Base": "L1"},
                    {"Ligand": "missing", "Base": "missing"},
                ]
            )
            matrix = featurizer.transform(frame)
            self.assertEqual(matrix.shape, (2, 20))
            self.assertEqual(matrix[0, 9], 1.0)
            self.assertEqual(matrix[0, 19], 1.0)
            self.assertEqual(matrix[1, 9], 0.0)
            self.assertEqual(matrix[1, 19], 0.0)
            self.assertTrue(np.isfinite(matrix).all())
            # The same chemical name in another role must not reuse Ligand values.
            self.assertFalse(np.allclose(matrix[0, :9], matrix[0, 10:19]))

    def test_extended_descriptor_profile_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "dft.csv"
            _write_table(feature_path)
            featurizer = DFTElectronicFeaturizer(
                "demo",
                ["Ligand", "Base"],
                feature_path,
                feature_columns=DFTElectronicFeaturizer.EXTENDED_FEATURE_COLUMNS,
            )
            matrix = featurizer.transform(
                pd.DataFrame([{"Ligand": "L2", "Base": "missing"}])
            )
            self.assertEqual(matrix.shape, (1, 32))
            self.assertTrue(np.isfinite(matrix).all())

    def test_outcome_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "dft_with_yield.csv"
            _write_table(feature_path, include_yield=True)
            with self.assertRaisesRegex(ValueError, "outcome"):
                DFTElectronicFeaturizer(
                    "demo", ["Ligand", "Base"], feature_path
                )

    def test_preference_guide_ignores_candidate_yield_and_gates_inverse_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "dft.csv"
            _write_table(feature_path)
            featurizer = DFTElectronicFeaturizer(
                "demo", ["Ligand", "Base"], feature_path
            )
            guide = DFTObservationPreferenceGuide(featurizer, neighbors=2)
            observed = pd.DataFrame(
                [
                    {"Ligand": "L1", "Base": "L1"},
                    {"Ligand": "L2", "Base": "L1"},
                ]
                * 3
            )
            candidates = observed.iloc[:2].copy()
            labelled = candidates.copy()
            labelled["Yield"] = [-1e9, 1e9]
            y = np.asarray([0.0, 10.0, 1.0, 9.0, 2.0, 8.0])
            shift, diagnostics = guide.lift(candidates, observed, y, strength=0.5)
            labelled_shift, _ = guide.lift(labelled, observed, y, strength=0.5)
            np.testing.assert_allclose(shift, labelled_shift)
            self.assertGreaterEqual(diagnostics["confidence"], 0.0)

    def test_mechanistic_features_are_role_aware_finite_and_label_blind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "mechanistic.csv"
            _write_mechanistic_table(feature_path)
            variables = list(MechanisticDFTFeaturizer.SUZUKI_FIELDS)
            featurizer = MechanisticDFTFeaturizer(
                "Suzuki", variables, feature_path, profile="compact"
            )
            frame = pd.DataFrame(
                [
                    {role: f"{role}_0" for role in variables},
                    {
                        **{role: f"{role}_1" for role in variables},
                        "Nucleophile": "not_calculated",
                        "Base": "not_calculated",
                    },
                ]
            )
            labelled = frame.copy()
            labelled["Yield"] = [-1e9, 1e9]
            matrix = featurizer.transform(frame)
            np.testing.assert_allclose(matrix, featurizer.transform(labelled))
            self.assertEqual(matrix.shape[1], len(featurizer.feature_names))
            self.assertTrue(np.isfinite(matrix).all())
            coverage = featurizer.coverage(frame)
            self.assertEqual(
                coverage["roles"]["Nucleophile"]["covered_unique_values"], 1
            )
            self.assertEqual(coverage["roles"]["Base"]["covered_unique_values"], 1)

    def test_mechanistic_outcome_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "mechanistic_with_yield.csv"
            _write_mechanistic_table(feature_path, include_yield=True)
            with self.assertRaisesRegex(ValueError, "outcomes"):
                MechanisticDFTFeaturizer(
                    "Suzuki",
                    list(MechanisticDFTFeaturizer.SUZUKI_FIELDS),
                    feature_path,
                    profile="compact",
                )

    def test_buchwald_poisoning_profile_has_declared_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "buchwald_mechanistic.csv"
            _write_mechanistic_table(feature_path, dataset="Buchwald_sub4")
            variables = list(MechanisticDFTFeaturizer.BUCHWALD_FIELDS)
            featurizer = MechanisticDFTFeaturizer(
                "Buchwald_sub4",
                variables,
                feature_path,
                profile="poisoning",
            )
            frame = pd.DataFrame(
                [{role: f"{role}_0" for role in variables}]
            )
            matrix = featurizer.transform(frame)
            self.assertEqual(matrix.shape, (1, 11))
            self.assertEqual(matrix.shape[1], len(featurizer.feature_names))


if __name__ == "__main__":
    unittest.main()
