from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "scripts"))

from dft_explainer import DFTMechanisticExplainer
from explain_dft_condition import build_output


def feature_path() -> Path:
    packaged = (
        PROJECT_ROOT
        / "results"
        / "audits"
        / "mechanistic_dft_component_features.csv"
    )
    if packaged.is_file():
        return packaged
    return (
        CODE_DIR
        / "config"
        / "mechanistic_dft_component_features_20260730_86plus1alias.csv"
    )


SUZUKI_CONDITION = {
    "Electrophile": "6-iodoquinoline",
    "Nucleophile": "[5-methyl-1-(oxan-2-yl)indazol-4-yl]boronic acid",
    "Ligand": "1,1'-Bis(diphenylphosphino)ferrocene",
    "Base": "sodium hydrogen carbonate",
    "Solvent": "N,N-dimethylformamide",
}


class DFTExplainerTests(unittest.TestCase):
    def test_explanation_is_invariant_to_an_outcome_in_condition_mapping(self) -> None:
        explainer = DFTMechanisticExplainer("Suzuki", feature_path())
        baseline = explainer.explain(SUZUKI_CONDITION)
        labelled = dict(SUZUKI_CONDITION, Yield=1e9)
        self.assertEqual(baseline, explainer.explain(labelled))
        self.assertIn("does not affect", baseline["disclaimer"])

    def test_percentiles_are_role_local_and_bounded(self) -> None:
        explainer = DFTMechanisticExplainer("Suzuki", feature_path())
        statistic = explainer.percentile(
            "Electrophile", "6-iodoquinoline", "c_leaving_mayer_bond_order"
        )
        self.assertIsNotNone(statistic)
        self.assertGreaterEqual(statistic["percentile"], 0.0)
        self.assertLessEqual(statistic["percentile"], 100.0)
        self.assertEqual(statistic["reference_n"], 5)

    def test_missing_component_reduces_coverage_without_imputation_claim(self) -> None:
        explainer = DFTMechanisticExplainer("Suzuki", feature_path())
        condition = dict(SUZUKI_CONDITION, Ligand="unregistered ligand")
        explanation = explainer.explain(condition)
        self.assertEqual(explanation["coverage"]["available_roles"], 4)
        ligand = next(
            item for item in explanation["components"] if item["role"] == "Ligand"
        )
        self.assertFalse(ligand["available"])
        self.assertEqual(ligand["metrics"], [])

    def test_outcome_column_in_feature_table_is_rejected(self) -> None:
        table = pd.read_csv(feature_path())
        table["Yield"] = 50.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "outcome columns"):
                DFTMechanisticExplainer("Suzuki", path)

    def test_agent_explanation_preserves_decision_and_has_no_ranking_effect(self) -> None:
        payload = {
            "candidate_index": 193,
            "condition": SUZUKI_CONDITION,
            "predicted_yield": 52.07,
            "uncertainty": 25.65,
            "acquisition_value": 3.32,
            "selection_mode": "expected_improvement",
        }
        output = build_output(payload, "Suzuki", feature_path())
        self.assertEqual(output["ranking_effect"], "none")
        self.assertEqual(output["analysis_phase"], "post_selection_pre_feedback")
        self.assertEqual(output["model_decision"]["candidate_index"], 193)
        self.assertEqual(
            output["dft_mechanistic_explanation"]["condition"],
            SUZUKI_CONDITION,
        )


if __name__ == "__main__":
    unittest.main()
