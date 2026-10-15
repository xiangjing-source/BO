from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


SKILL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_ROOT.parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from report_builder import compare_records, validate_one  # noqa: E402
from workflow_gateway import (  # noqa: E402
    FEASIBILITY_DIMENSIONS,
    WorkflowError,
    WorkflowGateway,
)


class WorkflowGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Path(tempfile.mkdtemp(prefix=".skill-v2-test-", dir=PROJECT_ROOT))
        self.gateway = WorkflowGateway(PROJECT_ROOT, self.runtime)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime)

    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_manifest_fixture(self, candidate_outcome: bool = False) -> str:
        root = self.runtime / "new"
        train = [
            {"Ligand": "L1", "Base": "B1", "Temperature": 40, "Yield": 30},
            {"Ligand": "L2", "Base": "B1", "Temperature": 60, "Yield": 70},
        ]
        candidates: list[dict[str, object]] = [
            {"Ligand": "L1", "Base": "B1", "Temperature": 50},
            {"Ligand": "L3", "Base": "B2", "Temperature": 90},
            {"Ligand": "L3", "Base": "B1", "Temperature": 55},
            {"Ligand": "L1", "Base": "B2", "Temperature": 70},
            {"Ligand": "L3", "Base": "B2", "Temperature": 80},
        ]
        if candidate_outcome:
            for row in candidates:
                row["Yield"] = 99
        self.write_csv(root / "train.csv", train)
        self.write_csv(root / "candidates.csv", candidates)
        mapping = {
            "records": [
                {
                    "variable": "Ligand", "value": value,
                    "review_status": "approved", "canonical_smiles": "CC",
                    "source": "test_fixture",
                }
                for value in ("L1", "L2", "L3")
            ] + [
                {
                    "variable": "Base", "value": value,
                    "review_status": "approved", "canonical_smiles": "N",
                    "source": "test_fixture",
                }
                for value in ("B1", "B2")
            ]
        }
        (root / "structures.json").write_text(json.dumps(mapping), encoding="utf-8")
        manifest = {
            "schema_version": "1.0",
            "reaction_family": "other",
            "train": "train.csv",
            "candidates": "candidates.csv",
            "yield_column": "Yield",
            "categorical_columns": ["Ligand", "Base"],
            "continuous_columns": ["Temperature"],
            "molecular_columns": ["Ligand", "Base"],
            "structure_map": "structures.json",
            "units": {"Temperature": "degC"},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return str(manifest_path.relative_to(PROJECT_ROOT))

    def write_completed_report_job(self, job_id: str) -> Path:
        report_dir = self.runtime / "jobs" / job_id / "report"
        report_dir.mkdir(parents=True)
        recommendations = {
            dataset: [
                {
                    "optimizer_rank": rank,
                    "condition": {"Ligand": f"{dataset}-L{rank}"},
                    "best_observed_yield": 100.0 - rank,
                    "times_queried": rank,
                    "dft_context": {"status": "computed", "ranking_effect": "none"},
                    "feasibility": {
                        dimension: {
                            "level": "unknown",
                            "basis": "unknown",
                            "confidence": "low",
                            "rationale": "awaiting LLM assessment",
                            "sources": [],
                        }
                        for dimension in FEASIBILITY_DIMENSIONS
                    }
                    | {"assessment_status": "awaiting_llm_assessment"},
                }
                for rank in range(1, 6)
            ]
            for dataset in ("Suzuki", "Buchwald_sub4")
        }
        review_request = {
            "status": "awaiting_llm_assessment",
            "assessment_owner": "LLM",
            "datasets": {
                dataset: [
                    {"optimizer_rank": rank, "condition": {"Ligand": f"{dataset}-L{rank}"}}
                    for rank in range(1, 6)
                ]
                for dataset in ("Suzuki", "Buchwald_sub4")
            },
        }
        metric_names = (
            "initial_round_found_best",
            "best_found",
            "round_to_95_global_best",
            "AUC_best_so_far",
        )
        metrics = {
            dataset: [
                {
                    "metric": metric,
                    "mean": float(position),
                    "std": 0.5,
                    "ci95_lower": float(position) - 0.25,
                    "ci95_upper": float(position) + 0.25,
                }
                for position, metric in enumerate(metric_names, start=1)
            ]
            for dataset in ("Suzuki", "Buchwald_sub4")
        }
        (report_dir / "report.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "recommendations": recommendations,
                    "feasibility_review_request": review_request,
                }
            ),
            encoding="utf-8",
        )
        (report_dir / "report.md").write_text("# base\n", encoding="utf-8")
        (report_dir.parent / "status.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        return report_dir

    @staticmethod
    def llm_assessment() -> dict[str, object]:
        candidates = []
        for dataset in ("Suzuki", "Buchwald_sub4"):
            for rank in range(1, 6):
                candidates.append(
                    {
                        "dataset": dataset,
                        "optimizer_rank": rank,
                        **{
                            dimension: {
                                "level": "medium",
                                "basis": "llm_inference",
                                "confidence": "medium",
                                "rationale": f"Preliminary {dimension} judgment for rank {rank}.",
                                "sources": [],
                            }
                            for dimension in FEASIBILITY_DIMENSIONS
                        },
                        "candidate_summary": f"Rank {rank} has a preliminary practical balance requiring verification.",
                        "critical_safety_flags": [],
                    }
                )
        return {
            "assessment_context": {
                "target_scale": "1 mmol",
                "location": "Shanghai, China",
                "assessment_date": "2026-08-13",
            },
            "candidate_assessments": candidates,
            "final_recommendations": [
                {
                    "dataset": "Suzuki",
                    "selected_optimizer_rank": 2,
                    "recommendation_status": "recommended",
                    "rationale": "Rank 2 has the preferred practical balance.",
                    "tradeoffs": ["Slightly lower numerical rank for easier sourcing."],
                    "alternative_optimizer_rank": 1,
                    "alternative_rationale": "Rank 1 remains the yield-led alternative.",
                    "unresolved_information": ["Current supplier stock remains unverified."],
                    "required_checks": ["Verify supplier quote and current SDS."],
                },
                {
                    "dataset": "Buchwald_sub4",
                    "selected_optimizer_rank": 1,
                    "recommendation_status": "recommended",
                    "rationale": "Rank 1 remains the preferred practical balance.",
                    "tradeoffs": ["Specialized ligand needs procurement confirmation."],
                    "alternative_optimizer_rank": 2,
                    "alternative_rationale": "Rank 2 is retained if sourcing changes.",
                    "unresolved_information": ["Current ligand lead time remains unverified."],
                    "required_checks": ["Verify supplier quote and current SDS."],
                },
            ],
        }

    def test_official_bundle_is_recognized_without_oracle_values(self) -> None:
        report = self.gateway.inspect_input("data")
        self.assertTrue(report["ready"])
        self.assertEqual(report["input_kind"], "official_competition_bundle")
        self.assertEqual(report["protocol"]["rounds"], 40)
        self.assertEqual(len(report["protocol"]["seeds"]), 20)
        serialized = json.dumps(report)
        self.assertNotIn("observed_yield", serialized)

    def test_absolute_input_outside_submission_is_denied(self) -> None:
        with self.assertRaises(WorkflowError) as context:
            self.gateway.inspect_input("/tmp/outside-catalystflow-data")
        self.assertEqual(context.exception.code, "path_denied")

    def test_relative_escape_outside_submission_is_denied(self) -> None:
        with self.assertRaises(WorkflowError) as context:
            self.gateway.inspect_input("../outside-catalystflow-data")
        self.assertEqual(context.exception.code, "path_denied")

    def test_unknown_directory_gets_deepseek_draft_then_deterministic_validation(self) -> None:
        source = self.runtime / "unstructured"
        self.write_csv(source / "observations.csv", [
            {"Ligand": "L1", "Base": "B1", "Temperature": 40, "Yield": 30},
            {"Ligand": "L2", "Base": "B1", "Temperature": 60, "Yield": 70},
        ])
        self.write_csv(source / "candidate_pool.csv", [
            {"Ligand": "L1", "Base": "B1", "Temperature": 50},
            {"Ligand": "L2", "Base": "B2", "Temperature": 70},
        ])
        inspection = self.gateway.inspect_input(str(source))
        self.assertEqual(inspection["input_kind"], "new_dataset_directory")
        self.assertFalse(inspection["ready"])
        self.assertEqual(inspection["llm_enrichment_requests"][0]["type"], "manifest_draft")
        drafted = self.gateway.new_dataset(
            action="draft_manifest",
            input_path=str(source),
            manifest_draft={
                "schema_version": "1.0",
                "reaction_family": "other",
                "train": "observations.csv",
                "candidates": "candidate_pool.csv",
                "yield_column": "Yield",
                "categorical_columns": ["Ligand", "Base"],
                "continuous_columns": ["Temperature"],
                "molecular_columns": [],
                "structure_map": "",
                "units": {"Temperature": "degC"},
            },
        )
        self.assertTrue(drafted["ready_for_live_optimization"])
        self.assertEqual(drafted["semantic_parser"], "DeepSeek")

    def test_deepseek_draft_cannot_promote_possible_answer_table_to_training(self) -> None:
        source = self.runtime / "unstructured_with_answer"
        self.write_csv(source / "observations.csv", [
            {"Ligand": "L1", "Base": "B1", "Yield": 30},
            {"Ligand": "L2", "Base": "B1", "Yield": 70},
        ])
        self.write_csv(source / "candidate_pool.csv", [
            {"Ligand": "L1", "Base": "B1"},
            {"Ligand": "L2", "Base": "B2"},
        ])
        self.write_csv(source / "answer.csv", [
            {"Ligand": "L1", "Base": "B1", "Yield": 99},
            {"Ligand": "L2", "Base": "B2", "Yield": 98},
        ])
        inspection = self.gateway.inspect_input(str(source))
        answer = next(
            item for item in inspection["table_inventory"]
            if item["filename"] == "answer.csv"
        )
        self.assertIn("possible_answer_table_do_not_use", answer["role_hints"])
        with self.assertRaises(WorkflowError) as context:
            self.gateway.new_dataset(
                action="draft_manifest",
                input_path=str(source),
                manifest_draft={
                    "schema_version": "1.0",
                    "reaction_family": "other",
                    "train": "answer.csv",
                    "candidates": "candidate_pool.csv",
                    "yield_column": "Yield",
                    "categorical_columns": ["Ligand", "Base"],
                    "continuous_columns": [],
                    "molecular_columns": [],
                    "structure_map": "",
                    "units": {},
                },
            )
        self.assertEqual(context.exception.code, "possible_answer_table")

    def test_direct_manifest_cannot_use_possible_answer_table_as_training(self) -> None:
        manifest_path = self.write_manifest_fixture()
        manifest = PROJECT_ROOT / manifest_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        original = manifest.parent / payload["train"]
        answer = manifest.parent / "answer.csv"
        original.rename(answer)
        payload["train"] = answer.name
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = self.gateway.profile_new_dataset(manifest_path)
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertTrue(any("answer/oracle" in item for item in report["blockers"]))

    def test_direct_manifest_custom_outcome_name_still_blocks_answer_table(self) -> None:
        manifest_path = self.write_manifest_fixture()
        manifest = PROJECT_ROOT / manifest_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        original = manifest.parent / payload["train"]
        headers, rows = [], []
        with original.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = ["Score" if item == "Yield" else item for item in (reader.fieldnames or [])]
            for row in reader:
                rows.append({("Score" if key == "Yield" else key): value for key, value in row.items()})
        answer = manifest.parent / "answer.csv"
        with answer.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        original.unlink()
        payload["train"] = answer.name
        payload["yield_column"] = "Score"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        report = self.gateway.profile_new_dataset(manifest_path)
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertTrue(any("answer/oracle" in item for item in report["blockers"]))

    def test_deepseek_draft_custom_outcome_name_blocks_answer_table(self) -> None:
        source = self.runtime / "unstructured_custom_answer"
        self.write_csv(source / "answer.csv", [
            {"Ligand": "L1", "Base": "B1", "Score": 30},
            {"Ligand": "L2", "Base": "B1", "Score": 70},
        ])
        self.write_csv(source / "candidate_pool.csv", [
            {"Ligand": "L1", "Base": "B1"},
            {"Ligand": "L2", "Base": "B2"},
        ])
        with self.assertRaises(WorkflowError) as context:
            self.gateway.new_dataset(
                action="draft_manifest",
                input_path=str(source),
                manifest_draft={
                    "schema_version": "1.0",
                    "reaction_family": "other",
                    "train": "answer.csv",
                    "candidates": "candidate_pool.csv",
                    "yield_column": "Score",
                    "categorical_columns": ["Ligand", "Base"],
                    "continuous_columns": [],
                    "molecular_columns": [],
                    "structure_map": "",
                    "units": {},
                },
            )
        self.assertEqual(context.exception.code, "possible_answer_table")

    def test_empty_continuous_candidate_table_returns_structured_blocker(self) -> None:
        manifest_path = self.write_manifest_fixture()
        manifest = PROJECT_ROOT / manifest_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        candidates = manifest.parent / payload["candidates"]
        candidates.write_text("Ligand,Base,Temperature\n", encoding="utf-8")
        report = self.gateway.profile_new_dataset(manifest_path)
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertIn("candidate table is empty", report["blockers"])

    def test_numeric_equivalent_continuous_conditions_are_duplicates(self) -> None:
        manifest_path = self.write_manifest_fixture()
        manifest = PROJECT_ROOT / manifest_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        candidates = manifest.parent / payload["candidates"]
        self.write_csv(candidates, [
            {"Ligand": "L1", "Base": "B1", "Temperature": "50"},
            {"Ligand": "L1", "Base": "B1", "Temperature": "50.0"},
            {"Ligand": "L3", "Base": "B2", "Temperature": "90"},
            {"Ligand": "L3", "Base": "B1", "Temperature": "55"},
            {"Ligand": "L1", "Base": "B2", "Temperature": "70"},
        ])
        report = self.gateway.profile_new_dataset(manifest_path)
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertTrue(any("duplicate full conditions" in item for item in report["blockers"]))

    def test_new_dataset_reports_unseen_categories_and_ood(self) -> None:
        report = self.gateway.profile_new_dataset(self.write_manifest_fixture())
        self.assertTrue(report["ready_for_live_optimization"])
        self.assertEqual(report["diagnostics"]["ood_risk"], "high")
        self.assertEqual(
            report["diagnostics"]["categorical_shift"]["Ligand"]["unseen_levels"],
            ["L3"],
        )
        self.assertEqual(report["generalization_claim"], "not_established")
        self.assertTrue(report["llm_enrichment_requests"])
        temperature = report["diagnostics"]["continuous_shift"]["Temperature"]
        self.assertIn("beyond_three_robust_deviations_fraction", temperature)

    def test_pending_structure_review_blocks_initialization(self) -> None:
        manifest_path = self.write_manifest_fixture()
        manifest = PROJECT_ROOT / manifest_path
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        mapping = manifest.parent / payload["structure_map"]
        records = json.loads(mapping.read_text(encoding="utf-8"))
        records["records"][0]["review_status"] = "pending_human_review"
        records["records"][0]["status"] = "resolved_pending_review"
        mapping.write_text(json.dumps(records), encoding="utf-8")
        report = self.gateway.profile_new_dataset(manifest_path)
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertTrue(any("structure map lacks" in item for item in report["blockers"]))

    def test_generic_molecular_route_uses_reviewed_structure_features(self) -> None:
        manifest = self.write_manifest_fixture()
        initialized = self.gateway.new_dataset(
            action="init", manifest_path=manifest, rounds=5, seed=123
        )
        self.assertEqual(
            initialized["route"], "generic_mixed_identity_structure_adapter"
        )
        recommendation = self.gateway.new_dataset(
            action="recommend", run_id=initialized["run_id"]
        )
        self.assertEqual(
            recommendation["method"],
            "generic_mixed_identity_structure_gp_forest_EI",
        )
        self.assertTrue(recommendation["transaction_id"])

    def test_new_dataset_requires_at_least_five_rounds_for_top5(self) -> None:
        with self.assertRaises(WorkflowError) as context:
            self.gateway.new_dataset(
                action="init",
                manifest_path=self.write_manifest_fixture(),
                rounds=4,
                seed=123,
            )
        self.assertEqual(context.exception.code, "invalid_round_budget")

    def test_candidate_outcome_column_blocks_new_dataset(self) -> None:
        report = self.gateway.profile_new_dataset(self.write_manifest_fixture(True))
        self.assertFalse(report["ready_for_live_optimization"])
        self.assertTrue(any("outcome-like" in item for item in report["blockers"]))

    def test_feedback_requires_transaction_id_before_subprocess(self) -> None:
        run = self.runtime / "live" / "example"
        run.mkdir(parents=True)
        (run / "state.json").write_text(
            json.dumps({"pending": {"transaction_id": "correct"}}), encoding="utf-8"
        )
        with self.assertRaises(WorkflowError) as context:
            self.gateway.new_dataset(
                action="feedback",
                run_id="example",
                observed_yield=50.0,
                transaction_id="wrong",
            )
        self.assertEqual(context.exception.code, "transaction_mismatch")

    def test_lost_reproduction_worker_is_failed_closed(self) -> None:
        job = self.runtime / "jobs" / "lost-job"
        job.mkdir(parents=True)
        (job / "status.json").write_text(
            json.dumps(
                {
                    "job_id": "lost-job",
                    "status": "running",
                    "stage": "optimization_2x20x40",
                    "pid": 999999999,
                    "updated_at_utc": "2000-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        status = self.gateway.reproduction_status("lost-job")
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_type"], "WorkerHeartbeatExpired")

    def test_invalid_reproduction_execution_mode_is_rejected(self) -> None:
        with self.assertRaises(WorkflowError) as context:
            self.gateway.start_reproduction(execution_mode="background-ish")
        self.assertEqual(context.exception.code, "invalid_execution_mode")

    def test_prepare_and_finalize_llm_top5_feasibility_report(self) -> None:
        job_id = "completed-test"
        report_dir = self.write_completed_report_job(job_id)
        status = self.gateway.reproduction_status(job_id)
        self.assertNotIn("report", status)
        self.assertFalse(status["report_embedded"])
        self.assertEqual(status["summary_metrics_status"], "available")
        self.assertEqual(
            set(status["summary_metrics"]["datasets"]["Suzuki"]),
            {"B1", "B40", "t95", "AUC"},
        )
        self.assertEqual(
            status["summary_metrics"]["datasets"]["Suzuki"]["B40"]["mean"],
            2.0,
        )
        self.assertNotIn("y_global", status["summary_metrics"])
        self.assertLess(len(json.dumps(status, ensure_ascii=False).encode("utf-8")), 8192)
        request = self.gateway.prepare_feasibility_review(job_id)
        self.assertEqual(request["status"], "ready_for_llm_assessment")
        self.assertEqual(len(request["top5"]["Suzuki"]), 5)
        self.assertLess(len(json.dumps(request, ensure_ascii=False).encode("utf-8")), 20000)
        self.assertEqual(
            request["top5"]["Suzuki"][0]["dft_context"]["ranking_effect"],
            "none",
        )
        self.assertIn("full_dft_context_location", request)
        self.assertIn("candidate_summary", request["template"]["candidate_required_fields"])
        self.assertIn("required_checks", request["template"]["final_recommendation_required_fields"])
        self.assertEqual(
            request["evidence_workflow"]["sequence"],
            ["science_list_dbs", "science_search", "science_fetch", "provenance_record"],
        )
        self.assertEqual(request["evidence_workflow"]["ranking_effect"], "none")
        result = self.gateway.finalize_report(job_id, self.llm_assessment())
        self.assertTrue(result["valid"])
        self.assertEqual(result["candidate_assessments"], 10)
        self.assertEqual(result["final_recommendations"], 2)
        enriched = json.loads(
            (report_dir / "enriched_report.json").read_text(encoding="utf-8")
        )
        top = enriched["numerical_report"]["recommendations"]["Suzuki"]
        self.assertEqual(top[0]["feasibility"]["procurement_difficulty"]["level"], "medium")
        final = enriched["numerical_report"]["practical_recommendation"]["Suzuki"]
        self.assertEqual(final["selected_optimizer_rank"], 2)
        self.assertEqual(final["condition"], {"Ligand": "Suzuki-L2"})
        self.assertTrue(enriched["optimizer_ranking_preserved"])

    def test_incremental_top5_assessment_is_resumable_and_finalizes(self) -> None:
        job_id = "incremental-test"
        report_dir = self.write_completed_report_job(job_id)
        assessment = self.llm_assessment()
        context = assessment["assessment_context"]
        candidates = assessment["candidate_assessments"]
        for index, candidate in enumerate(candidates, start=1):
            result = self.gateway.finalize_report(
                job_id,
                assessment_context=context,
                candidate_assessment=candidate,
            )
            self.assertEqual(result["received_count"], index)
            self.assertEqual(len(result["remaining"]), 10 - index)
        repeated = self.gateway.finalize_report(
            job_id,
            assessment_context=context,
            candidate_assessment=candidates[-1],
        )
        self.assertTrue(repeated["idempotent"])
        final = self.gateway.finalize_report(
            job_id,
            final_recommendations=assessment["final_recommendations"],
        )
        self.assertTrue(final["valid"])
        self.assertEqual(final["candidate_assessments"], 10)
        self.assertTrue((report_dir / "enriched_report.json").is_file())

    def test_finalize_report_rejects_wrong_source_tier_for_hazard(self) -> None:
        job_id = "source-tier-test"
        self.write_completed_report_job(job_id)
        evidence = self.llm_assessment()
        first = evidence["candidate_assessments"][0]
        first["hazard"] = {
            "level": "low",
            "basis": "verified_evidence",
            "confidence": "high",
            "rationale": "Registry-only hazard claim.",
            "sources": [{
                "type": "chemical_registry",
                "title": "Registry",
                "locator": "https://example.invalid/registry",
                "retrieved_at": "2026-08-13",
                "claim": "The registry identifies the compound but is not an SDS.",
            }],
        }
        with self.assertRaises(WorkflowError) as context:
            self.gateway.finalize_report(job_id, evidence)
        self.assertEqual(context.exception.code, "insufficient_source_tier")

    def test_program_blocks_high_hazard_final_selection(self) -> None:
        job_id = "blocked-selection-test"
        self.write_completed_report_job(job_id)
        evidence = self.llm_assessment()
        candidate = evidence["candidate_assessments"][1]
        candidate["hazard"] = {
            "level": "high",
            "basis": "llm_inference",
            "confidence": "medium",
            "rationale": "A material preliminary hazard concern is present.",
            "sources": [],
        }
        with self.assertRaises(WorkflowError) as context:
            self.gateway.finalize_report(job_id, evidence)
        self.assertEqual(context.exception.code, "unsafe_final_recommendation")

    def test_completed_live_run_gets_llm_top5_report_and_one_final_condition(self) -> None:
        run_id = "completed-live-test"
        run_root = self.runtime / "live" / run_id
        run_root.mkdir(parents=True)
        feedback = [
            {
                "round": rank,
                "candidate_index": 100 + rank,
                "condition": {"Ligand": f"L{rank}", "Base": "B1"},
                "observed_yield": float(80 + rank),
                "predicted_yield": float(70 + rank),
                "uncertainty": 2.0,
                "acquisition_value": 1.0,
                "selection_mode": "expected_improvement",
            }
            for rank in range(1, 6)
        ]
        (run_root / "state.json").write_text(
            json.dumps({"pending": None, "round_budget": 5, "feedback": feedback}),
            encoding="utf-8",
        )
        request = self.gateway.prepare_live_feasibility_review(run_id)
        self.assertEqual([item["candidate_index"] for item in request["top5"]], [105, 104, 103, 102, 101])
        self.assertIn("candidate_summary", request["candidate_assessment_template"])
        self.assertEqual(request["top5"][0]["dft_context"]["status"], "unavailable")
        dimension = {
            "level": "medium",
            "basis": "llm_inference",
            "confidence": "medium",
            "rationale": "Preliminary LLM assessment pending supplier and EHS checks.",
            "sources": [],
        }
        assessment = {
            "assessment_context": {
                "target_scale": "1 mmol",
                "location": "Shanghai, China",
                "assessment_date": "2026-08-13",
            },
            "candidate_assessments": [
                {
                    "optimizer_rank": rank,
                    **{name: dict(dimension) for name in FEASIBILITY_DIMENSIONS},
                    "candidate_summary": "Preliminary practical assessment pending source verification.",
                    "critical_safety_flags": [],
                }
                for rank in range(1, 6)
            ],
            "final_recommendation": {
                "selected_optimizer_rank": 2,
                "recommendation_status": "recommended",
                "rationale": "Rank 2 gives the preferred practical balance.",
                "tradeoffs": ["One rank below the highest observed-yield condition."],
                "alternative_optimizer_rank": 1,
                "alternative_rationale": "Rank 1 remains the yield-led alternative.",
                "unresolved_information": ["Current quote and scale-up data remain unavailable."],
                "required_checks": ["Confirm supplier quote, current SDS, SOP and EHS approval."],
            },
        }
        result = self.gateway.finalize_live_feasibility(run_id, assessment)
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_optimizer_rank"], 2)
        self.assertEqual(result["final_condition"], {"Ligand": "L4", "Base": "B1"})
        report = json.loads((run_root / "feasibility_report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["top5"]), 5)
        self.assertTrue(report["optimizer_ranking_preserved"])
        self.assertEqual(report["top5"][0]["feasibility"]["computed_assessment"]["feasibility_total_score"], 12)
        self.assertEqual(report["top5"][0]["feasibility"]["computed_assessment"]["safety_gate"], "conditional")

    def test_all_blocked_conditions_allow_no_recommendation(self) -> None:
        run_id = "all-blocked-live-test"
        run_root = self.runtime / "live" / run_id
        run_root.mkdir(parents=True)
        feedback = [
            {
                "round": rank,
                "candidate_index": rank,
                "condition": {"Ligand": f"L{rank}"},
                "observed_yield": float(90 - rank),
            }
            for rank in range(1, 6)
        ]
        (run_root / "state.json").write_text(
            json.dumps({"pending": None, "round_budget": 5, "feedback": feedback}),
            encoding="utf-8",
        )
        dimension = {
            "level": "medium",
            "basis": "llm_inference",
            "confidence": "medium",
            "rationale": "Preliminary assessment.",
            "sources": [],
        }
        candidates = []
        for rank in range(1, 6):
            item = {
                "optimizer_rank": rank,
                **{name: dict(dimension) for name in FEASIBILITY_DIMENSIONS},
                "candidate_summary": "A critical issue requires rejection.",
                "critical_safety_flags": ["Critical incompatibility requires investigation."],
            }
            candidates.append(item)
        result = self.gateway.finalize_live_feasibility(run_id, {
            "assessment_context": {
                "target_scale": "not_provided",
                "location": "not_provided",
                "assessment_date": "2026-08-13",
            },
            "candidate_assessments": candidates,
            "final_recommendation": {
                "recommendation_status": "no_condition_recommended",
                "selected_optimizer_rank": None,
                "rationale": "All five conditions are blocked by critical safety flags.",
                "tradeoffs": ["No yield advantage justifies bypassing the safety gate."],
                "alternative_optimizer_rank": None,
                "alternative_rationale": "No non-blocked alternative is available.",
                "unresolved_information": ["Critical incompatibilities require resolution."],
                "required_checks": ["Perform an independent process-safety review."],
            },
        })
        self.assertIsNone(result["selected_optimizer_rank"])
        self.assertIsNone(result["final_condition"])

    def test_trajectory_comparison_has_tolerant_numeric_and_exact_identity(self) -> None:
        record = {
            "step": 1,
            "query_index": 7,
            "condition": {"Ligand": "L1"},
            "selection_mode": "expected_improvement",
            "observed_yield": 80.0,
            "best_so_far": 80.0,
            "predicted_yield": 75.0,
            "uncertainty": 2.0,
            "acquisition_value": 1.0,
        }
        changed = dict(record)
        changed["predicted_yield"] = 75.0 + 1e-12
        state, differences = compare_records([changed], [record])
        self.assertEqual(state, "numerically_equivalent")
        self.assertEqual(differences, [])
        changed["query_index"] = 8
        state, differences = compare_records([changed], [record])
        self.assertEqual(state, "different")
        self.assertEqual(differences[0]["field"], "query_index")

    def test_trajectory_audit_checks_condition_against_candidate_index(self) -> None:
        path = PROJECT_ROOT / "results" / "optimization_trajectories" / "Suzuki" / "seed_100.pt"
        import pickle

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        candidate_rows = [
            {} for _ in range(max(record["query_index"] for record in payload["trajectory"]) + 1)
        ]
        for record in payload["trajectory"]:
            candidate_rows[record["query_index"]] = dict(record["condition"])
        first_index = payload["trajectory"][0]["query_index"]
        candidate_rows[first_index] = {
            key: "wrong" for key in candidate_rows[first_index]
        }
        _, errors = validate_one(path, "Suzuki", 100, candidate_rows=candidate_rows)
        self.assertTrue(any("condition/query-index mismatch" in item for item in errors))

    def test_trajectory_audit_rejects_missing_condition_field(self) -> None:
        source = PROJECT_ROOT / "results" / "optimization_trajectories" / "Suzuki" / "seed_100.pt"
        import pickle

        with source.open("rb") as handle:
            payload = pickle.load(handle)
        payload["trajectory"][0]["condition"].pop("Solvent")
        target = self.runtime / "missing-condition.pt"
        with target.open("wb") as handle:
            pickle.dump(payload, handle)
        _, errors = validate_one(target, "Suzuki", 100)
        self.assertTrue(any("condition fields mismatch" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
