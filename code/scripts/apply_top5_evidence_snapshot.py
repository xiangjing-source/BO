#!/usr/bin/env python3
"""Apply the reviewed Top-5 evidence snapshot to the submitted agent report.

This post-selection step cannot alter trajectories, observed yields or optimizer ranks.
It keeps the saved DeepSeek assessment as the interpretive baseline, replaces unsupported
claims with ``unknown``, attaches qualifying external evidence, and recomputes the safety
gate according to the submitted evidence policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


DIMENSIONS = (
    "budget",
    "procurement_difficulty",
    "hazard",
    "operational_complexity",
    "scale_up_risk",
    "waste_disposal_burden",
)


POLICY = {
    "Suzuki": {
        1: ("medium", "high", "high", ["LiOtBu is Self-heating Category 1."]),
        2: ("medium", "high", "high", ["LiOtBu is Self-heating Category 1."]),
        3: ("medium", "medium", "medium", []),
        4: ("high", "high", "high", ["LiOtBu is Self-heating Category 1."]),
        5: ("medium", "high", "high", ["Triethylamine is flammable, acutely toxic and corrosive."]),
    },
    "Buchwald_sub4": {
        1: ("medium", "medium", "medium", []),
        2: ("medium", "medium", "medium", []),
        3: ("medium", "medium", "medium", []),
        4: ("medium", "medium", "medium", []),
        5: ("medium", "medium", "medium", []),
    },
}


def load_queried_top5(root: Path) -> dict[str, list[dict[str, object]]]:
    """Rebuild queried Top-5 and DFT context from submitted trajectories only."""
    module_path = root / "skill" / "catalystflow" / "scripts" / "report_builder.py"
    spec = importlib.util.spec_from_file_location("catalystflow_evidence_report_builder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("report_builder module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output: dict[str, list[dict[str, object]]] = {}
    for dataset in module.DATASETS:
        records: list[dict[str, object]] = []
        for seed in module.SEEDS:
            path = root / "results" / "optimization_trajectories" / dataset / f"seed_{seed}.pt"
            payload = module.load_trajectory(path)
            records.extend({**row, "_seed": seed} for row in payload["trajectory"])
        ranked = module.top_conditions(records)
        for item in ranked:
            item["dft_context"] = module.dft_context(root, dataset, item["condition"])
        output[dataset] = ranked
    return output


def update_report(root: Path) -> None:
    report_path = root / "results" / "results_and_analysis.json"
    evidence_path = root / "results" / "top5_external_evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assessment = report["feasibility_assessment"]
    assessment.update(
        {
            "owner": "DeepSeek",
            "evidence_review_owner": "CatalystFlow deterministic evidence policy",
            "basis": "mixed_verified_evidence_llm_inference_unknown",
            "ranking_effect": "none",
            "assessment_context": {
                "target_scale": "not_provided",
                "location": "not_provided",
                "assessment_date": evidence["retrieved_at"],
            },
            "evidence_retrieval": {
                "status": "completed",
                "policy": evidence["retrieval_policy"],
                "source_count": len(evidence["sources"]),
                "snapshot": "results/top5_external_evidence.json",
                "retrieved_at": evidence["retrieved_at"],
                "numerical_ranking_effect": "none",
                "llm_fallback": "saved DeepSeek assessment retained only where qualifying evidence is absent",
            },
        }
    )

    for dataset, rows in assessment["levels_by_rank"].items():
        for row in rows:
            rank = int(row["optimizer_rank"])
            procurement, hazard, operation, flags = POLICY[dataset][rank]
            levels = {
                "budget": "unknown",
                "procurement_difficulty": procurement,
                "hazard": hazard,
                "operational_complexity": operation,
                "scale_up_risk": "unknown",
                "waste_disposal_burden": "medium",
            }
            row.update(levels)
            row["feasibility_total_score"] = None
            row["score_coverage"] = "4/6"
            row["safety_gate"] = "blocked" if flags or hazard == "high" else "conditional"
            row["basis"] = "mixed"
            row["confidence"] = "medium"
            row["basis_by_dimension"] = {
                "budget": "unknown",
                "procurement_difficulty": "verified_evidence",
                "hazard": "verified_evidence",
                "operational_complexity": "llm_inference",
                "scale_up_risk": "unknown",
                "waste_disposal_burden": "llm_inference",
            }
            row["confidence_by_dimension"] = {
                "budget": "low",
                "procurement_difficulty": "medium",
                "hazard": "medium",
                "operational_complexity": "low",
                "scale_up_risk": "low",
                "waste_disposal_burden": "low",
            }
            row["source_ids"] = evidence["candidate_source_map"][dataset][str(rank)]
            row["critical_safety_flags"] = flags

    assessment["evidence_boundary"] = (
        "Supplier catalog and SDS claims are verified only for the cited product, region and retrieval date. "
        "Target scale and location are absent, so budget and scale-up risk remain unknown; operation and waste "
        "statements remain low-confidence LLM inference."
    )
    report["safety_boundary"] = (
        "The external evidence snapshot does not replace current China-region quotation, inventory, full-component "
        "SDS review, laboratory SOP, calorimetry, waste-vendor acceptance or EHS approval."
    )

    final = report["final_recommended_reaction_conditions"]
    final["Suzuki"].update(
        {
            "alternative_optimizer_rank": None,
            "rationale": (
                "Rank 3 is the only non-blocked queried Top-5 condition after SDS review: it avoids the "
                "Self-heating Category 1 LiOtBu used in ranks 1, 2 and 4 and the higher-hazard "
                "triethylamine/acetonitrile combination in rank 5, while retaining 99.05 observed yield."
            ),
            "evidence_basis": "verified supplier catalogs and SDS plus low-confidence LLM fallback",
            "unresolved_information": [
                "Target-scale China-region quotation and inventory",
                "Current SDS for every substrate and ligand",
                "Concentration, addition order, calorimetry and waste-vendor acceptance",
            ],
        }
    )
    final["Buchwald_sub4"].update(
        {
            "rationale": (
                "Rank 1 retains the highest observed yield; its tBuXPhos catalog price is materially below the "
                "retrieved AdBrettPhos price for rank 2, while all five candidates remain conditional because "
                "target-scale budget and scale-up evidence are unavailable."
            ),
            "evidence_basis": "verified supplier catalogs and MTBD SDS plus low-confidence LLM fallback",
            "unresolved_information": [
                "Target-scale China-region quotation for all components",
                "Current SDS for every aryl halide, ligand and additive",
                "Concentration, catalyst loading, calorimetry and waste-vendor acceptance",
            ],
        }
    )
    report["external_evidence_snapshot"] = {
        "path": "results/top5_external_evidence.json",
        "retrieved_at": evidence["retrieved_at"],
        "source_count": len(evidence["sources"]),
        "ranking_effect": "none",
    }
    report["queried_top5"] = load_queried_top5(root)
    report["dft_post_interpretation"] = {
        **evidence["dft_post_interpretation"],
        "computed_artifact_boundary": (
            "Only packaged ORCA artifacts determine computed values, coverage and status; "
            "external sources and DeepSeek provide post-selection interpretation only."
        ),
        "llm_fallback": (
            "Use explicitly labelled low- or medium-confidence inference only when no "
            "directly relevant registry or primary-literature source is available."
        ),
    }
    review_path = root / "results" / "deepseek_database_enhanced_review.json"
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        report["deepseek_database_enhanced_review"] = {
            "path": "results/deepseek_database_enhanced_review.json",
            "framework": review["framework"],
            "model": review["model"],
            "session_id": review["session_id"],
            "review_date": review["review_date"],
            "review_status": review["review_status"],
            "ranking_effect": review["ranking_effect"],
            "database_checks": review["database_checks"],
            "recommendations": review["recommendations"],
            "new_data_boundary": review["new_data_boundary"],
            "required_checks": review["required_checks"],
        }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    args = parser.parse_args()
    update_report(Path(args.package_root).resolve())


if __name__ == "__main__":
    main()
