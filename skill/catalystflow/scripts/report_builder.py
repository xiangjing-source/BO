#!/usr/bin/env python3
"""Audit fresh CatalystFlow trajectories and build deterministic JSON/Markdown reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import math
from pathlib import Path
import pickle
import platform
import sys
from typing import Any, Mapping, Sequence


DATASETS = ("Suzuki", "Buchwald_sub4")
DATASET_VARIABLES = {
    "Suzuki": {"Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"},
    "Buchwald_sub4": {"Reactant2", "Ligand", "Additive", "Base", "Product"},
}
SEEDS = tuple(range(100, 2001, 100))
ROUNDS = 40
FLOAT_FIELDS = ("observed_yield", "best_so_far", "predicted_yield", "uncertainty", "acquisition_value")
EXACT_FIELDS = ("step", "query_index", "condition", "selection_mode")
FEASIBILITY_DIMENSIONS = (
    "budget",
    "procurement_difficulty",
    "hazard",
    "operational_complexity",
    "scale_up_risk",
    "waste_disposal_burden",
)
FEASIBILITY_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "feasibility_assessment_template.json"
)


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # noqa: ARG002
        raise pickle.UnpicklingError("global objects are not allowed")


def load_trajectory(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = PrimitiveUnpickler(handle).load()
    if not isinstance(value, dict):
        raise ValueError(f"trajectory root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_one(
    path: Path,
    dataset: str,
    seed: int,
    candidate_rows: Sequence[Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = load_trajectory(path)
    errors: list[str] = []
    if payload.get("dataset") != dataset:
        errors.append("dataset metadata mismatch")
    if int(payload.get("seed", -1)) != seed:
        errors.append("seed metadata mismatch")
    records = payload.get("trajectory")
    if not isinstance(records, list) or len(records) != ROUNDS:
        errors.append(f"expected {ROUNDS} records")
        return [], errors
    seen: set[int] = set()
    previous_best = -math.inf
    for expected, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"round {expected} is not an object")
            continue
        if int(record.get("step", -1)) != expected:
            errors.append(f"round {expected} step mismatch")
        index = int(record.get("query_index", -1))
        if index < 0 or index in seen:
            errors.append(f"round {expected} invalid/repeated query index")
        seen.add(index)
        observed = finite_or_none(record.get("observed_yield"))
        best = finite_or_none(record.get("best_so_far"))
        if observed is None or best is None or best + 1e-12 < observed or best + 1e-12 < previous_best:
            errors.append(f"round {expected} invalid observed/best-so-far")
        if best is not None:
            previous_best = best
        if not isinstance(record.get("condition"), dict) or not record["condition"]:
            errors.append(f"round {expected} missing condition")
        elif set(record["condition"]) != DATASET_VARIABLES[dataset]:
            errors.append(f"round {expected} condition fields mismatch")
        elif candidate_rows is not None:
            if index >= len(candidate_rows):
                errors.append(f"round {expected} query index outside candidate table")
            else:
                expected_condition = {
                    key: candidate_rows[index].get(key)
                    for key in DATASET_VARIABLES[dataset]
                }
                if record["condition"] != expected_condition:
                    errors.append(f"round {expected} condition/query-index mismatch")
    return records, errors


def compare_records(fresh: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if len(fresh) != len(reference):
        return "different", [{"field": "trajectory_length", "fresh": len(fresh), "reference": len(reference)}]
    exact = True
    mismatches: list[dict[str, Any]] = []
    for position, (left, right) in enumerate(zip(fresh, reference), start=1):
        for field in EXACT_FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append({"round": position, "field": field, "fresh": left.get(field), "reference": right.get(field)})
        for field in FLOAT_FIELDS:
            a, b = finite_or_none(left.get(field)), finite_or_none(right.get(field))
            if a is None or b is None:
                if a != b:
                    mismatches.append({"round": position, "field": field, "fresh": a, "reference": b})
                continue
            if a != b:
                exact = False
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
                mismatches.append({"round": position, "field": field, "fresh": a, "reference": b})
        if len(mismatches) >= 50:
            break
    if mismatches:
        return "different", mismatches[:50]
    return ("identical" if exact else "numerically_equivalent"), []


def load_summary(path: Path, dataset: str | None = None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, Any]] = []
    for row in rows:
        if dataset is not None and row.get("dataset") != dataset:
            continue
        result.append({
            "metric": row["metric"],
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "ci95_lower": float(row["ci95_lower"]),
            "ci95_upper": float(row["ci95_upper"]),
        })
    return result


def top_conditions(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        condition = dict(record["condition"])
        key = json.dumps(condition, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        group = groups.setdefault(key, {"condition": condition, "values": [], "seeds": [], "rounds": []})
        group["values"].append(float(record["observed_yield"]))
        group["seeds"].append(int(record["_seed"]))
        group["rounds"].append(int(record["step"]))
    ranked = sorted(
        groups.values(),
        key=lambda item: (-max(item["values"]), -sum(item["values"]) / len(item["values"]), -len(item["values"]), json.dumps(item["condition"], sort_keys=True)),
    )[:5]
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        output.append({
            "optimizer_rank": rank,
            "ranking_scope": "unique conditions queried across 20 completed trajectories",
            "condition": item["condition"],
            "best_observed_yield": max(item["values"]),
            "mean_observed_yield_when_queried": sum(item["values"]) / len(item["values"]),
            "times_queried": len(item["values"]),
            "seeds": sorted(set(item["seeds"])),
            "earliest_query_round": min(item["rounds"]),
        })
    return output


def dft_context(project_root: Path, dataset: str, condition: Mapping[str, Any]) -> dict[str, Any]:
    module_path = project_root / "code" / "dft_explainer.py"
    table = project_root / "results" / "audits" / "mechanistic_dft_component_features.csv"
    if not module_path.is_file() or not table.is_file():
        return {"status": "unavailable", "ranking_effect": "none", "coverage": None, "confidence": None}
    try:
        spec = importlib.util.spec_from_file_location("catalystflow_report_dft", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("module loader unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = module.DFTMechanisticExplainer(dataset, table).explain(condition)
        coverage = result["coverage"]
        status = "computed" if coverage["fraction"] == 1.0 else "partial" if coverage["fraction"] > 0 else "not_computed"
        return {
            "status": status,
            "ranking_effect": "none",
            "coverage": coverage,
            "confidence": result["confidence"],
            "support_evidence": result["support_evidence"],
            "risk_flags": result["risk_flags"],
            "context_notes": result["context_notes"],
            "disclaimer": result["disclaimer"],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "ranking_effect": "none",
            "coverage": None,
            "confidence": None,
            "reason": f"{type(exc).__name__}: DFT adapter failed",
        }


def environment_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "pandas", "scikit-learn", "rdkit"):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def empty_feasibility_dimension() -> dict[str, Any]:
    return {
        "level": "unknown",
        "basis": "unknown",
        "confidence": "low",
        "rationale": "awaiting DeepSeek assessment",
        "sources": [],
    }


def feasibility_review_request(
    recommendations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    template = json.loads(FEASIBILITY_TEMPLATE.read_text(encoding="utf-8"))
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        datasets[dataset] = [
            {
                "optimizer_rank": int(item["optimizer_rank"]),
                "condition": dict(item["condition"]),
                **{
                    dimension: empty_feasibility_dimension()
                    for dimension in FEASIBILITY_DIMENSIONS
                },
                "candidate_summary": "awaiting DeepSeek assessment",
                "critical_safety_flags": [],
            }
            for item in recommendations[dataset]
        ]
    return {
        "schema_version": template["schema_version"],
        "status": "awaiting_llm_assessment",
        "assessment_owner": "DeepSeek",
        "assessment_context": template["assessment_context"],
        "assessment_scope": template["assessment_scope"],
        "allowed_values": template["allowed_values"],
        "dimension_definitions": template["dimension_definitions"],
        "scoring_policy": template["scoring_policy"],
        "safety_gate_policy": template["safety_gate_policy"],
        "program_computed_output": template["program_computed_output"],
        "datasets": datasets,
        "candidate_assessment_template": template["candidate_assessments"][0],
        "final_recommendation_template": template["final_recommendations"][0],
        "required_output": (
            "Assess all six dimensions for ranks 1..5. The program derives dimension "
            "scores, feasibility_total_score, score_coverage, and safety_gate. Select one non-blocked "
            "final practical condition per dataset, or return no_condition_recommended "
            "when all five conditions are blocked. Do not alter optimizer_rank."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CatalystFlow 优化结果与已查询 Top-5 现实可行性报告",
        "",
        f"生成时间：{report['created_at_utc']}",
        "",
        "## 结论",
        "",
        f"本次由最终模型重新生成 {report['trajectory_audit']['trajectory_files']} 条轨迹、"
        f"共 {report['trajectory_audit']['query_records']} 条顺序查询记录。"
        f"轨迹审计状态为 **{report['trajectory_audit']['status']}**，与提交基准的比较为 "
        f"**{report['reference_comparison']['status']}**。",
        "",
        "DeepSeek 与 DFT 均未参与产率预测、EI 或候选排序；它们只用于查询完成后的证据与解释。",
        "",
        "## 复现协议与环境",
        "",
        "- 两个数据集各 20 个固定随机种子，每个种子 40 轮，单轮只查询一个候选。",
        f"- 固定模型配置 SHA-256：`{report['provenance']['production_config_sha256']}`。",
        f"- Python：{report['environment']['python']}；scikit-learn：{report['environment']['packages']['scikit-learn']}；RDKit：{report['environment']['packages']['rdkit']}。",
        "",
        "## 核心优化指标",
        "",
        "| 数据集 | 指标 | 均值 | 标准差 | 95% 置信区间 |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for row in report["metrics"][dataset]:
            lines.append(
                f"| {dataset} | {row['metric']} | {row['mean']:.6f} | {row['std']:.6f} | "
                f"[{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
            )
    lines += ["", "## 已查询 Top-5 与 DFT 解释（不参与候选排序）", ""]
    for dataset in DATASETS:
        lines += [f"### {dataset}", ""]
        for item in report["recommendations"][dataset]:
            condition = "；".join(f"{key}={value}" for key, value in item["condition"].items())
            dft = item["dft_context"]
            fraction = dft.get("coverage", {}).get("fraction") if isinstance(dft.get("coverage"), dict) else None
            lines.append(
                f"{item['optimizer_rank']}. {condition}。已查询最高产率 {item['best_observed_yield']:.4f}，"
                f"被查询 {item['times_queried']} 次；DFT 状态 {dft['status']}，覆盖率 "
                f"{fraction if fraction is not None else 'unknown'}，排序影响 none。"
            )
        lines.append("")
    lines += [
        "## 现实可行性状态",
        "",
        "本节评估对象是已查询 Top-5 反应条件，不是复现过程或软件运行的可行性。最终模型"
        "只负责产生已查询 Top-5；数值运行结束后，DeepSeek 必须逐项评估预算、采购难度、"
        "初步危险性、操作复杂度、放大风险和废物处置负担。各维固定按 low=3、medium=2、"
        "high=1、unknown=null 等权计分；任一维度未知时不计算总分。观测产率不计入现实可行性"
        "总分，程序计算安全门槛，DeepSeek 不得改写优化器排名。此处当前状态为 "
        "`awaiting_llm_assessment`；DeepSeek 推断必须显式标记，初步危险性结论仍需 SDS、SOP 与 "
        "EHS 复核。",
        "",
        "## 限制与安全边界",
        "",
        "- 已查询 Top-5 只在 20 条已完成轨迹实际查询过的条件中汇总，不是对未实验候选的额外预测。",
        "- DFT 是角色内百分位机理语境，不是因果证明，也不改变最终模型排序。",
        "- 比赛数据上的复现结果不构成对新反应族的性能保证；新数据必须先通过 OOD 与结构审计。",
        "- 最终实验安全决策必须遵循实验室 SOP，并由 EHS 人员审核。",
        "",
    ]
    return "\n".join(lines)


def build(
    project_root: Path,
    run_root: Path,
    report_root: Path,
    reference_root: Path | None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    input_data_root = (data_root or project_root / "data").resolve()
    if not input_data_root.is_dir():
        raise SystemExit(f"input data root does not exist: {input_data_root}")
    all_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_hashes: dict[str, str] = {}
    candidate_table_hashes: dict[str, str] = {}
    errors: list[dict[str, Any]] = []
    comparison_counts = {"identical": 0, "numerically_equivalent": 0, "different": 0, "not_compared": 0}
    mismatches: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidate_name = (
            "suzuki_test_features.csv"
            if dataset == "Suzuki"
            else "buchwald_sub4_test_features.csv"
        )
        candidate_path = input_data_root / dataset / candidate_name
        if not candidate_path.is_file():
            raise SystemExit(f"input candidate table does not exist: {candidate_path}")
        candidate_table_hashes[dataset] = sha256_file(candidate_path)
        with candidate_path.open(newline="", encoding="utf-8-sig") as handle:
            candidate_rows = list(csv.DictReader(handle))
        for seed in SEEDS:
            path = run_root / dataset / "optimization_trajectories" / dataset / f"seed_{seed}.pt"
            if not path.is_file():
                path = run_root / "optimization_trajectories" / dataset / f"seed_{seed}.pt"
            if not path.is_file():
                errors.append({"dataset": dataset, "seed": seed, "error": "trajectory missing"})
                continue
            records, local_errors = validate_one(
                path, dataset, seed, candidate_rows=candidate_rows
            )
            trajectory_hashes[f"{dataset}/seed_{seed}.pt"] = sha256_file(path)
            for error in local_errors:
                errors.append({"dataset": dataset, "seed": seed, "error": error})
            for record in records:
                all_records[dataset].append({**record, "_seed": seed})
            if reference_root is None:
                comparison_counts["not_compared"] += 1
            else:
                reference = reference_root / "optimization_trajectories" / dataset / f"seed_{seed}.pt"
                if not reference.is_file():
                    comparison_counts["different"] += 1
                    mismatches.append({"dataset": dataset, "seed": seed, "field": "reference_missing"})
                else:
                    reference_records = load_trajectory(reference).get("trajectory", [])
                    state, differences = compare_records(records, reference_records)
                    comparison_counts[state] += 1
                    for difference in differences[:10]:
                        mismatches.append({"dataset": dataset, "seed": seed, **difference})
    trajectory_files = len(trajectory_hashes)
    query_records = sum(len(value) for value in all_records.values())
    audit_status = "valid" if trajectory_files == 40 and query_records == 1600 and not errors else "invalid"
    if comparison_counts["different"]:
        comparison_status = "different"
    elif comparison_counts["not_compared"]:
        comparison_status = "not_compared"
    elif comparison_counts["numerically_equivalent"]:
        comparison_status = "numerically_equivalent"
    else:
        comparison_status = "identical"
    metrics: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        dataset_summary = run_root / dataset / f"{dataset}_summary_metrics.csv"
        if dataset_summary.is_file():
            metrics[dataset] = load_summary(dataset_summary)
        else:
            metrics[dataset] = load_summary(run_root / "summary_metrics.csv", dataset)
    recommendations: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        recommendations[dataset] = top_conditions(all_records[dataset])
        for item in recommendations[dataset]:
            item["dft_context"] = dft_context(project_root, dataset, item["condition"])
            item["feasibility"] = {
                **{
                    dimension: empty_feasibility_dimension()
                    for dimension in FEASIBILITY_DIMENSIONS
                },
                "assessment_status": "awaiting_llm_assessment",
            }
    review_request = feasibility_review_request(recommendations)
    report = {
        "schema_version": "2.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"datasets": list(DATASETS), "seeds": list(SEEDS), "rounds": ROUNDS, "batch_size": 1},
        "trajectory_audit": {
            "status": audit_status,
            "trajectory_files": trajectory_files,
            "query_records": query_records,
            "expected_trajectory_files": 40,
            "expected_query_records": 1600,
            "errors": errors[:100],
        },
        "reference_comparison": {
            "status": comparison_status,
            "counts": comparison_counts,
            "mismatches": mismatches[:100],
        },
        "metrics": metrics,
        "recommendations": recommendations,
        "feasibility_review_request": review_request,
        "environment": environment_versions(),
        "provenance": {
            "production_config_sha256": sha256_file(project_root / "code" / "config" / "production_config.json"),
            "trajectory_sha256": trajectory_hashes,
            "input_data_root": str(input_data_root),
            "candidate_table_sha256": candidate_table_hashes,
            "optimizer_entrypoint": "code/main/run_selected.py",
            "llm_effect_on_ranking": "none",
            "dft_effect_on_ranking": "none",
        },
        "generalization_claim": "official datasets only; new datasets require a separate OOD audit",
        "safety_note": "最终实验安全决策必须遵循实验室 SOP 并经 EHS 人工审核。",
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (report_root / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if audit_status != "valid":
        raise SystemExit("fresh trajectory audit failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--data-root",
        help="Actual competition input root containing Suzuki/ and Buchwald_sub4/; defaults to PROJECT_ROOT/data for backward compatibility",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--reference-root")
    args = parser.parse_args()
    report = build(
        Path(args.project_root).resolve(),
        Path(args.run_root).resolve(),
        Path(args.report_root).resolve(),
        Path(args.reference_root).resolve() if args.reference_root else None,
        Path(args.data_root).resolve() if args.data_root else None,
    )
    print(json.dumps({
        "status": report["trajectory_audit"]["status"],
        "trajectory_files": report["trajectory_audit"]["trajectory_files"],
        "query_records": report["trajectory_audit"]["query_records"],
        "comparison": report["reference_comparison"]["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
