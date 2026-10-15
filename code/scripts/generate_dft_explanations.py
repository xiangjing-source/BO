#!/usr/bin/env python3
"""Generate post-selection DFT explanations from frozen trajectory records."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from dft_explainer import DFTMechanisticExplainer, ROLE_LABELS


DATASETS = ("Suzuki", "Buchwald_sub4")
SEEDS = tuple(range(100, 2001, 100))


def load_records(results_root: Path, dataset: str, seed: int) -> list[dict[str, Any]]:
    roots = (
        results_root / "optimization_trajectories" / dataset,
        results_root / dataset / "optimization_trajectories" / dataset,
    )
    for root in roots:
        for name in (f"seed_{seed}.pt", f"seed_{seed:04d}.pt"):
            path = root / name
            if path.is_file():
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                records = payload["trajectory"]
                if len(records) != 40:
                    raise ValueError(f"Expected 40 rounds: {path}")
                return records
    raise FileNotFoundError(f"Trajectory not found for {dataset}, seed {seed}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.ones(count, dtype=float)
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        candidate = float(p_values[original_index]) * count / rank
        running = min(running, candidate)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def component_spearman(
    values: list[float],
    component_mean_yields: list[float],
) -> tuple[float, float, str]:
    rho, asymptotic_p = spearmanr(values, component_mean_yields)
    count = len(values)
    if count > 8:
        return float(rho), float(asymptotic_p), "asymptotic"
    x_rank = rankdata(values)
    y_rank = rankdata(component_mean_yields)
    observed = abs(float(np.corrcoef(x_rank, y_rank)[0, 1]))
    extreme = 0
    total = 0
    for permutation in itertools.permutations(y_rank.tolist()):
        candidate = abs(float(np.corrcoef(x_rank, permutation)[0, 1]))
        extreme += int(candidate >= observed - 1e-12)
        total += 1
    return float(rho), extreme / total, "exact_permutation"


def evidence_line(item: dict[str, Any]) -> str:
    return (
        f"{item['role_label']}（{item['component']}）：{item['text']}；"
        f"同角色第 {item['percentile']:.0f} 百分位，参照 n={item['reference_n']}。"
    )


def model_block(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(record["step"]),
        "query_index": int(record["query_index"]),
        "predicted_yield": float(record["predicted_yield"]),
        "uncertainty": float(record["uncertainty"]),
        "acquisition_value": float(record["acquisition_value"]),
        "selection_mode": str(record["selection_mode"]),
        "observed_yield_post_query": float(record["observed_yield"]),
        "best_so_far_post_query": float(record["best_so_far"]),
    }


def choose_examples(
    records: list[dict[str, Any]],
    explainer: DFTMechanisticExplainer,
) -> dict[str, Any]:
    first = records[0]
    best = max(records, key=lambda row: float(row["observed_yield"]))
    low = min(records, key=lambda row: float(row["observed_yield"]))
    best_condition = best["condition"]

    def distance(row: dict[str, Any]) -> tuple[int, int]:
        changes = sum(
            str(row["condition"][role]) != str(best_condition[role])
            for role in explainer.roles
        )
        return changes, int(row["query_index"])

    counterfactual = min(
        (record for record in records if record is not best),
        key=distance,
    )
    examples = {}
    for label, record in (
        ("first_recommendation", first),
        ("trajectory_best", best),
        ("representative_low_observation", low),
        ("nearest_observed_counterfactual", counterfactual),
    ):
        examples[label] = {
            "model_record": model_block(record),
            "condition": record["condition"],
            "dft_explanation": explainer.explain(record["condition"]),
        }
    examples["first_vs_trajectory_best"] = explainer.compare(
        first["condition"], best["condition"]
    )
    examples["best_vs_nearest_observed_counterfactual"] = explainer.compare(
        best["condition"], counterfactual["condition"]
    )
    return examples


def unique_queried_records(
    results_root: Path,
    dataset: str,
) -> dict[int, dict[str, Any]]:
    unique: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        for record in load_records(results_root, dataset, seed):
            index = int(record["query_index"])
            if index in unique:
                previous = unique[index]
                if (
                    previous["condition"] != record["condition"]
                    or not math.isclose(
                        float(previous["observed_yield"]),
                        float(record["observed_yield"]),
                        rel_tol=0,
                        abs_tol=1e-9,
                    )
                ):
                    raise ValueError(f"Inconsistent repeated query index {dataset}:{index}")
            else:
                unique[index] = record
    return unique


def coverage_rows(
    dataset: str,
    records: dict[int, dict[str, Any]],
    explainer: DFTMechanisticExplainer,
) -> list[dict[str, Any]]:
    counts = {role: 0 for role in explainer.roles}
    complete = 0
    for record in records.values():
        explanation = explainer.explain(record["condition"])
        present = {
            item["role"]: bool(item["available"])
            for item in explanation["components"]
        }
        for role, available in present.items():
            counts[role] += int(available)
        complete += int(all(present.values()))
    total = len(records)
    rows = [{
        "dataset": dataset,
        "role": "complete_condition",
        "unique_queried_conditions": total,
        "covered_conditions": complete,
        "coverage_fraction": complete / total if total else 0.0,
    }]
    rows.extend({
        "dataset": dataset,
        "role": role,
        "unique_queried_conditions": total,
        "covered_conditions": counts[role],
        "coverage_fraction": counts[role] / total if total else 0.0,
    } for role in explainer.roles)
    return rows


def association_rows(
    dataset: str,
    records: dict[int, dict[str, Any]],
    explainer: DFTMechanisticExplainer,
) -> list[dict[str, Any]]:
    rows = []
    for role, field in explainer.registered_features:
        by_component: dict[str, dict[str, Any]] = {}
        queried_count = 0
        for record in records.values():
            name = str(record["condition"][role])
            statistic = explainer.percentile(role, name, field)
            if statistic is None:
                continue
            queried_count += 1
            item = by_component.setdefault(
                name,
                {"percentile": float(statistic["percentile"]), "yields": []},
            )
            item["yields"].append(float(record["observed_yield"]))
        if len(by_component) < 3:
            continue
        values = [item["percentile"] for item in by_component.values()]
        component_mean_yields = [
            float(np.mean(item["yields"])) for item in by_component.values()
        ]
        if len(set(values)) < 2 or len(set(component_mean_yields)) < 2:
            continue
        rho, p_value, p_method = component_spearman(values, component_mean_yields)
        if not math.isfinite(float(rho)) or not math.isfinite(float(p_value)):
            continue
        rows.append({
            "dataset": dataset,
            "role": role,
            "role_label": ROLE_LABELS[role],
            "feature": field,
            "n_unique_queried_conditions": queried_count,
            "n_unique_components": len(by_component),
            "spearman_rho": float(rho),
            "p_value_unadjusted": float(p_value),
            "p_value_method": p_method,
        })
    adjusted = bh_adjust([row["p_value_unadjusted"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["q_value_bh"] = q_value
        row["interpretation_scope"] = (
            "post_hoc_descriptive_only_not_used_for_ranking"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    examples: dict[str, Any],
    coverage: list[dict[str, Any]],
    associations: list[dict[str, Any]],
) -> str:
    lines = [
        "# 已完成轨迹的 DFT 机理解释",
        "",
        "> 本报告在候选已经由最终模型选定后生成。DFT 不参与代理模型拟合、",
        "> 期望改进计算或候选排序；全部产率关联仅属于轨迹完成后的探索性分析。",
        "",
    ]
    for dataset in DATASETS:
        dataset_examples = examples[dataset]
        first = dataset_examples["first_recommendation"]
        best = dataset_examples["trajectory_best"]
        explanation = first["dft_explanation"]
        lines.extend([
            f"## {dataset}",
            "",
            "### seed 100 首轮推荐",
            "",
            f"- 候选索引：{first['model_record']['query_index']}；",
            f"- 模型预测：{first['model_record']['predicted_yield']:.2f}；",
            f"- 模型不确定性：{first['model_record']['uncertainty']:.2f}；",
            f"- 期望改进值：{first['model_record']['acquisition_value']:.3f}；",
            f"- 查询后观测产率：{first['model_record']['observed_yield_post_query']:.2f}%；",
            f"- DFT 覆盖：{explanation['coverage']['available_roles']}/"
            f"{explanation['coverage']['total_roles']} 个角色；",
            f"- 证据一致性：{explanation['evidence_consistency']}；"
            f"解释可信度：{explanation['confidence']}。",
            "",
            "**DFT 支持证据**",
            "",
        ])
        supports = explanation["support_evidence"]
        lines.extend(
            [f"- {evidence_line(item)}" for item in supports]
            if supports else
            ["- 没有预注册指标落入同角色分布的支持极端区间。"]
        )
        lines.extend(["", "**DFT 风险提示**", ""])
        risks = explanation["risk_flags"]
        lines.extend(
            [f"- {evidence_line(item)}" for item in risks]
            if risks else
            ["- 没有预注册指标落入同角色分布的风险极端区间。"]
        )
        changed = dataset_examples["first_vs_trajectory_best"]["changed_role_labels"]
        lines.extend([
            "",
            "### 与 seed 100 轨迹最优条件对比",
            "",
            f"- 轨迹最优观测产率：{best['model_record']['observed_yield_post_query']:.2f}%；",
            f"- 首轮推荐与轨迹最优条件变化角色：{'、'.join(changed) if changed else '无'}；",
            "- 该对比是轨迹完成后的反事实说明，不是模型在首轮可见的信息。",
            "",
        ])

    lines.extend([
        "## 覆盖与事后关联说明",
        "",
    ])
    for dataset in DATASETS:
        row = next(
            item for item in coverage
            if item["dataset"] == dataset and item["role"] == "complete_condition"
        )
        lines.append(
            f"- {dataset}：{row['covered_conditions']}/"
            f"{row['unique_queried_conditions']} 个唯一已查询条件具有全角色 DFT 覆盖"
            f"（{100 * row['coverage_fraction']:.1f}%）。"
        )
    lines.extend([
        "",
        "`posthoc_associations.csv` 先按同角色组分汇总其已查询条件的平均产率，再对",
        "所有预注册角色—特征组合统一报告组分级 Spearman 相关、未校正 p 值和",
        "Benjamini–Hochberg q 值。组分数很少且候选来自主动采样，这些相关性不能",
        "解释为因果关系，也没有反馈给优化器。",
        "",
    ])
    discoveries = sum(float(row["q_value_bh"]) < 0.05 for row in associations)
    lines.append(
        f"在 Benjamini–Hochberg 校正后，q < 0.05 的预注册关联共有 {discoveries} 个；"
        "该结果支持继续把 DFT 作为解释层，而不是据此改写固定排序。"
    )
    lines.append("")
    for dataset in DATASETS:
        subset = [row for row in associations if row["dataset"] == dataset]
        strongest = sorted(subset, key=lambda row: abs(row["spearman_rho"]), reverse=True)[:3]
        lines.append(f"### {dataset} 描述性相关绝对值最大的三个预注册特征")
        lines.append("")
        for row in strongest:
            lines.append(
                f"- {row['role_label']} / `{row['feature']}`："
                f"ρ={row['spearman_rho']:.3f}，q={row['q_value_bh']:.3g}，"
                f"组分数 n={row['n_unique_components']}。"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    feature_table = Path(args.feature_table).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    examples: dict[str, Any] = {}
    coverage: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for dataset in DATASETS:
        explainer = DFTMechanisticExplainer(dataset, feature_table)
        seed100 = load_records(results_root, dataset, 100)
        examples[dataset] = choose_examples(seed100, explainer)
        unique = unique_queried_records(results_root, dataset)
        coverage.extend(coverage_rows(dataset, unique, explainer))
        associations.extend(association_rows(dataset, unique, explainer))

    payload = {
        "analysis_phase": "post_hoc_after_model_and_trajectories_were_frozen",
        "ranking_effect": "none",
        "feature_table": feature_table.name,
        "examples": examples,
    }
    (output / "dft_explanation_examples.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "dft_coverage_summary.csv", coverage)
    write_csv(output / "dft_posthoc_associations.csv", associations)
    (output / "dft_explanation_examples.md").write_text(
        render_markdown(examples, coverage, associations),
        encoding="utf-8",
    )
    print(f"Generated DFT explanations in {output}")


if __name__ == "__main__":
    main()
