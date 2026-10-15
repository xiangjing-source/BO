"""Post-selection, label-blind mechanistic explanations from frozen DFT data.

The explainer never changes a surrogate prediction or acquisition score.  It
compares each component only with DFT calculations from the same dataset and
reaction role, then reports percentile-based supporting evidence, cautions,
coverage and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROLE_LABELS = {
    "Product": "目标产物",
    "Reactant2": "芳基卤代物",
    "Electrophile": "亲电试剂",
    "Nucleophile": "有机硼试剂",
    "Ligand": "配体",
    "Additive": "添加剂",
    "Base": "碱",
    "Solvent": "溶剂",
}

DATASET_ROLES = {
    "Buchwald_sub4": ("Product", "Reactant2", "Ligand", "Additive", "Base"),
    "Suzuki": ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"),
}


@dataclass(frozen=True)
class InterpretationRule:
    label: str
    low_kind: str
    low_text: str
    high_kind: str
    high_text: str


RULES: dict[str, dict[str, dict[str, InterpretationRule]]] = {
    "Buchwald_sub4": {
        "Reactant2": {
            "c_leaving_mayer_bond_order": InterpretationRule(
                "C–离去基团 Mayer 键级",
                "support",
                "C–离去基团键级处于较低水平，与相对较弱的待活化键一致，可作为氧化加成易达性的支持证据",
                "risk",
                "C–离去基团键级处于较高水平，提示该键的活化可能需要更有利的配体或反应环境",
            ),
            "electrophile_c_hirshfeld": InterpretationRule(
                "反应碳 Hirshfeld 电荷",
                "risk",
                "反应碳相对不缺电子，电子效应对氧化加成的支持较弱",
                "support",
                "反应碳相对更缺电子，与亲电位点特征一致，可作为氧化加成的辅助支持",
            ),
            "lumo_ev": InterpretationRule(
                "LUMO 能量",
                "support",
                "LUMO 能量处于较低水平，显示相对较强的受电子特征，可作为氧化加成电子因素的间接证据",
                "risk",
                "LUMO 能量处于较高水平，受电子特征相对较弱",
            ),
        },
        "Ligand": {
            "homo_ev": InterpretationRule(
                "HOMO 能量",
                "risk",
                "HOMO 能量处于较低水平，配体电子给予能力可能相对较弱",
                "support",
                "HOMO 能量处于较高水平，提示相对较强的电子给予能力，可能有利于部分氧化加成步骤",
            ),
            "phosphorus_hirshfeld_mean": InterpretationRule(
                "磷位点 Hirshfeld 电荷",
                "support",
                "磷位点电荷处于较低水平，与相对富电子的配位位点一致",
                "risk",
                "磷位点电荷处于较高水平，磷中心电子丰富程度相对较低",
            ),
        },
        "Additive": {
            "homo_ev": InterpretationRule(
                "HOMO 能量",
                "context",
                "添加剂 HOMO 能量较低，电子给予和竞争配位倾向相对有限",
                "risk",
                "添加剂 HOMO 能量较高，可能具有更强电子给予或竞争配位倾向",
            ),
            "additive_n_hirshfeld": InterpretationRule(
                "添加剂氮位点电荷",
                "risk",
                "氮位点相对富电子，存在竞争配位或改变钯中心环境的可能",
                "context",
                "氮位点电子密度相对较低，竞争配位风险在该指标上不突出",
            ),
            "additive_o_hirshfeld": InterpretationRule(
                "添加剂氧位点电荷",
                "risk",
                "氧位点相对富电子，可能竞争配位，也可能稳定带电中间体，净效应需实验判断",
                "context",
                "氧位点电子密度相对较低，强配位风险在该指标上不突出",
            ),
        },
        "Base": {
            "donor_hirshfeld_min": InterpretationRule(
                "最富电子供体位点电荷",
                "support",
                "存在相对富电子的局部供体位点，与较强去质子化/离子相互作用能力一致，但也需注意配位竞争",
                "risk",
                "局部供体位点电子密度相对较低，去质子化能力的电子结构支持较弱",
            ),
            "dipole_debye": InterpretationRule(
                "偶极矩",
                "context",
                "偶极矩较低，电荷分离特征相对有限",
                "support",
                "偶极矩较高，显示较明显的电荷分离，可为离子型去质子化环境提供间接支持",
            ),
        },
    },
    "Suzuki": {
        "Electrophile": {
            "c_leaving_mayer_bond_order": InterpretationRule(
                "C–离去基团 Mayer 键级",
                "support",
                "C–离去基团键级处于较低水平，与相对较弱的待活化键一致，可支持氧化加成倾向",
                "risk",
                "C–离去基团键级处于较高水平，可能增加氧化加成活化要求",
            ),
            "electrophile_c_hirshfeld": InterpretationRule(
                "反应碳 Hirshfeld 电荷",
                "risk",
                "反应碳相对不缺电子，氧化加成的电子效应支持较弱",
                "support",
                "反应碳相对更缺电子，与亲电位点特征一致",
            ),
            "lumo_ev": InterpretationRule(
                "LUMO 能量",
                "support",
                "LUMO 能量处于较低水平，受电子特征相对较强，可作为氧化加成电子因素的间接支持",
                "risk",
                "LUMO 能量处于较高水平，受电子特征相对较弱",
            ),
        },
        "Nucleophile": {
            "c_b_mayer_bond_order": InterpretationRule(
                "C–B Mayer 键级",
                "support",
                "C–B 键级处于较低水平，与转移基团相对易活化的方向一致",
                "risk",
                "C–B 键级处于较高水平，转金属化可能更依赖碱和溶剂协同",
            ),
            "boron_hirshfeld": InterpretationRule(
                "硼位点 Hirshfeld 电荷",
                "risk",
                "硼位点相对不缺电子，碱活化的电子驱动力支持较弱",
                "support",
                "硼位点相对更缺电子，可能更容易接受碱/杂原子作用而进入活化态",
            ),
            "transfer_c_hirshfeld": InterpretationRule(
                "转移碳 Hirshfeld 电荷",
                "context",
                "转移碳处于同角色较低电荷分位，提示 C–B 极化方向与其他硼试剂不同",
                "context",
                "转移碳处于同角色较高电荷分位，提示明显的 C–B 电荷极化",
            ),
        },
        "Ligand": {
            "homo_ev": InterpretationRule(
                "HOMO 能量",
                "risk",
                "HOMO 能量处于较低水平，配体电子给予能力可能相对较弱",
                "support",
                "HOMO 能量处于较高水平，提示相对较强的电子给予能力",
            ),
            "phosphorus_hirshfeld_mean": InterpretationRule(
                "磷位点 Hirshfeld 电荷",
                "support",
                "磷位点相对富电子，与较强供电子配位位点一致",
                "risk",
                "磷位点电子丰富程度相对较低",
            ),
        },
        "Base": {
            "donor_hirshfeld_min": InterpretationRule(
                "最富电子供体位点电荷",
                "support",
                "局部供体位点相对富电子，可作为硼物种活化能力的间接支持",
                "risk",
                "局部供体位点电子密度相对较低，碱活化的电子结构支持较弱",
            ),
            "dipole_debye": InterpretationRule(
                "偶极矩",
                "context",
                "偶极矩较低，电荷分离特征相对有限",
                "support",
                "偶极矩较高，显示较明显的电荷分离，可为硼物种活化提供间接背景证据",
            ),
        },
        "Solvent": {
            "dipole_debye": InterpretationRule(
                "偶极矩",
                "context",
                "偶极矩处于较低水平，对离子型中间体的稳定作用可能相对有限",
                "support",
                "偶极矩处于较高水平，可能更有利于稳定极性或离子型中间体",
            ),
            "donor_hirshfeld_min": InterpretationRule(
                "最富电子供体位点电荷",
                "risk",
                "溶剂供体位点相对富电子，可能增强溶剂化，也需警惕对金属中心的竞争配位",
                "context",
                "溶剂供体位点电子密度相对较低，强配位风险在该指标上不突出",
            ),
        },
    },
}


def _percentile(values: np.ndarray, value: float) -> float:
    less = float(np.sum(values < value))
    equal = float(np.sum(values == value))
    return 100.0 * (less + 0.5 * equal) / float(len(values))


class DFTMechanisticExplainer:
    """Generate outcome-independent DFT evidence after a candidate is chosen."""

    def __init__(self, dataset_name: str, feature_path: str | Path):
        if dataset_name not in DATASET_ROLES:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.roles = DATASET_ROLES[dataset_name]
        table = pd.read_csv(feature_path)
        forbidden = [column for column in table if "yield" in column.lower()]
        if forbidden:
            raise ValueError(f"DFT explanation table contains outcome columns: {forbidden}")
        required = {
            "dataset",
            "calculation_id",
            "original_dataset_column",
            "chemical_name",
        }
        missing = sorted(required - set(table))
        if missing:
            raise ValueError(f"DFT explanation table is missing columns: {missing}")
        table = table.loc[table["dataset"].astype(str).eq(dataset_name)].copy()
        if table.empty:
            raise ValueError(f"No DFT explanation rows for {dataset_name}")
        keys = ["original_dataset_column", "chemical_name"]
        if table.duplicated(keys, keep=False).any():
            raise ValueError("Ambiguous role/name keys in DFT explanation table")
        self.table = table
        self.lookup = {
            (str(row["original_dataset_column"]), str(row["chemical_name"])): row
            for _, row in table.iterrows()
        }

    def percentile(self, role: str, name: str, field: str) -> dict[str, float] | None:
        row = self.lookup.get((role, name))
        if row is None or field not in self.table or pd.isna(row[field]):
            return None
        reference = pd.to_numeric(
            self.table.loc[
                self.table["original_dataset_column"].astype(str).eq(role), field
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if not len(reference):
            return None
        value = float(row[field])
        median = float(np.median(reference))
        q25, q75 = np.quantile(reference, [0.25, 0.75])
        scale = float(q75 - q25)
        robust_z = (value - median) / (scale if scale > 1e-12 else 1.0)
        return {
            "raw_value": value,
            "percentile": _percentile(reference, value),
            "robust_z": float(robust_z),
            "reference_n": int(len(reference)),
        }

    def explain(self, condition: Mapping[str, object]) -> dict[str, object]:
        missing_roles = [role for role in self.roles if role not in condition]
        if missing_roles:
            raise ValueError(f"Condition lacks roles: {missing_roles}")
        components = []
        evidence: dict[str, list[dict[str, object]]] = {
            "support": [],
            "risk": [],
            "context": [],
        }
        available_roles = 0
        minimum_reference = None
        for role in self.roles:
            name = str(condition[role])
            available = (role, name) in self.lookup
            if available:
                available_roles += 1
            metrics = []
            for field, rule in RULES[self.dataset_name].get(role, {}).items():
                statistic = self.percentile(role, name, field)
                if statistic is None:
                    continue
                minimum_reference = (
                    statistic["reference_n"]
                    if minimum_reference is None
                    else min(minimum_reference, statistic["reference_n"])
                )
                item = {
                    "role": role,
                    "role_label": ROLE_LABELS[role],
                    "component": name,
                    "field": field,
                    "field_label": rule.label,
                    **statistic,
                }
                metrics.append(item)
                percentile = float(statistic["percentile"])
                if percentile <= 25.0:
                    interpreted = {**item, "kind": rule.low_kind, "text": rule.low_text}
                    evidence[rule.low_kind].append(interpreted)
                elif percentile >= 75.0:
                    interpreted = {**item, "kind": rule.high_kind, "text": rule.high_text}
                    evidence[rule.high_kind].append(interpreted)
            components.append({
                "role": role,
                "role_label": ROLE_LABELS[role],
                "component": name,
                "available": available,
                "metrics": metrics,
            })

        coverage = available_roles / len(self.roles)
        support_n = len(evidence["support"])
        risk_n = len(evidence["risk"])
        if coverage < 0.6:
            consistency = "证据不足"
        elif support_n and risk_n:
            consistency = "混合"
        elif support_n:
            consistency = "支持"
        elif risk_n:
            consistency = "存疑"
        else:
            consistency = "中性"
        if coverage == 1.0 and (minimum_reference or 0) >= 5:
            confidence = "高"
        elif coverage >= 0.8 and (minimum_reference or 0) >= 3:
            confidence = "中"
        else:
            confidence = "低"

        return {
            "dataset": self.dataset_name,
            "condition": {role: str(condition[role]) for role in self.roles},
            "coverage": {
                "available_roles": available_roles,
                "total_roles": len(self.roles),
                "fraction": coverage,
                "minimum_role_reference_n": minimum_reference or 0,
            },
            "components": components,
            "support_evidence": evidence["support"][:3],
            "risk_flags": evidence["risk"][:2],
            "context_notes": evidence["context"][:3],
            "evidence_consistency": consistency,
            "confidence": confidence,
            "disclaimer": (
                "DFT explanation is generated after selection from outcome-free electronic "
                "structure data. It does not affect surrogate fitting, acquisition values "
                "or candidate ranking."
            ),
        }

    def compare(
        self,
        candidate: Mapping[str, object],
        reference: Mapping[str, object],
    ) -> dict[str, object]:
        changed = [
            role for role in self.roles
            if str(candidate[role]) != str(reference[role])
        ]
        return {
            "changed_roles": changed,
            "changed_role_labels": [ROLE_LABELS[role] for role in changed],
            "candidate": self.explain(candidate),
            "reference": self.explain(reference),
        }

    @property
    def registered_features(self) -> Sequence[tuple[str, str]]:
        return tuple(
            (role, field)
            for role, fields in RULES[self.dataset_name].items()
            for field in fields
        )
