#!/usr/bin/env python3
"""Attach a post-selection DFT explanation to one Agent recommendation.

The input may be either a direct condition mapping or the JSON object emitted
by ``reaction_agent.py recommend``.  This command never reads an outcome table
and never changes the optimizer state or candidate ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dft_explainer import DFTMechanisticExplainer  # noqa: E402


MODEL_FIELDS = (
    "candidate_index",
    "round",
    "predicted_yield",
    "uncertainty",
    "acquisition_value",
    "selection_mode",
    "method",
)


def _condition(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("condition")
    if isinstance(nested, Mapping):
        return nested
    return payload


def build_output(
    payload: Mapping[str, Any],
    dataset: str,
    feature_table: str | Path,
) -> dict[str, Any]:
    explainer = DFTMechanisticExplainer(dataset, feature_table)
    decision = {field: payload[field] for field in MODEL_FIELDS if field in payload}
    return {
        "analysis_phase": "post_selection_pre_feedback",
        "ranking_effect": "none",
        "model_decision": decision,
        "dft_mechanistic_explanation": explainer.explain(_condition(payload)),
        "boundary_statement": (
            "The optimizer selected this candidate before the DFT explanation was "
            "generated. This output is read-only and is not written back to the "
            "surrogate, acquisition function, optimizer state or candidate order."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Suzuki", "Buchwald_sub4"), required=True)
    parser.add_argument("--recommendation", required=True, help="Agent recommendation JSON")
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recommendation = Path(args.recommendation)
    payload = json.loads(recommendation.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Recommendation JSON must contain an object")
    result = build_output(payload, args.dataset, args.feature_table)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
