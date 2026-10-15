#!/usr/bin/env python3
"""Fail-closed public-boundary and schema preflight for reaction optimization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def _rdkit_parseable(smiles: str) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        return False
    return Chem.MolFromSmiles(smiles) is not None


OUTCOMES = {
    "yield", "yld", "产率", "label", "target", "outcome",
    "response", "conversion",
    "yieldpercent", "yieldpercentage", "yieldpct", "reactionyield",
    "observedyield", "candidateyield", "trueyield", "measuredyield",
    "experimentalyield", "outputyield",
}


def normalized(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def inspect(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_preflight(
    *,
    train_path: Path,
    candidates_path: Path,
    variables: list[str],
    yield_column: str,
    structure_map: Path | None,
    structure_variables: list[str] | None = None,
    continuous_variables: list[str] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    name = candidates_path.name.casefold()
    if name == "test.csv" or "searchspace" in name:
        blockers.append("candidate path looks like an answer/search-space table")
    train_headers, train_data = inspect(train_path)
    candidate_headers, candidate_data = inspect(candidates_path)
    answer_tokens = (
        "test", "answer", "oracle", "label", "result",
        "ground_truth", "groundtruth", "solution",
    )
    if yield_column in train_headers and any(
        token in train_path.name.casefold() for token in answer_tokens
    ):
        blockers.append("training path looks like a possible answer/oracle table")
    train_rows = len(train_data)
    candidate_rows = len(candidate_data)
    if yield_column not in train_headers:
        blockers.append(f"training table lacks {yield_column}")
    else:
        invalid_yields = 0
        for row in train_data:
            try:
                value = float(row.get(yield_column, ""))
                if not math.isfinite(value) or value < 0 or value > 100:
                    invalid_yields += 1
            except (TypeError, ValueError):
                invalid_yields += 1
        if invalid_yields:
            blockers.append(
                f"training table has {invalid_yields} nonnumeric, nonfinite, or out-of-range yields"
            )
    declared_outcome = normalized(yield_column)
    if yield_column in variables or declared_outcome in {normalized(item) for item in variables}:
        blockers.append("yield column cannot also be a model input variable")
    leaked = [
        column for column in candidate_headers
        if normalized(column) in OUTCOMES or normalized(column) == declared_outcome
    ]
    if leaked:
        blockers.append(f"candidate table contains outcome-like columns: {leaked}")
    for label, headers in (("training", train_headers), ("candidate", candidate_headers)):
        missing = sorted(set(variables) - set(headers))
        if missing:
            blockers.append(f"{label} table lacks variables: {missing}")
    undeclared_train = sorted(set(train_headers) - set(variables) - {yield_column})
    undeclared_candidates = sorted(set(candidate_headers) - set(variables))
    if undeclared_train:
        blockers.append(f"training table has undeclared columns: {undeclared_train}")
    if undeclared_candidates:
        blockers.append(f"candidate table has undeclared columns: {undeclared_candidates}")
    if train_rows < 2:
        blockers.append("training table must contain at least two rows")
    if candidate_rows < 1:
        blockers.append("candidate table is empty")

    if variables and set(variables).issubset(candidate_headers):
        continuous_set = set(continuous_variables or [])
        candidate_keys: list[tuple[Any, ...]] = []
        incomplete = 0
        for row in candidate_data:
            key: list[Any] = []
            row_incomplete = False
            for column in variables:
                raw_value = str(row.get(column, "")).strip()
                if not raw_value:
                    row_incomplete = True
                if column in continuous_set:
                    try:
                        number = float(raw_value)
                        value: Any = ("continuous", number) if math.isfinite(number) else ("invalid", raw_value)
                    except (TypeError, ValueError):
                        value = ("invalid", raw_value)
                else:
                    value = ("categorical", raw_value)
                key.append(value)
            candidate_keys.append(tuple(key))
            incomplete += int(row_incomplete)
        duplicates = len(candidate_keys) - len(set(candidate_keys))
        if incomplete:
            blockers.append(f"candidate table has {incomplete} incomplete conditions")
        if duplicates:
            blockers.append(f"candidate table has {duplicates} duplicate conditions")

    if structure_map:
        payload = json.loads(structure_map.read_text(encoding="utf-8"))
        resolved = {
            (
                str(item.get("variable", item.get("role", ""))),
                str(item.get("value", "")),
            )
            for item in payload.get("records", [])
            if str(item.get("review_status", "")).lower()
            in {"approved", "approved_before_agent_run"}
            and str(item.get("canonical_smiles", "")).strip()
            and _rdkit_parseable(str(item.get("canonical_smiles", "")).strip())
            and (
                str(item.get("source", item.get("source_url", ""))).strip()
                or str(item.get("status", "")) == "resolved_direct_smiles"
            )
        }
        required = {
            (column, str(row.get(column, "")).strip())
            for column in (structure_variables or variables)
            for row in [*train_data, *candidate_data]
        }
        missing = sorted(required - resolved)
        if missing:
            blockers.append(f"structure map lacks {len(missing)} required records")

    return {
        "status": "blocked" if blockers else "ready",
        "train_rows": train_rows,
        "candidate_rows": candidate_rows,
        "variables": variables,
        "candidate_outcomes_present": bool(leaked),
        "warnings": warnings,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--variables", required=True)
    parser.add_argument("--yield-column", default="Yield")
    parser.add_argument("--structure-map")
    parser.add_argument("--structure-variables")
    parser.add_argument("--continuous-variables")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = run_preflight(
        train_path=Path(args.train),
        candidates_path=Path(args.candidates),
        variables=[item.strip() for item in args.variables.split(",") if item.strip()],
        yield_column=args.yield_column,
        structure_map=Path(args.structure_map) if args.structure_map else None,
        structure_variables=[
            item.strip()
            for item in (args.structure_variables or "").split(",")
            if item.strip()
        ] or None,
        continuous_variables=[
            item.strip()
            for item in (args.continuous_variables or "").split(",")
            if item.strip()
        ] or None,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        atomic_text(Path(args.report), text + "\n")
    print(text)
    if result["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
