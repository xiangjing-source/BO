#!/usr/bin/env python3
"""Validate round continuity, unique queries, and running-best consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED = {
    "seed",
    "method",
    "round",
    "candidate_index",
    "observed_yield",
    "best_so_far",
    "selection_mode",
    "transaction_id",
}


def validate(path: Path, expected_rounds: int | None) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        rows = list(reader)
    findings: list[dict[str, object]] = []
    missing = sorted(REQUIRED - headers)
    if missing:
        findings.append({"code": "missing_columns", "columns": missing})
        return {"passed": False, "rows": len(rows), "groups": 0, "findings": findings}
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["seed"]), []).append(row)
    for (method, seed), group in groups.items():
        group.sort(key=lambda row: int(row["round"]))
        rounds = [int(row["round"]) for row in group]
        indices = [int(row["candidate_index"]) for row in group]
        transactions = [row["transaction_id"] for row in group]
        if rounds != list(range(1, len(group) + 1)):
            findings.append(
                {"code": "non_contiguous_rounds", "method": method, "seed": seed}
            )
        if len(indices) != len(set(indices)):
            findings.append(
                {"code": "duplicate_candidate", "method": method, "seed": seed}
            )
        if len(transactions) != len(set(transactions)):
            findings.append(
                {"code": "duplicate_transaction", "method": method, "seed": seed}
            )
        if expected_rounds is not None and len(group) != expected_rounds:
            findings.append(
                {
                    "code": "unexpected_round_count",
                    "method": method,
                    "seed": seed,
                    "actual": len(group),
                    "expected": expected_rounds,
                }
            )
        running = -math.inf
        for row in group:
            observed = float(row["observed_yield"])
            recorded = float(row["best_so_far"])
            if not math.isfinite(observed) or not math.isfinite(recorded):
                findings.append(
                    {
                        "code": "non_finite_value",
                        "method": method,
                        "seed": seed,
                        "round": row["round"],
                    }
                )
                continue
            running = max(running, observed)
            if not math.isclose(running, recorded, rel_tol=1e-9, abs_tol=1e-8):
                findings.append(
                    {
                        "code": "best_so_far_mismatch",
                        "method": method,
                        "seed": seed,
                        "round": row["round"],
                    }
                )
    return {
        "passed": not findings,
        "rows": len(rows),
        "groups": len(groups),
        "findings": findings,
        "scope": (
            "This validates the recorded trajectory but cannot prove that an "
            "answer table was not opened by another process."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--expected-rounds", type=int)
    args = parser.parse_args()
    result = validate(Path(args.trajectory), args.expected_rounds)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
