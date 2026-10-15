#!/usr/bin/env python3
"""Choose structural capacity from public metadata only."""

from __future__ import annotations

import argparse
import json
from typing import Any


POLICY_VERSION = "coupling-fingerprint-v1"
BITS = (32, 64, 128)
DESCRIPTORS = 8


def initial_bits(labeled_rows: int) -> int:
    if labeled_rows < 64:
        return 32
    if labeled_rows < 256:
        return 64
    return 128


def choose_configuration(
    *,
    reaction: str,
    labeled_rows: int,
    unique_molecules: int,
    candidate_rows: int,
    molecular_roles: int,
    invalid_fraction: float,
    memory_budget_mb: int,
    enable_structure: bool = False,
) -> dict[str, Any]:
    if reaction not in {"buchwald", "suzuki"}:
        raise ValueError("reaction must be buchwald or suzuki")
    if labeled_rows < 2 or candidate_rows < 1:
        raise ValueError("at least two labels and one candidate are required")
    if molecular_roles < 1 or memory_budget_mb < 128:
        raise ValueError("at least one role and 128 MiB are required")
    if not 0 <= invalid_fraction <= 1:
        raise ValueError("invalid_fraction must be in 0..1")

    selected = initial_bits(labeled_rows)
    allowed_bytes = memory_budget_mb * 1024 * 1024 * 0.5

    def working_bytes(bits: int) -> int:
        dimensions = molecular_roles * (bits + DESCRIPTORS)
        return candidate_rows * dimensions * 4 * 4

    while selected > 32 and working_bytes(selected) > allowed_bytes:
        selected = BITS[BITS.index(selected) - 1]

    dimensions = molecular_roles * (selected + DESCRIPTORS)
    bytes_per_candidate = max(dimensions * 4 * 4, 1)
    batch_size = max(1, min(candidate_rows, int(allowed_bytes // bytes_per_candidate)))
    errors: list[str] = []
    warnings: list[str] = []
    if reaction == "buchwald" and invalid_fraction > 0:
        errors.append("Buchwald structure coverage must be 100%.")
    if selected < initial_bits(labeled_rows):
        warnings.append("Fingerprint capacity was reduced to respect the memory budget.")
    if batch_size < candidate_rows:
        warnings.append("Candidate scoring will use deterministic batches.")

    return {
        "status": "blocked" if errors else "ready",
        "policy_version": POLICY_VERSION,
        "public_inputs": {
            "labeled_rows": labeled_rows,
            "candidate_rows": candidate_rows,
            "unique_molecules": unique_molecules,
            "molecular_roles": molecular_roles,
            "invalid_structure_fraction": invalid_fraction,
            "memory_budget_mb": memory_budget_mb,
        },
        "structure_features": {
            "enabled": reaction == "buchwald" or enable_structure,
            "fingerprint_type": "Morgan",
            "fingerprint_radius": 2,
            "fingerprint_bits": selected,
            "physicochemical_features_per_component": DESCRIPTORS,
            "dtype": "float32",
            "candidate_batch_size": batch_size,
        },
        "surrogate": {
            "type": "random_forest" if reaction == "buchwald" else "hybrid_gp_forest",
            "random_forest_trees": 100,
            "minimum_leaf_size": 2,
            "processes": 1,
            "suzuki_forest_weight": 0.75,
            "gp_max_train_rows": 800,
            "gp_capacity_fallback": "random_forest",
        },
        "acquisition": {
            "type": "expected_improvement",
            "epsilon_start": 0.15,
            "epsilon_end": 0.01,
            "queries_per_round": 1,
        },
        "estimated_working_mb": round(working_bytes(selected) / 1024 / 1024, 4),
        "warnings": warnings,
        "required_actions": errors,
        "selection_note": "No candidate outcomes were accepted or inspected.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction", choices=("buchwald", "suzuki"), required=True)
    parser.add_argument("--labeled-rows", type=int, required=True)
    parser.add_argument("--candidate-rows", type=int, required=True)
    parser.add_argument(
        "--roles", "--molecular-roles", dest="roles", type=int, required=True
    )
    parser.add_argument("--unique-molecules", type=int, default=0)
    parser.add_argument("--invalid-fraction", type=float, default=0)
    parser.add_argument("--available-memory-mb", type=int, default=2048)
    args = parser.parse_args()
    result = choose_configuration(
        reaction=args.reaction,
        labeled_rows=args.labeled_rows,
        candidate_rows=args.candidate_rows,
        molecular_roles=args.roles,
        unique_molecules=args.unique_molecules,
        invalid_fraction=args.invalid_fraction,
        memory_budget_mb=args.available_memory_mb,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
