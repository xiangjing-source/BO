#!/usr/bin/env python3
"""Run an offline end-to-end health test of both reaction-agent routes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def run(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != expected:
        raise RuntimeError(
            f"command returned {result.returncode}, expected {expected}: "
            f"{' '.join(arguments)}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def main() -> None:
    scripts = Path(__file__).resolve().parent
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="coupling-agent-self-test-") as directory:
        root = Path(directory)

        structures = root / "structures.csv"
        write_csv(
            structures,
            ["variable", "value", "identifier_type", "identifier"],
            [
                ["ArylHalide", "bromobenzene", "smiles", "Brc1ccccc1"],
                ["ArylHalide", "chlorobenzene", "smiles", "Clc1ccccc1"],
                ["Amine", "aniline", "smiles", "Nc1ccccc1"],
                [
                    "Ligand",
                    "triphenylphosphine",
                    "smiles",
                    "P(c1ccccc1)(c1ccccc1)c1ccccc1",
                ],
                ["Ligand", "triethylphosphine", "smiles", "P(CC)(CC)CC"],
            ],
        )
        mapping = root / "structure_map.json"
        run(
            str(scripts / "resolve_structures.py"),
            "--input",
            str(structures),
            "--output",
            str(mapping),
            "--cache",
            str(root / "structure_cache.json"),
            "--offline",
        )
        checks.append("offline_authoritative_structure_map")

        molecular = ["ArylHalide", "Amine", "Ligand"]
        buchwald_train = root / "buchwald_train.csv"
        buchwald_candidates = root / "buchwald_test_features.csv"
        write_csv(
            buchwald_train,
            [*molecular, "Yield"],
            [
                ["bromobenzene", "aniline", "triphenylphosphine", 55.0],
                ["chlorobenzene", "aniline", "triphenylphosphine", 32.0],
            ],
        )
        write_csv(
            buchwald_candidates,
            molecular,
            [
                ["bromobenzene", "aniline", "triethylphosphine"],
                ["chlorobenzene", "aniline", "triethylphosphine"],
                ["bromobenzene", "aniline", "triphenylphosphine"],
            ],
        )
        state = root / "buchwald_state.json"
        run(
            str(scripts / "reaction_agent.py"),
            "init",
            "--reaction",
            "buchwald",
            "--train",
            str(buchwald_train),
            "--candidates",
            str(buchwald_candidates),
            "--mapping",
            str(mapping),
            "--molecular-columns",
            ",".join(molecular),
            "--state",
            str(state),
            "--rounds",
            "2",
            "--seed",
            "100",
            "--memory-budget-mb",
            "512",
        )
        first = json.loads(
            run(
                str(scripts / "reaction_agent.py"),
                "recommend",
                "--state",
                str(state),
            ).stdout
        )
        if first["round"] != 1:
            raise RuntimeError("first Buchwald recommendation has the wrong round")
        run(
            str(scripts / "reaction_agent.py"),
            "recommend",
            "--state",
            str(state),
            expected=1,
        )
        checks.append("single_pending_transaction_guard")
        run(
            str(scripts / "reaction_agent.py"),
            "feedback",
            "--state",
            str(state),
            "--yield",
            "61.0",
        )
        run(
            str(scripts / "reaction_agent.py"),
            "recommend",
            "--state",
            str(state),
        )
        run(
            str(scripts / "reaction_agent.py"),
            "feedback",
            "--state",
            str(state),
            "--yield",
            "73.0",
        )
        status = json.loads(
            run(
                str(scripts / "reaction_agent.py"),
                "status",
                "--state",
                str(state),
            ).stdout
        )
        if status["completed_rounds"] != 2 or status["pending"] is not None:
            raise RuntimeError("Buchwald state did not commit two feedback rounds")
        run(
            str(scripts / "validate_trajectory.py"),
            "--trajectory",
            str(root / "buchwald_state.trajectory.csv"),
            "--expected-rounds",
            "2",
        )
        checks.extend(
            [
                "buchwald_structure_random_forest",
                "atomic_feedback_checkpoint",
                "trajectory_integrity",
            ]
        )

        leaked = root / "leaked_candidates.csv"
        write_csv(
            leaked,
            [*molecular, "Yield"],
            [["bromobenzene", "aniline", "triethylphosphine", 99.0]],
        )
        run(
            str(scripts / "preflight_reaction_data.py"),
            "--train",
            str(buchwald_train),
            "--candidates",
            str(leaked),
            "--variables",
            ",".join(molecular),
            expected=2,
        )
        checks.append("candidate_outcome_rejection")

        suzuki_train = root / "suzuki_train.csv"
        suzuki_candidates = root / "suzuki_candidates.csv"
        suzuki_columns = ["Case", "Catalyst", "Temperature", "ResidenceTime"]
        write_csv(
            suzuki_train,
            [*suzuki_columns, "Yield"],
            [
                ["Case1", "CatA", 80, 5, 40],
                ["Case1", "CatB", 100, 10, 75],
                ["Case2", "CatA", 90, 8, 62],
                ["Case2", "CatB", 110, 12, 88],
            ],
        )
        write_csv(
            suzuki_candidates,
            suzuki_columns,
            [
                ["Case1", "CatA", 95, 9],
                ["Case1", "CatB", 105, 11],
                ["Case2", "CatA", 115, 13],
            ],
        )
        suzuki_state = root / "suzuki_state.json"
        run(
            str(scripts / "reaction_agent.py"),
            "init",
            "--reaction",
            "suzuki",
            "--train",
            str(suzuki_train),
            "--candidates",
            str(suzuki_candidates),
            "--categorical-columns",
            "Case,Catalyst",
            "--continuous-columns",
            "Temperature,ResidenceTime",
            "--state",
            str(suzuki_state),
            "--rounds",
            "1",
        )
        run(
            str(scripts / "reaction_agent.py"),
            "recommend",
            "--state",
            str(suzuki_state),
        )
        run(
            str(scripts / "reaction_agent.py"),
            "feedback",
            "--state",
            str(suzuki_state),
            "--yield",
            "91.0",
        )
        checks.append("suzuki_hybrid_gp_forest")

        provider = json.loads(
            run(
                str(scripts / "reaction_agent.py"),
                "provider-check",
                "--base-url",
                "https://fxb.supa.net.cn:6443/v1",
                "--model",
                "minimax-2.7",
            ).stdout
        )
        if not provider["valid"] or provider["api_key_checked"]:
            raise RuntimeError("provider check violated the non-secret contract")
        checks.append("non_secret_provider_configuration")

    print(json.dumps({"passed": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
