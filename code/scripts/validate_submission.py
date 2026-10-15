"""Validate saved trajectory files against public candidate tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from protocol import load_public_dataset, load_trajectory, validate_trajectory
from run_compliant import OFFICIAL_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate competition trajectory files")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-name", required=True, choices=["Suzuki", "Buchwald_sub4"])
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    public = load_public_dataset(args.dataset_dir, args.dataset_name)
    results_root = Path(args.results_dir)
    trajectory_dir = results_root / "optimization_trajectories" / args.dataset_name
    if not trajectory_dir.is_dir():
        # ``run_selected.py --dataset all`` groups outputs once by reaction family,
        # while the compact submission groups both families below one results root.
        grouped = (
            results_root
            / args.dataset_name
            / "optimization_trajectories"
            / args.dataset_name
        )
        if grouped.is_dir():
            trajectory_dir = grouped
    failures = []
    for seed in OFFICIAL_SEEDS:
        path = trajectory_dir / f"seed_{seed}.pt"
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        payload = load_trajectory(path)
        if payload.get("seed") != seed or payload.get("dataset") != args.dataset_name:
            failures.append(f"metadata mismatch in {path.name}")
            continue
        try:
            validate_trajectory(payload.get("trajectory", []), public, expected_rounds=40)
        except Exception as exc:  # report every invalid file in one run
            failures.append(f"{path.name}: {exc}")
    if failures:
        raise SystemExit("Trajectory validation failed:\n- " + "\n- ".join(failures))
    print(f"Validated 20 complete {args.dataset_name} trajectories in {trajectory_dir}")


if __name__ == "__main__":
    main()
