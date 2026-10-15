"""Leakage-safe data protocol for sequential reaction-condition optimization.

The optimizer receives only public inputs: the labelled training table and the
unlabelled candidate table.  The local evaluator owns the labelled test table
and releases one yield only after an index has been selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


DATASET_FILES: Mapping[str, Mapping[str, str]] = {
    "Suzuki": {
        "train": "suzuki_train.csv",
        "candidates": "suzuki_test_features.csv",
        "oracle": "suzuki_test.csv",
    },
    "Buchwald_sub4": {
        "train": "buchwald_sub4_train.csv",
        "candidates": "buchwald_sub4_test_features.csv",
        "oracle": "buchwald_sub4_test.csv",
    },
}

VARIABLE_COLUMNS: Mapping[str, List[str]] = {
    "Suzuki": ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"],
    "Buchwald_sub4": ["Reactant2", "Ligand", "Additive", "Base", "Product"],
}


@dataclass(frozen=True)
class PublicDataset:
    """The only dataset object allowed inside the optimizer."""

    name: str
    train: pd.DataFrame
    candidates: pd.DataFrame
    variable_columns: List[str]
    categories: Dict[str, List[str]]
    target_train: pd.DataFrame


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_public_dataset(dataset_dir: str | Path, dataset_name: str) -> PublicDataset:
    """Load train.csv and test_features.csv without opening any test labels."""
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    root = Path(dataset_dir)
    files = DATASET_FILES[dataset_name]
    train = pd.read_csv(root / files["train"])
    candidates = pd.read_csv(root / files["candidates"])
    columns = list(VARIABLE_COLUMNS[dataset_name])

    _require_columns(train, [*columns, "Yield"], "training table")
    _require_columns(candidates, columns, "candidate table")
    if "Yield" in candidates.columns:
        raise ValueError("Candidate table must not contain Yield; use test_features.csv")
    if candidates.duplicated(columns).any():
        raise ValueError("Candidate table contains duplicate condition combinations")

    categories: Dict[str, List[str]] = {}
    for column in columns:
        values = pd.concat([train[column], candidates[column]], ignore_index=True)
        categories[column] = sorted(values.astype(str).unique().tolist())

    if dataset_name == "Buchwald_sub4":
        products = candidates["Product"].astype(str).unique()
        if len(products) != 1:
            raise ValueError("Buchwald candidate pool must represent exactly one target product")
        target_train = train.loc[train["Product"].astype(str) == products[0]].copy()
        if target_train.empty:
            raise ValueError("No target-product labels found in Buchwald training table")
    else:
        target_train = train.copy()

    return PublicDataset(
        name=dataset_name,
        train=train.copy(),
        candidates=candidates.copy(),
        variable_columns=columns,
        categories=categories,
        target_train=target_train,
    )


class OfflineOracle:
    """Local-only oracle that releases the label for one selected candidate."""

    def __init__(self, dataset_dir: str | Path, public: PublicDataset):
        root = Path(dataset_dir)
        oracle = pd.read_csv(root / DATASET_FILES[public.name]["oracle"])
        _require_columns(oracle, [*public.variable_columns, "Yield"], "oracle table")
        if len(oracle) != len(public.candidates):
            raise ValueError("Oracle and candidate tables have different numbers of rows")

        expected = public.candidates[public.variable_columns].astype(str).reset_index(drop=True)
        received = oracle[public.variable_columns].astype(str).reset_index(drop=True)
        if not expected.equals(received):
            raise ValueError("Oracle row order/conditions do not match the public candidate table")

        self._oracle = oracle.reset_index(drop=True)
        self._public = public
        self._queried: set[int] = set()

    def query(self, query_index: int) -> float:
        """Return one true yield after validating a non-repeated public index."""
        if not 0 <= int(query_index) < len(self._oracle):
            raise IndexError(f"Invalid candidate index: {query_index}")
        if int(query_index) in self._queried:
            raise ValueError(f"Candidate index {query_index} was queried twice")
        self._queried.add(int(query_index))
        return float(self._oracle.iloc[int(query_index)]["Yield"])

    def final_global_best(self) -> float:
        """Evaluator-only value. Call only after the trajectory is complete."""
        return float(self._oracle["Yield"].max())


def condition_dict(row: pd.Series, variable_columns: Sequence[str]) -> Dict[str, str]:
    return {column: str(row[column]) for column in variable_columns}


def validate_trajectory(
    records: Sequence[Mapping[str, Any]],
    public: PublicDataset,
    expected_rounds: int = 40,
) -> None:
    """Validate the required competition trajectory fields before saving."""
    if len(records) != expected_rounds:
        raise ValueError(f"Expected {expected_rounds} trajectory steps, got {len(records)}")
    queried: set[int] = set()
    required = {"step", "query_index", "condition", "observed_yield", "best_so_far"}
    for expected_step, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Trajectory step {expected_step} missing {sorted(missing)}")
        if int(record["step"]) != expected_step:
            raise ValueError("Trajectory step values must be consecutive and one-indexed")
        index = int(record["query_index"])
        if not 0 <= index < len(public.candidates) or index in queried:
            raise ValueError(f"Invalid or repeated query index at step {expected_step}: {index}")
        queried.add(index)
        expected_condition = condition_dict(public.candidates.iloc[index], public.variable_columns)
        if dict(record["condition"]) != expected_condition:
            raise ValueError(f"Condition does not match query index at step {expected_step}")


def save_trajectory(
    output_dir: str | Path,
    public: PublicDataset,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    expected_rounds: int = 40,
) -> Path:
    """Save one fully auditable trajectory in the required .pt form."""
    validate_trajectory(records, public, expected_rounds=expected_rounds)
    target = Path(output_dir) / "optimization_trajectories" / public.name
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"seed_{seed}.pt"
    # A .pt suffix is requested by the competition, but no tensor object is
    # stored.  Standard-library serialization keeps the official runtime free
    # from a large, otherwise unnecessary PyTorch dependency.
    with path.open("wb") as handle:
        pickle.dump({"seed": int(seed), "dataset": public.name, "trajectory": list(records)}, handle)
    return path


def load_trajectory(path: str | Path) -> Dict[str, Any]:
    """Load a trajectory written by :func:`save_trajectory`."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def official_metrics(records: Sequence[Mapping[str, Any]], global_best: float) -> Dict[str, float]:
    """Compute the four published metrics from a validated completed trajectory."""
    values = np.asarray([float(record["best_so_far"]) for record in records], dtype=float)
    threshold = 0.95 * float(global_best)
    hits = np.flatnonzero(values >= threshold)
    return {
        "initial_round_found_best": float(values[0]),
        "best_found": float(values[-1]),
        "round_to_95_global_best": float(hits[0] + 1) if len(hits) else float(len(values) + 1),
        "AUC_best_so_far": float(np.mean(values)),
    }
