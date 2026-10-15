#!/usr/bin/env python3
"""Stateful, leakage-safe Buchwald/Suzuki closed-loop optimization agent."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preflight_reaction_data import run_preflight
from select_fingerprint_config import choose_configuration


STATE_VERSION = 1
ANNOUNCED_MODELS = {"minimax-2.7", "deepseek-v4-pro", "deepseek-v4-flash"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    condition_columns: list[str] = []
    for row in rows:
        for column in row.get("condition", {}):
            if column not in condition_columns:
                condition_columns.append(column)
    fields = [
        "seed",
        "method",
        "round",
        "candidate_index",
        "observed_yield",
        "best_so_far",
        "predicted_yield",
        "uncertainty",
        "acquisition_value",
        "selection_mode",
        "transaction_id",
        *condition_columns,
    ]
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in rows:
            flat = {key: record.get(key) for key in fields}
            flat.update(record.get("condition", {}))
            writer.writerow(flat)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def split_columns(text: str | None) -> list[str]:
    return [value.strip() for value in (text or "").split(",") if value.strip()]


def safe_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("value must be finite")
    return parsed


def load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATE_VERSION:
        raise ValueError("unsupported state schema")
    return payload


def verify_inputs(state: dict[str, Any]) -> None:
    for name in ("train", "candidates"):
        path = Path(state["paths"][name])
        if sha256(path) != state["hashes"][name]:
            raise RuntimeError(f"{name} file changed after initialization")
    mapping = state["paths"].get("mapping")
    if mapping and sha256(Path(mapping)) != state["hashes"]["mapping"]:
        raise RuntimeError("structure map changed after initialization")


def mapping_records(path: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in payload.get("records", []):
        if str(item.get("review_status", "")).lower() not in {
            "approved", "approved_before_agent_run"
        }:
            continue
        variable = str(item.get("variable", item.get("role", "")))
        smiles = str(item.get("canonical_smiles", "")).strip()
        source = str(item.get("source", item.get("source_url", ""))).strip()
        if not source and str(item.get("status", "")) == "resolved_direct_smiles":
            source = "packaged_direct_smiles"
        if smiles and source:
            result[(variable, str(item.get("value", "")))] = smiles
    return result


def physicochemical(molecule: Any) -> np.ndarray:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    heavy = max(molecule.GetNumHeavyAtoms(), 1)
    aromatic = sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
    return np.asarray(
        [
            np.clip(Descriptors.MolWt(molecule) / 700.0, 0.0, 1.5),
            np.clip((Crippen.MolLogP(molecule) + 3.0) / 12.0, 0.0, 1.0),
            np.clip(rdMolDescriptors.CalcTPSA(molecule) / 200.0, 0.0, 1.0),
            np.clip(Lipinski.NumHDonors(molecule) / 5.0, 0.0, 1.0),
            np.clip(Lipinski.NumHAcceptors(molecule) / 15.0, 0.0, 1.0),
            np.clip(Lipinski.NumRotatableBonds(molecule) / 20.0, 0.0, 1.0),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            aromatic / heavy,
        ],
        dtype=np.float32,
    )


def make_structure_featurizer(
    mapping: dict[tuple[str, str], str],
    variables: list[str],
    bits: int,
) -> Callable[[pd.DataFrame], np.ndarray]:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    cache: dict[tuple[str, str], np.ndarray] = {}

    def component(variable: str, value: str) -> np.ndarray:
        key = (variable, value)
        if key in cache:
            return cache[key]
        smiles = mapping.get(key)
        if not smiles:
            raise KeyError(f"missing structure: {variable}={value}")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit rejected mapped structure: {variable}={value}")
        fingerprint = generator.GetFingerprint(molecule)
        vector = np.zeros(bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        features = np.concatenate([vector, physicochemical(molecule)]).astype(
            np.float32
        )
        if not np.isfinite(features).all():
            raise ValueError(f"non-finite structure features: {variable}={value}")
        cache[key] = features
        return features

    def transform(frame: pd.DataFrame) -> np.ndarray:
        rows = [
            np.concatenate(
                [
                    component(variable, str(row[variable]))
                    for variable in variables
                ]
            )
            for _, row in frame.iterrows()
        ]
        return np.asarray(rows, dtype=np.float32)

    return transform


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_suzuki_transformer(
    public: pd.DataFrame,
    categorical: list[str],
    continuous: list[str],
) -> ColumnTransformer:
    blocks = []
    if categorical:
        blocks.append(("category", one_hot_encoder(), categorical))
    if continuous:
        blocks.append(("continuous", StandardScaler(), continuous))
    transformer = ColumnTransformer(blocks, sparse_threshold=0.0)
    transformer.fit(public[[*categorical, *continuous]])
    return transformer


def fit_surrogate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    state: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = state["configuration"]["surrogate"]
    forest = RandomForestRegressor(
        n_estimators=int(config["random_forest_trees"]),
        min_samples_leaf=int(config["minimum_leaf_size"]),
        max_features=1.0,
        n_jobs=1,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).fit(train_x, train_y)
    model: dict[str, Any] = {"forest": forest, "gp": None, "forest_weight": 1.0}
    events: list[dict[str, Any]] = []
    if state["reaction"] != "suzuki":
        return model, events
    if len(train_y) > int(config["gp_max_train_rows"]):
        events.append(
            {
                "time": utc_now(),
                "type": "gp_capacity_fallback",
                "observed_rows": len(train_y),
                "fallback": "random_forest",
            }
        )
        return model, events

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")
        + WhiteKernel(0.05, noise_level_bounds="fixed")
    )
    last_error = ""
    for alpha in (1e-4, 1e-3, 1e-2):
        try:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                optimizer=None,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(train_x, train_y)
            model.update(
                gp=gp,
                forest_weight=float(config["suzuki_forest_weight"]),
            )
            if alpha != 1e-4:
                events.append(
                    {
                        "time": utc_now(),
                        "type": "gp_jitter_retry",
                        "alpha": alpha,
                    }
                )
            return model, events
        except Exception as error:  # numerical backends raise several exception types
            last_error = f"{type(error).__name__}: {error}"
    events.append(
        {
            "time": utc_now(),
            "type": "gp_numerical_fallback",
            "fallback": "random_forest",
            "error": last_error,
        }
    )
    return model, events


def predict_surrogate(
    model: dict[str, Any], candidates_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    forest = model["forest"]
    tree_values = np.asarray(
        [tree.predict(candidates_x) for tree in forest.estimators_], dtype=float
    )
    forest_mean = tree_values.mean(axis=0)
    forest_std = np.maximum(tree_values.std(axis=0), 1e-8)
    gp = model.get("gp")
    if gp is None:
        return forest_mean, forest_std
    gp_mean, gp_std = gp.predict(candidates_x, return_std=True)
    weight = float(model["forest_weight"])
    mean = weight * forest_mean + (1.0 - weight) * gp_mean
    variance = (
        weight * forest_std**2
        + (1.0 - weight) * gp_std**2
        + weight * (1.0 - weight) * (forest_mean - gp_mean) ** 2
    )
    return mean, np.maximum(np.sqrt(variance), 1e-8)


def expected_improvement(
    mean: np.ndarray, uncertainty: np.ndarray, incumbent: float
) -> np.ndarray:
    uncertainty = np.maximum(uncertainty, 1e-8)
    improvement = mean - incumbent
    z = improvement / uncertainty
    return improvement * norm.cdf(z) + uncertainty * norm.pdf(z)


def init_state(args: argparse.Namespace) -> dict[str, Any]:
    train = Path(args.train).resolve()
    candidates = Path(args.candidates).resolve()
    mapping = Path(args.mapping).resolve() if args.mapping else None
    molecular = split_columns(args.molecular_columns)
    categorical = split_columns(args.categorical_columns)
    continuous = split_columns(args.continuous_columns)
    if args.reaction == "buchwald" and molecular and not categorical:
        # The Buchwald route represents every molecular identity as a
        # categorical variable even though its CLI historically required only
        # --molecular-columns.
        categorical = list(molecular)
    variables = molecular if args.reaction == "buchwald" else [*categorical, *continuous]
    if args.reaction == "buchwald" and (not mapping or not molecular):
        raise ValueError("Buchwald requires --mapping and --molecular-columns")
    if args.reaction == "suzuki" and not variables:
        raise ValueError("Suzuki requires categorical and/or continuous columns")
    if molecular and not mapping:
        raise ValueError("molecular columns require --mapping")
    if molecular and not set(molecular).issubset(categorical):
        raise ValueError("molecular columns must also be categorical columns")
    preflight = run_preflight(
        train_path=train,
        candidates_path=candidates,
        variables=variables,
        yield_column=args.yield_column,
        structure_map=mapping if molecular else None,
        structure_variables=molecular or None,
        continuous_variables=continuous,
    )
    if preflight["status"] != "ready":
        raise ValueError("preflight blocked: " + "; ".join(preflight["blockers"]))
    train_frame = pd.read_csv(train)
    candidate_frame = pd.read_csv(candidates)
    if args.rounds < 1 or args.rounds > len(candidate_frame):
        raise ValueError("rounds must be in 1..candidate-row-count")
    unique_molecules = (
        sum(
            len(
                set(train_frame[column].astype(str))
                | set(candidate_frame[column].astype(str))
            )
            for column in molecular
        )
        if molecular
        else 0
    )
    configuration = choose_configuration(
        reaction=args.reaction,
        labeled_rows=len(train_frame),
        unique_molecules=unique_molecules,
        candidate_rows=len(candidate_frame),
        molecular_roles=max(len(molecular), len(variables), 1),
        invalid_fraction=0.0,
        memory_budget_mb=args.memory_budget_mb,
        enable_structure=bool(mapping and molecular),
    )
    if configuration["status"] != "ready":
        raise ValueError("configuration blocked: " + "; ".join(configuration["required_actions"]))
    paths = {
        "train": str(train),
        "candidates": str(candidates),
        "mapping": str(mapping) if mapping else None,
        "trajectory": str(Path(args.state).resolve().with_suffix(".trajectory.csv")),
    }
    hashes = {"train": sha256(train), "candidates": sha256(candidates)}
    if mapping:
        hashes["mapping"] = sha256(mapping)
    return {
        "schema_version": STATE_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "reaction": args.reaction,
        "seed": args.seed,
        "round_budget": args.rounds,
        "yield_column": args.yield_column,
        "molecular_columns": molecular,
        "categorical_columns": categorical,
        "continuous_columns": continuous,
        "paths": paths,
        "hashes": hashes,
        "preflight": preflight,
        "configuration": configuration,
        "pending": None,
        "feedback": [],
        "fallback_events": [],
        "consecutive_surrogate_failures": 0,
    }


def recommend(state_path: Path, output: Path | None) -> dict[str, Any]:
    state = load_state(state_path)
    verify_inputs(state)
    if state["pending"] is not None:
        raise RuntimeError("feedback is required for the pending recommendation")
    if len(state["feedback"]) >= state["round_budget"]:
        raise RuntimeError("round budget is complete")
    train = pd.read_csv(state["paths"]["train"])
    candidates = pd.read_csv(state["paths"]["candidates"])
    queried = {int(item["candidate_index"]) for item in state["feedback"]}
    available = [index for index in range(len(candidates)) if index not in queried]
    if not available:
        raise RuntimeError("candidate pool is exhausted")

    train_y = train[state["yield_column"]].to_numpy(float)
    feedback_indices = [int(item["candidate_index"]) for item in state["feedback"]]
    feedback_y = np.asarray(
        [float(item["observed_yield"]) for item in state["feedback"]], dtype=float
    )
    round_index = len(state["feedback"])
    rng = np.random.default_rng(int(state["seed"]) * 1_000_003 + round_index)
    chunk_size = int(
        state["configuration"]["structure_features"]["candidate_batch_size"]
    )

    if state["reaction"] == "buchwald":
        variables = state["molecular_columns"]
        mapping = mapping_records(Path(state["paths"]["mapping"]))
        transform = make_structure_featurizer(
            mapping,
            variables,
            int(state["configuration"]["structure_features"]["fingerprint_bits"]),
        )
        initial_x = transform(train[variables])
        feedback_x = (
            transform(candidates.iloc[feedback_indices][variables])
            if feedback_indices
            else np.empty((0, initial_x.shape[1]), dtype=np.float32)
        )

        def candidate_features(indices: list[int]) -> np.ndarray:
            return transform(candidates.iloc[indices][variables])

        method = "buchwald_structure_random_forest_EI"
    else:
        variables = [*state["categorical_columns"], *state["continuous_columns"]]
        public = pd.concat([train[variables], candidates[variables]], ignore_index=True)
        transformer = make_suzuki_transformer(
            public,
            state["categorical_columns"],
            state["continuous_columns"],
        )
        initial_identity = np.asarray(transformer.transform(train[variables]), dtype=float)
        feedback_identity = (
            np.asarray(
                transformer.transform(candidates.iloc[feedback_indices][variables]),
                dtype=float,
            )
            if feedback_indices
            else np.empty((0, initial_identity.shape[1]), dtype=float)
        )

        molecular = state.get("molecular_columns", [])
        mapping_path = state["paths"].get("mapping")
        if molecular and mapping_path:
            mapping = mapping_records(Path(mapping_path))
            structure_transform = make_structure_featurizer(
                mapping,
                molecular,
                int(state["configuration"]["structure_features"]["fingerprint_bits"]),
            )
            initial_x = np.hstack([initial_identity, structure_transform(train[molecular])])
            feedback_x = (
                np.hstack(
                    [
                        feedback_identity,
                        structure_transform(candidates.iloc[feedback_indices][molecular]),
                    ]
                )
                if feedback_indices
                else np.empty((0, initial_x.shape[1]), dtype=float)
            )

            def candidate_features(indices: list[int]) -> np.ndarray:
                identity = np.asarray(
                    transformer.transform(candidates.iloc[indices][variables]), dtype=float
                )
                structure = structure_transform(candidates.iloc[indices][molecular])
                return np.hstack([identity, structure])

            method = "generic_mixed_identity_structure_gp_forest_EI"
        else:
            initial_x = initial_identity
            feedback_x = feedback_identity

            def candidate_features(indices: list[int]) -> np.ndarray:
                return np.asarray(
                    transformer.transform(candidates.iloc[indices][variables]), dtype=float
                )

            method = "suzuki_hybrid_rbf_gp_forest_EI"

    train_x = np.vstack([initial_x, feedback_x])
    all_y = np.concatenate([train_y, feedback_y])
    selection_mode = "expected_improvement"
    fallback_events: list[dict[str, Any]] = []
    try:
        model, model_events = fit_surrogate(train_x, all_y, state, rng)
        fallback_events.extend(model_events)
        state["consecutive_surrogate_failures"] = 0
        progress = round_index / max(state["round_budget"] - 1, 1)
        epsilon = 0.15 + progress * (0.01 - 0.15)
        incumbent = float(np.max(all_y))
        if rng.random() < epsilon:
            selected = int(rng.choice(available))
            selected_x = candidate_features([selected])
            mean, uncertainty = predict_surrogate(model, selected_x)
            acquisition = expected_improvement(mean, uncertainty, incumbent)
            predicted_value = float(mean[0])
            uncertainty_value = float(uncertainty[0])
            acquisition_value = float(acquisition[0])
            selection_mode = "random_exploration"
        else:
            best_score = -math.inf
            selected = -1
            predicted_value = 0.0
            uncertainty_value = 0.0
            acquisition_value = 0.0
            for start in range(0, len(available), max(chunk_size, 1)):
                indices = available[start : start + max(chunk_size, 1)]
                mean, uncertainty = predict_surrogate(
                    model, candidate_features(indices)
                )
                acquisition = expected_improvement(mean, uncertainty, incumbent)
                local_best = float(np.max(acquisition))
                ties = np.flatnonzero(
                    np.isclose(acquisition, local_best, rtol=1e-10, atol=1e-12)
                )
                local = int(rng.choice(ties))
                candidate_index = int(indices[local])
                if local_best > best_score or (
                    math.isclose(local_best, best_score) and candidate_index < selected
                ):
                    best_score = local_best
                    selected = candidate_index
                    predicted_value = float(mean[local])
                    uncertainty_value = float(uncertainty[local])
                    acquisition_value = float(acquisition[local])
    except Exception as error:
        state["consecutive_surrogate_failures"] += 1
        if state["consecutive_surrogate_failures"] >= 3:
            raise RuntimeError("three consecutive surrogate failures; run is blocked") from error
        selected = int(rng.choice(available))
        predicted_value = None
        uncertainty_value = None
        acquisition_value = None
        selection_mode = "surrogate_failure_exploration"
        fallback_events.append(
            {
                "time": utc_now(),
                "type": "surrogate_failure_exploration",
                "error": f"{type(error).__name__}: {error}",
            }
        )

    transaction = hashlib.sha256(
        f"{state['seed']}:{round_index + 1}:{selected}:{state['hashes']['candidates']}".encode()
    ).hexdigest()[:24]
    pending = {
        "transaction_id": transaction,
        "created_at": utc_now(),
        "round": round_index + 1,
        "candidate_index": selected,
        "condition": {
            column: candidates.iloc[selected][column].item()
            if hasattr(candidates.iloc[selected][column], "item")
            else candidates.iloc[selected][column]
            for column in variables
        },
        "predicted_yield": predicted_value,
        "uncertainty": uncertainty_value,
        "acquisition_value": acquisition_value,
        "selection_mode": selection_mode,
        "method": method,
    }
    state["fallback_events"].extend(fallback_events)
    state["pending"] = pending
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    if output:
        atomic_json(output, pending)
    return pending


def feedback(state_path: Path, observed_yield: float) -> dict[str, Any]:
    state = load_state(state_path)
    verify_inputs(state)
    pending = state.get("pending")
    if pending is None:
        raise RuntimeError("there is no pending recommendation")
    observed = safe_float(observed_yield)
    prior = [float(item["observed_yield"]) for item in state["feedback"]]
    record = {
        "seed": state["seed"],
        "method": pending["method"],
        "round": pending["round"],
        "candidate_index": pending["candidate_index"],
        "observed_yield": observed,
        "best_so_far": max([*prior, observed]),
        "predicted_yield": pending["predicted_yield"],
        "uncertainty": pending["uncertainty"],
        "acquisition_value": pending["acquisition_value"],
        "selection_mode": pending["selection_mode"],
        "transaction_id": pending["transaction_id"],
        "condition": pending["condition"],
        "committed_at": utc_now(),
    }
    state["feedback"].append(record)
    state["pending"] = None
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    atomic_csv(Path(state["paths"]["trajectory"]), state["feedback"])
    return {
        "committed": True,
        "round": record["round"],
        "candidate_index": record["candidate_index"],
        "observed_yield": observed,
        "best_so_far": record["best_so_far"],
        "remaining_rounds": state["round_budget"] - len(state["feedback"]),
    }


def status(state_path: Path) -> dict[str, Any]:
    state = load_state(state_path)
    verify_inputs(state)
    return {
        "reaction": state["reaction"],
        "seed": state["seed"],
        "completed_rounds": len(state["feedback"]),
        "round_budget": state["round_budget"],
        "pending": state["pending"],
        "queried_indices": [item["candidate_index"] for item in state["feedback"]],
        "best_observed": max(
            [float(item["observed_yield"]) for item in state["feedback"]],
            default=None,
        ),
        "configuration": state["configuration"],
        "fallback_events": state["fallback_events"],
        "hashes": state["hashes"],
    }


def doctor() -> dict[str, Any]:
    required = ("numpy", "pandas", "scipy", "sklearn", "rdkit")
    packages = {name: importlib.util.find_spec(name) is not None for name in required}
    return {
        "ready": all(packages.values()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "credential_policy": (
            "Provider keys are configured through secure host settings; "
            "this command never reads or prints them."
        ),
    }


def provider_check(base_url: str, model: str) -> dict[str, Any]:
    parsed = urlparse(base_url)
    errors = []
    warnings = []
    if parsed.scheme != "https" or not parsed.netloc:
        errors.append("provider URL must be valid HTTPS")
    official_deepseek = parsed.netloc.lower() == "api.deepseek.com" and parsed.path.rstrip(
        "/"
    ) in {"", "/v1"}
    if not official_deepseek and not parsed.path.rstrip("/").endswith("/v1"):
        warnings.append("OpenAI-compatible endpoints normally end in /v1")
    if model not in ANNOUNCED_MODELS:
        warnings.append("verify this model against the latest organizer notice")
    return {
        "valid": not errors,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "currently_announced_model": model in ANNOUNCED_MODELS,
        "api_key_checked": False,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--reaction", choices=("buchwald", "suzuki"), required=True)
    init_parser.add_argument("--train", required=True)
    init_parser.add_argument("--candidates", required=True)
    init_parser.add_argument("--mapping")
    init_parser.add_argument("--molecular-columns")
    init_parser.add_argument("--categorical-columns")
    init_parser.add_argument("--continuous-columns")
    init_parser.add_argument("--yield-column", default="Yield")
    init_parser.add_argument("--state", required=True)
    init_parser.add_argument("--rounds", type=int, default=40)
    init_parser.add_argument("--seed", type=int, default=100)
    init_parser.add_argument("--memory-budget-mb", type=int, default=2048)
    recommend_parser = subparsers.add_parser("recommend")
    recommend_parser.add_argument("--state", required=True)
    recommend_parser.add_argument("--output")
    feedback_parser = subparsers.add_parser("feedback")
    feedback_parser.add_argument("--state", required=True)
    feedback_parser.add_argument("--yield", dest="observed_yield", type=float, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state", required=True)
    subparsers.add_parser("doctor")
    provider_parser = subparsers.add_parser("provider-check")
    provider_parser.add_argument("--base-url", required=True)
    provider_parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.command == "init":
        result = init_state(args)
        atomic_json(Path(args.state).resolve(), result)
    elif args.command == "recommend":
        result = recommend(
            Path(args.state).resolve(),
            Path(args.output).resolve() if args.output else None,
        )
    elif args.command == "feedback":
        result = feedback(Path(args.state).resolve(), args.observed_yield)
    elif args.command == "status":
        result = status(Path(args.state).resolve())
    elif args.command == "doctor":
        result = doctor()
    else:
        result = provider_check(args.base_url, args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command in {"doctor", "provider-check"} and not result.get(
        "ready", result.get("valid", False)
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
