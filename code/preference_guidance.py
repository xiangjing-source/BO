"""Leakage-safe mechanistic preferences for region-lifted Bayesian optimization.

The preference prior is fixed before an optimization run.  It reads only
human-auditable descriptor weights and public condition identities.  Its
influence is calibrated from labels that have already been observed in the
current trajectory; unseen candidate yields are never accepted by this
module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from chemical_knowledge import ChemicalKnowledgeFeaturizer, FEATURE_NAMES


class MechanisticPreferencePrior:
    """Turn a fixed chemistry preference into a normalized regional mean lift.

    For a preferred candidate region ``G`` with uniform weights ``a``, the
    returned direction is

        K(X, G) a / sqrt(a.T K(G, G) a)

    which is the finite-candidate analogue of the region-lifted GP mean shift
    in LGBO.  A categorical Laplacian kernel is used so the mechanism layer is
    independent of the fitted surrogate.  The predictive variance is not
    modified.
    """

    def __init__(
        self,
        dataset_name: str,
        variable_columns: Sequence[str],
        config_path: str | Path,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.variable_columns = list(variable_columns)
        path = Path(config_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            settings = payload["datasets"][self.dataset_name]
        except KeyError as exc:
            raise ValueError(f"No preference prior for {self.dataset_name}") from exc

        self.base_confidence = float(settings.get("base_confidence", 0.6))
        self.region_quantile = float(settings.get("region_quantile", 0.9))
        self.kernel_bandwidth = float(settings.get("kernel_bandwidth", 0.35))
        if not 0.0 <= self.base_confidence <= 1.0:
            raise ValueError("base_confidence must lie in [0, 1]")
        if not 0.5 <= self.region_quantile < 1.0:
            raise ValueError("region_quantile must lie in [0.5, 1)")
        if self.kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")

        card_path = settings.get("knowledge_cards")
        if card_path:
            card_path = (path.parent / str(card_path)).resolve()
        self.featurizer = ChemicalKnowledgeFeaturizer(self.variable_columns, card_path)

        self.weights: dict[tuple[str, str], float] = {}
        for item in settings.get("weights", []):
            variable = str(item["variable"])
            feature = str(item["feature"])
            if variable not in self.variable_columns:
                raise ValueError(f"Unknown preference variable: {variable}")
            if feature not in FEATURE_NAMES:
                raise ValueError(f"Unknown preference feature: {feature}")
            self.weights[(variable, feature)] = float(item["weight"])
        if not self.weights:
            raise ValueError("Preference prior must contain at least one weight")

        self.rationale = list(settings.get("rationale", []))
        self.sources = list(settings.get("sources", []))
        self._cache_key: tuple[Any, ...] | None = None
        self._cached_direction: np.ndarray | None = None
        self._cached_region_size = 0

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a fixed mechanistic score without reading outcome labels."""
        rows: list[float] = []
        for _, row in frame.loc[:, self.variable_columns].iterrows():
            value = 0.0
            for variable in self.variable_columns:
                descriptor = self.featurizer.descriptors(variable, str(row[variable]))
                for feature in FEATURE_NAMES:
                    value += self.weights.get((variable, feature), 0.0) * descriptor[feature]
            rows.append(value)
        return np.asarray(rows, dtype=float)

    def confidence(
        self,
        observed_conditions: pd.DataFrame,
        observed_y: Sequence[float],
    ) -> tuple[float, float]:
        """Calibrate preference strength using current observed evidence only."""
        y = np.asarray(observed_y, dtype=float)
        if len(y) < 5 or np.std(y) < 1e-12:
            return 0.0, 0.0
        prior = self.score(observed_conditions)
        if np.std(prior) < 1e-12:
            return 0.0, 0.0
        result = kendalltau(prior, y, nan_policy="omit")
        tau = 0.0 if result.statistic is None or not np.isfinite(result.statistic) else float(result.statistic)
        # Negative or unrelated guidance is switched off.  The shrinkage term
        # prevents a handful of chance-aligned observations from dominating.
        evidence = max(tau, 0.0) * np.sqrt(len(y) / (len(y) + 20.0))
        return float(np.clip(self.base_confidence * evidence, 0.0, 1.0)), tau

    def _direction(self, candidates: pd.DataFrame) -> np.ndarray:
        values = candidates.loc[:, self.variable_columns].astype(str).to_numpy(dtype=object)
        key = (values.shape, *pd.util.hash_pandas_object(
            candidates.loc[:, self.variable_columns].astype(str), index=True
        ).astype("uint64").tolist())
        if self._cache_key == key and self._cached_direction is not None:
            return self._cached_direction.copy()

        scores = self.score(candidates)
        threshold = float(np.quantile(scores, self.region_quantile))
        region_indices = np.flatnonzero(scores >= threshold)
        if len(region_indices) == 0:
            region_indices = np.asarray([int(np.argmax(scores))])
        region = values[region_indices]

        distance_xg = (values[:, None, :] != region[None, :, :]).mean(axis=2)
        kernel_xg = np.exp(-distance_xg / self.kernel_bandwidth)
        distance_gg = (region[:, None, :] != region[None, :, :]).mean(axis=2)
        kernel_gg = np.exp(-distance_gg / self.kernel_bandwidth)
        weights = np.full(len(region), 1.0 / len(region), dtype=float)
        denominator = float(np.sqrt(max(weights @ kernel_gg @ weights, 1e-12)))
        direction = kernel_xg @ weights / denominator

        self._cache_key = key
        self._cached_direction = direction.copy()
        self._cached_region_size = int(len(region))
        return direction

    def lift(
        self,
        candidates: pd.DataFrame,
        observed_conditions: pd.DataFrame,
        observed_y: Sequence[float],
        strength: float = 1.0,
    ) -> tuple[np.ndarray, Mapping[str, float | int]]:
        """Return a yield-unit mean lift and compact audit diagnostics."""
        y = np.asarray(observed_y, dtype=float)
        calibration_conditions = observed_conditions
        calibration_y = y
        # When the candidate pool fixes a task-defining category (Buchwald's
        # Product is the important example), prefer same-task evidence.  This
        # avoids declaring a mechanism aligned merely because it ranks other
        # products correctly.  Fall back to all observations if fewer than
        # five relevant labels are available.
        relevant = np.ones(len(observed_conditions), dtype=bool)
        for variable in self.variable_columns:
            candidate_values = candidates[variable].astype(str).unique()
            if len(candidate_values) == 1:
                relevant &= (
                    observed_conditions[variable].astype(str).to_numpy()
                    == str(candidate_values[0])
                )
        if int(relevant.sum()) >= 5:
            calibration_conditions = observed_conditions.loc[relevant].reset_index(drop=True)
            calibration_y = y[relevant]
        confidence, tau = self.confidence(calibration_conditions, calibration_y)
        scale = max(float(np.std(y, ddof=1)), 1.0)
        direction = self._direction(candidates)
        shift = float(strength) * confidence * scale * direction
        return shift, {
            "confidence": confidence,
            "kendall_tau": tau,
            "region_size": self._cached_region_size,
            "calibration_count": int(len(calibration_y)),
            "max_shift": float(np.max(shift)) if len(shift) else 0.0,
        }
