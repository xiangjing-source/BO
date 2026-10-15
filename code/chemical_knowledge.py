"""Transparent, editable chemistry descriptors for categorical conditions.

Values are deliberately *descriptors*, not outcome labels.  Each value is
scaled from 0 to 1, where 0 means low/not applicable and 1 means high.  A
human-maintained JSON card file can override the conservative name-based
defaults for any exact variable/value pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "electron_donation",
    "electron_withdrawal",
    "steric_bulk",
    "leaving_group_activation",
    "base_strength",
    "solvent_polarity",
    "coordination_propensity",
    "boron_transferability",
    "stability",
)


class ChemicalKnowledgeFeaturizer:
    """Append role-aware chemistry descriptors to categorical input features."""

    def __init__(self, variable_columns: Sequence[str], card_path: str | Path | None = None):
        self.variable_columns = list(variable_columns)
        self.cards: Dict[tuple[str, str], Dict[str, float]] = {}
        if card_path is not None:
            self._load_cards(card_path)

    def _load_cards(self, card_path: str | Path) -> None:
        payload = json.loads(Path(card_path).read_text(encoding="utf-8"))
        for card in payload.get("cards", []):
            variable = str(card["variable"])
            value = str(card["value"])
            features = card.get("features", {})
            unknown = set(features) - set(FEATURE_NAMES)
            if unknown:
                raise ValueError(f"Unknown chemistry feature names: {sorted(unknown)}")
            parsed = {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
            if any(not 0.0 <= value <= 1.0 for value in parsed.values()):
                raise ValueError("Chemical feature scores must be between 0 and 1")
            self.cards[(variable, value)] = parsed

    @staticmethod
    def _blank() -> Dict[str, float]:
        return {name: 0.0 for name in FEATURE_NAMES}

    def _fallback(self, variable: str, value: str) -> Dict[str, float]:
        """Conservative, auditable name-based descriptors used before manual review."""
        features = self._blank()
        name = value.lower()
        role = variable.lower()

        if role in {"ligand"}:
            # Electron-rich, bulky trialkyl/biaryl phosphines generally
            # promote oxidative addition/reductive elimination, but values are
            # intentionally graded rather than treated as hard exclusions.
            if "adamantyl" in name:
                features["electron_donation"] = 0.95
                features["steric_bulk"] = 1.00
            elif "tert-butyl" in name or "ditertbutyl" in name or "tritertbutyl" in name:
                features["electron_donation"] = 0.90
                features["steric_bulk"] = 0.85
            elif "cyclohexyl" in name:
                features["electron_donation"] = 0.80
                features["steric_bulk"] = 0.75
            elif "diphenyl" in name or "triphenyl" in name:
                features["electron_donation"] = 0.40
                features["steric_bulk"] = 0.40
            if "tri(propan-2-yl)" in name:
                features["steric_bulk"] = max(features["steric_bulk"], 0.75)
            if "dimethoxy" in name:
                features["electron_donation"] = max(features["electron_donation"], 0.75)
            if "ferrocene" in name or "xanthen" in name:
                features["coordination_propensity"] = 0.75
                features["steric_bulk"] = max(features["steric_bulk"], 0.60)
            if name == "nothing":
                features["stability"] = 0.0
            else:
                features["stability"] = 0.60

        if role in {"electrophile", "reactant2"}:
            if "iodo" in name:
                features["leaving_group_activation"] = 1.0
            elif "bromo" in name:
                features["leaving_group_activation"] = 0.70
            elif "chloro" in name:
                features["leaving_group_activation"] = 0.40
            elif "trifluoromethanesulfonate" in name:
                features["leaving_group_activation"] = 0.65
            if any(token in name for token in ("trifluoro", "pyrid", "quinol", "indazol")):
                features["electron_withdrawal"] = 0.60
            if "methoxy" in name:
                features["electron_donation"] = 0.70
            elif "ethyl" in name or "methyl" in name:
                features["electron_donation"] = 0.25
            if any(token in name for token in ("pyrid", "quinol", "indazol")):
                features["coordination_propensity"] = 0.55
            features["stability"] = 0.70

        if role == "product":
            # Product descriptors make cross-product transfer explicit in the
            # Buchwald task.  They are parsed from names only, never fitted to
            # yields: aryl electronics and N-heteroarene coordination are the
            # primary transferable distinctions in the supplied products.
            if "trifluoro" in name:
                features["electron_withdrawal"] = 0.80
            if "methoxy" in name:
                features["electron_donation"] = 0.70
            elif "ethyl" in name or "methyl" in name:
                features["electron_donation"] = 0.25
            if "pyridin" in name or "quinolin" in name:
                features["electron_withdrawal"] = max(features["electron_withdrawal"], 0.60)
                features["coordination_propensity"] = 0.80
            features["stability"] = 0.70

        if role == "nucleophile":
            if "trifluoro" in name and "bor" in name:
                features["boron_transferability"] = 0.65
                features["stability"] = 0.85
            elif "boronic acid" in name:
                features["boron_transferability"] = 0.85
                features["stability"] = 0.55
            elif "dioxaborol" in name or "boronate" in name:
                features["boron_transferability"] = 0.60
                features["stability"] = 0.80
            if any(token in name for token in ("pyrid", "quinol", "indazol")):
                features["coordination_propensity"] = 0.50

        if role == "base":
            if name == "nothing":
                return features
            if "phosphanyl" in name or "phosphazene" in name:
                features["base_strength"] = 1.00
                features["coordination_propensity"] = 0.10
            elif "hexahydropyrimido" in name:
                features["base_strength"] = 0.85
                features["coordination_propensity"] = 0.20
            elif any(token in name for token in ("butoxide", "hydroxide", "guanidine")):
                features["base_strength"] = 0.85
            elif any(token in name for token in ("phosphate", "carbonate", "bicarbonate", "fluoride")):
                features["base_strength"] = 0.55
            elif "diethyl" in name or "amine" in name:
                features["base_strength"] = 0.40
            features["coordination_propensity"] = 0.40 if any(
                token in name for token in ("fluoride", "hydroxide", "amine")
            ) else 0.15
            features["stability"] = 0.70

        if role == "solvent":
            if any(token in name for token in ("dimethylformamide", "methanol")):
                features["solvent_polarity"] = 0.90
            elif "acetonitrile" in name:
                features["solvent_polarity"] = 0.75
            elif "oxolane" in name or "ether" in name:
                features["solvent_polarity"] = 0.45
            features["coordination_propensity"] = 0.60 if any(
                token in name for token in ("dimethylformamide", "oxolane", "ether", "acetonitrile")
            ) else 0.20
            features["stability"] = 0.80

        if role == "additive":
            if any(token in name for token in ("oxazole", "benzoxazole")):
                features["coordination_propensity"] = 0.45
            if "carboxylate" in name:
                features["electron_withdrawal"] = 0.45
                features["coordination_propensity"] = max(features["coordination_propensity"], 0.55)
            if "difluoro" in name:
                features["electron_withdrawal"] = max(features["electron_withdrawal"], 0.60)
            if "pyrrol" in name:
                features["electron_donation"] = 0.45
            if "phenyl" in name or "furan" in name or "thiophen" in name:
                features["steric_bulk"] = 0.40
            features["stability"] = 0.60

        return features

    def descriptors(self, variable: str, value: str) -> Dict[str, float]:
        fallback = self._fallback(variable, value)
        override = self.cards.get((variable, value))
        return fallback if override is None else override

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        rows: List[List[float]] = []
        for _, row in frame.iterrows():
            feature_row: List[float] = []
            for variable in self.variable_columns:
                descriptor = self.descriptors(variable, str(row[variable]))
                feature_row.extend(descriptor[name] for name in FEATURE_NAMES)
            rows.append(feature_row)
        return np.asarray(rows, dtype=np.float64)

    @property
    def feature_names(self) -> List[str]:
        return [f"{variable}__{name}" for variable in self.variable_columns for name in FEATURE_NAMES]
