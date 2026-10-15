"""Fixed, label-independent molecular features for reaction components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


class MolecularStructureFeaturizer:
    """Concatenate one molecular representation per reaction-variable role.

    Structures come from a reviewed JSON mapping and are never inferred from
    outcome labels.  Each role gets its own feature block so, for example, a
    ligand substructure is not confused with the same substructure in an
    additive.
    """

    PHYSICOCHEMICAL_NAMES = (
        "molecular_weight",
        "logp",
        "tpsa",
        "h_bond_donors",
        "h_bond_acceptors",
        "rotatable_bonds",
        "fraction_csp3",
        "aromatic_atom_fraction",
    )

    def __init__(
        self,
        variable_columns: Sequence[str],
        structure_path: str | Path,
        fingerprint_bits: int = 64,
        radius: int = 2,
        include_fingerprint: bool = True,
        include_physicochemical: bool = True,
    ):
        if fingerprint_bits < 8:
            raise ValueError("fingerprint_bits must be at least 8")
        if not include_fingerprint and not include_physicochemical:
            raise ValueError("At least one molecular feature family must be enabled")
        self.variable_columns = list(variable_columns)
        self.fingerprint_bits = int(fingerprint_bits)
        self.radius = int(radius)
        self.include_fingerprint = bool(include_fingerprint)
        self.include_physicochemical = bool(include_physicochemical)
        payload = json.loads(Path(structure_path).read_text(encoding="utf-8"))
        self._molecules: dict[tuple[str, str], Chem.Mol] = {}
        for record in payload.get("records", []):
            if not str(record.get("status", "")).startswith("resolved"):
                continue
            molecule = Chem.MolFromSmiles(str(record["canonical_smiles"]))
            if molecule is None:
                raise ValueError(
                    f"Invalid mapped structure for {record['variable']}={record['value']}"
                )
            self._molecules[(str(record["variable"]), str(record["value"]))] = molecule
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.fingerprint_bits
        )
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _physicochemical(molecule: Chem.Mol) -> np.ndarray:
        heavy_atoms = max(molecule.GetNumHeavyAtoms(), 1)
        aromatic_atoms = sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())
        # Fixed, label-independent scaling keeps the descriptor magnitudes
        # comparable to binary fingerprint bits without fitting a scaler on
        # candidate outcomes.
        return np.asarray(
            [
                np.clip(Descriptors.MolWt(molecule) / 700.0, 0.0, 1.5),
                np.clip((Crippen.MolLogP(molecule) + 3.0) / 12.0, 0.0, 1.0),
                np.clip(rdMolDescriptors.CalcTPSA(molecule) / 200.0, 0.0, 1.0),
                np.clip(Lipinski.NumHDonors(molecule) / 5.0, 0.0, 1.0),
                np.clip(Lipinski.NumHAcceptors(molecule) / 15.0, 0.0, 1.0),
                np.clip(Lipinski.NumRotatableBonds(molecule) / 20.0, 0.0, 1.0),
                rdMolDescriptors.CalcFractionCSP3(molecule),
                aromatic_atoms / heavy_atoms,
            ],
            dtype=np.float64,
        )

    def _component_features(self, variable: str, value: str) -> np.ndarray:
        key = (variable, value)
        if key in self._cache:
            return self._cache[key]
        molecule = self._molecules.get(key)
        if molecule is None:
            raise KeyError(f"No reviewed molecular structure for {variable}={value}")
        blocks: list[np.ndarray] = []
        if self.include_fingerprint:
            fingerprint = self._generator.GetFingerprint(molecule)
            values = np.zeros(self.fingerprint_bits, dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fingerprint, values)
            blocks.append(values)
        if self.include_physicochemical:
            blocks.append(self._physicochemical(molecule))
        features = np.concatenate(blocks)
        self._cache[key] = features
        return features

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        rows = [
            np.concatenate(
                [
                    self._component_features(variable, str(row[variable]))
                    for variable in self.variable_columns
                ]
            )
            for _, row in frame.iterrows()
        ]
        return np.asarray(rows, dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        per_role: list[str] = []
        if self.include_fingerprint:
            per_role.extend(f"morgan_{index}" for index in range(self.fingerprint_bits))
        if self.include_physicochemical:
            per_role.extend(self.PHYSICOCHEMICAL_NAMES)
        return [
            f"{variable}__{name}"
            for variable in self.variable_columns
            for name in per_role
        ]


class CombinedFeaturizer:
    """Concatenate multiple independently constructed feature blocks."""

    def __init__(self, *featurizers: object):
        if not featurizers:
            raise ValueError("CombinedFeaturizer requires at least one feature block")
        self.featurizers = tuple(featurizers)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.hstack([featurizer.transform(frame) for featurizer in self.featurizers])

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for featurizer in self.featurizers:
            names.extend(featurizer.feature_names)
        return names
