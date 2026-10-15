"""Leakage-safe, missing-aware electronic-structure features.

The frozen feature table contains calculations for individual reaction
components, never reaction yields.  Lookup keys include the dataset and the
reaction-variable column so that identically named chemicals in different
roles cannot be silently conflated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import KFold


class DFTElectronicFeaturizer:
    """Concatenate robustly scaled DFT descriptors for each reaction role.

    Scaling uses only the frozen DFT table.  Missing components receive the
    role median (therefore a scaled value of zero) and a separate availability
    flag.  No outcome-dependent imputation or fitting is performed.
    """

    FEATURE_COLUMNS = (
        "homo_ev",
        "lumo_ev",
        "homo_lumo_gap_ev",
        "dipole_debye",
        "atom_count",
        "hirshfeld_charge_min",
        "hirshfeld_charge_max",
        "hirshfeld_abs_charge_mean",
        "mayer_bond_density",
    )
    EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + (
        "hirshfeld_charge_std",
        "mayer_atomic_charge_abs_mean",
        "mayer_total_valence_mean",
        "mayer_total_valence_std",
        "mayer_bond_order_mean",
        "mayer_bond_order_max",
    )

    REQUIRED_COLUMNS = (
        "dataset",
        "original_dataset_column",
        "chemical_name",
        "calculation_id",
        *FEATURE_COLUMNS,
    )

    def __init__(
        self,
        dataset_name: str,
        variable_columns: Sequence[str],
        feature_path: str | Path,
        clip: float = 5.0,
        feature_columns: Sequence[str] | None = None,
        availability_dropout: float = 0.0,
        dropout_seed: int = 0,
    ):
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.dataset_name = str(dataset_name)
        self.variable_columns = list(variable_columns)
        self.clip = float(clip)
        if not 0.0 <= float(availability_dropout) < 1.0:
            raise ValueError("availability_dropout must lie in [0, 1)")
        self.availability_dropout = float(availability_dropout)
        self.dropout_seed = int(dropout_seed)
        self.feature_columns = tuple(feature_columns or self.FEATURE_COLUMNS)
        if (
            not self.feature_columns
            or not set(self.feature_columns) <= set(self.EXTENDED_FEATURE_COLUMNS)
        ):
            raise ValueError(
                "feature_columns must be a non-empty subset of EXTENDED_FEATURE_COLUMNS"
            )
        table = pd.read_csv(feature_path)
        forbidden = [column for column in table.columns if "yield" in column.lower()]
        if forbidden:
            raise ValueError(f"DFT feature table must not contain outcome columns: {forbidden}")
        required = (
            "dataset",
            "original_dataset_column",
            "chemical_name",
            "calculation_id",
            *self.feature_columns,
        )
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f"DFT feature table is missing columns: {missing}")
        table = table.loc[table["dataset"].astype(str) == self.dataset_name].copy()
        if table.empty:
            raise ValueError(f"No DFT records for dataset {self.dataset_name}")
        if self.availability_dropout > 0:
            rng = np.random.default_rng(self.dropout_seed)
            drop_indices: list[int] = []
            for variable in self.variable_columns:
                indices = table.index[
                    table["original_dataset_column"].astype(str) == variable
                ].to_numpy()
                count = min(
                    int(np.floor(self.availability_dropout * len(indices))),
                    max(len(indices) - 1, 0),
                )
                if count:
                    drop_indices.extend(
                        rng.choice(indices, size=count, replace=False).tolist()
                    )
            table = table.drop(index=drop_indices)
        keys = ["original_dataset_column", "chemical_name"]
        duplicates = table.duplicated(keys, keep=False)
        if duplicates.any():
            values = table.loc[duplicates, keys].drop_duplicates().to_dict("records")
            raise ValueError(f"Ambiguous DFT component keys: {values}")
        numeric = table.loc[:, self.feature_columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("DFT feature table contains non-finite values")
        table.loc[:, self.feature_columns] = numeric
        self._lookup = {
            (str(row["original_dataset_column"]), str(row["chemical_name"])): np.asarray(
                [float(row[column]) for column in self.feature_columns], dtype=np.float64
            )
            for _, row in table.iterrows()
        }
        self._centres: dict[str, np.ndarray] = {}
        self._scales: dict[str, np.ndarray] = {}
        for variable in self.variable_columns:
            subset = table.loc[
                table["original_dataset_column"].astype(str) == variable,
                self.feature_columns,
            ].to_numpy(dtype=float)
            # A role can be entirely absent from an incremental DFT campaign.
            # Dataset-level statistics are a deterministic label-free fallback.
            if len(subset) == 0:
                subset = table.loc[:, self.feature_columns].to_numpy(dtype=float)
            centre = np.median(subset, axis=0)
            q25, q75 = np.quantile(subset, [0.25, 0.75], axis=0)
            scale = q75 - q25
            # Constant descriptors are uninformative within a role.  Setting
            # their scale to one leaves both observed and imputed values finite.
            scale = np.where(scale > 1e-12, scale, 1.0)
            self._centres[variable] = centre
            self._scales[variable] = scale

    def _component_features(self, variable: str, value: str) -> np.ndarray:
        raw = self._lookup.get((variable, value))
        available = raw is not None
        if raw is None:
            raw = self._centres[variable]
        scaled = np.clip(
            (raw - self._centres[variable]) / self._scales[variable],
            -self.clip,
            self.clip,
        )
        return np.concatenate([scaled, np.asarray([float(available)], dtype=np.float64)])

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing_columns = [column for column in self.variable_columns if column not in frame]
        if missing_columns:
            raise ValueError(f"Condition frame is missing variables: {missing_columns}")
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

    def coverage(self, frame: pd.DataFrame) -> dict[str, object]:
        """Return component and complete-condition coverage without labels."""
        by_role: dict[str, dict[str, float | int]] = {}
        availability: list[np.ndarray] = []
        for variable in self.variable_columns:
            present = np.asarray(
                [
                    (variable, str(value)) in self._lookup
                    for value in frame[variable].astype(str)
                ],
                dtype=bool,
            )
            availability.append(present)
            by_role[variable] = {
                "rows": int(len(present)),
                "covered_rows": int(present.sum()),
                "row_fraction": float(present.mean()) if len(present) else 0.0,
                "unique_values": int(frame[variable].astype(str).nunique()),
                "covered_unique_values": int(
                    sum(
                        (variable, value) in self._lookup
                        for value in frame[variable].astype(str).unique()
                    )
                ),
            }
        matrix = np.column_stack(availability)
        return {
            "dataset": self.dataset_name,
            "rows": int(len(frame)),
            "roles": by_role,
            "complete_rows": int(matrix.all(axis=1).sum()),
            "complete_row_fraction": float(matrix.all(axis=1).mean()) if len(matrix) else 0.0,
            "mean_components_available": float(matrix.sum(axis=1).mean()) if len(matrix) else 0.0,
            "simulated_availability_dropout": self.availability_dropout,
            "dropout_seed": self.dropout_seed if self.availability_dropout else None,
        }

    @property
    def feature_names(self) -> list[str]:
        per_role = [f"dft_{name}_robust_z" for name in self.feature_columns]
        per_role.append("dft_available")
        return [
            f"{variable}__{name}"
            for variable in self.variable_columns
            for name in per_role
        ]


class MechanisticDFTFeaturizer:
    """Low-dimensional reaction-site and mechanism-interaction descriptors.

    Unlike :class:`DFTElectronicFeaturizer`, this representation does not
    repeat every global molecular scalar in every reaction role.  It selects
    atom- and bond-local quantities associated with oxidative addition,
    transmetalation/deprotonation and donor competition, then adds a small
    number of explicit cross-role interactions.  Scaling is outcome-free and
    uses only the frozen component table.
    """

    IDENTIFIERS = (
        "dataset",
        "calculation_id",
        "original_dataset_column",
        "chemical_name",
    )
    BUCHWALD_FIELDS = {
        "Reactant2": (
            "lumo_ev",
            "electrophile_c_hirshfeld",
            "leaving_group_hirshfeld",
            "c_leaving_mayer_bond_order",
            "c_leaving_distance_angstrom",
            "heteroaryl_n_min_hirshfeld",
        ),
        "Ligand": (
            "homo_ev",
            "phosphorus_hirshfeld_mean",
            "phosphorus_mayer_charge_mean",
            "phosphorus_total_valence_mean",
        ),
        "Additive": (
            "homo_ev",
            "lumo_ev",
            "additive_n_hirshfeld",
            "additive_o_hirshfeld",
            "additive_c3_hirshfeld",
            "additive_c5_hirshfeld",
            "additive_n_o_mayer_bond_order",
            "additive_n_o_distance_angstrom",
        ),
        "Base": (
            "homo_ev",
            "donor_hirshfeld_min",
            "donor_free_valence_max",
        ),
        "Product": (
            "lumo_ev",
            "donor_hirshfeld_min",
        ),
    }
    SUZUKI_FIELDS = {
        "Electrophile": (
            "lumo_ev",
            "electrophile_c_hirshfeld",
            "leaving_group_hirshfeld",
            "c_leaving_mayer_bond_order",
            "c_leaving_distance_angstrom",
            "heteroaryl_n_min_hirshfeld",
        ),
        "Nucleophile": (
            "homo_ev",
            "boron_hirshfeld",
            "transfer_c_hirshfeld",
            "c_b_mayer_bond_order",
            "b_hetero_mayer_bond_order_mean",
        ),
        "Ligand": (
            "homo_ev",
            "phosphorus_hirshfeld_mean",
            "phosphorus_mayer_charge_mean",
            "phosphorus_total_valence_mean",
            "phosphorus_count",
        ),
        "Base": (
            "homo_ev",
            "donor_hirshfeld_min",
            "donor_free_valence_max",
        ),
        "Solvent": (
            "homo_ev",
            "dipole_debye",
            "donor_hirshfeld_min",
        ),
    }

    def __init__(
        self,
        dataset_name: str,
        variable_columns: Sequence[str],
        feature_path: str | Path,
        profile: str = "full",
        clip: float = 5.0,
    ):
        if profile not in {"full", "compact", "interactions", "poisoning"}:
            raise ValueError(
                "profile must be full, compact, interactions or poisoning"
            )
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.dataset_name = str(dataset_name)
        self.variable_columns = list(variable_columns)
        self.profile = profile
        self.clip = float(clip)
        if self.dataset_name == "Buchwald_sub4":
            self._fields = self.BUCHWALD_FIELDS
        elif self.dataset_name == "Suzuki":
            self._fields = self.SUZUKI_FIELDS
        else:
            raise ValueError(
                f"No mechanism mapping is registered for {self.dataset_name}"
            )
        if profile == "poisoning" and self.dataset_name != "Buchwald_sub4":
            raise ValueError("poisoning profile is specific to Buchwald additives")
        missing_roles = [
            role for role in self._fields if role not in self.variable_columns
        ]
        if missing_roles:
            raise ValueError(f"Condition variables lack mechanism roles: {missing_roles}")

        table = pd.read_csv(feature_path)
        forbidden = [column for column in table if "yield" in column.lower()]
        if forbidden:
            raise ValueError(
                f"Mechanistic DFT table must not contain outcomes: {forbidden}"
            )
        required = set(self.IDENTIFIERS)
        required.update(
            column for fields in self._fields.values() for column in fields
        )
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Mechanistic DFT table is missing columns: {missing}")
        table = table.loc[
            table["dataset"].astype(str).eq(self.dataset_name)
        ].copy()
        if table.empty:
            raise ValueError(f"No mechanism-aware DFT rows for {self.dataset_name}")
        duplicates = table.duplicated(
            ["original_dataset_column", "chemical_name"], keep=False
        )
        if duplicates.any():
            raise ValueError("Ambiguous role/name keys in mechanistic DFT table")
        self._table = table
        self._lookup = {
            (str(row["original_dataset_column"]), str(row["chemical_name"])): row
            for _, row in table.iterrows()
        }
        self._centres: dict[tuple[str, str], float] = {}
        self._scales: dict[tuple[str, str], float] = {}
        for role, fields in self._fields.items():
            role_table = table.loc[
                table["original_dataset_column"].astype(str).eq(role)
            ]
            for field in fields:
                values = pd.to_numeric(
                    role_table[field], errors="coerce"
                ).dropna().to_numpy(dtype=float)
                if not len(values):
                    raise ValueError(
                        f"No finite values for mechanism field {role}.{field}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite values in {role}.{field}")
                centre = float(np.median(values))
                q25, q75 = np.quantile(values, [0.25, 0.75])
                scale = float(q75 - q25)
                self._centres[(role, field)] = centre
                self._scales[(role, field)] = scale if scale > 1e-12 else 1.0

    def _value(self, role: str, name: str, field: str) -> tuple[float, float]:
        row = self._lookup.get((role, name))
        available = row is not None and pd.notna(row[field])
        raw = (
            float(row[field])
            if available
            else self._centres[(role, field)]
        )
        scaled = np.clip(
            (raw - self._centres[(role, field)])
            / self._scales[(role, field)],
            -self.clip,
            self.clip,
        )
        return float(scaled), float(available)

    def _role_values(
        self, row: pd.Series, role: str
    ) -> tuple[dict[str, float], float]:
        values: dict[str, float] = {}
        availability = []
        name = str(row[role])
        for field in self._fields[role]:
            value, present = self._value(role, name, field)
            values[field] = value
            availability.append(present)
        return values, float(min(availability))

    @staticmethod
    def _buchwald_interactions(
        values: Mapping[str, Mapping[str, float]]
    ) -> list[float]:
        e = values["Reactant2"]
        ligand = values["Ligand"]
        additive = values["Additive"]
        base = values["Base"]
        product = values["Product"]
        return [
            e["lumo_ev"] - ligand["homo_ev"],
            e["electrophile_c_hirshfeld"] - e["leaving_group_hirshfeld"],
            e["c_leaving_mayer_bond_order"] * ligand["homo_ev"],
            e["electrophile_c_hirshfeld"]
            * ligand["phosphorus_hirshfeld_mean"],
            additive["homo_ev"] - ligand["homo_ev"],
            additive["lumo_ev"] - e["lumo_ev"],
            e["c_leaving_mayer_bond_order"]
            - additive["additive_n_o_mayer_bond_order"],
            additive["additive_o_hirshfeld"]
            - ligand["phosphorus_hirshfeld_mean"],
            base["donor_hirshfeld_min"]
            * e["electrophile_c_hirshfeld"],
            product["lumo_ev"] - e["lumo_ev"],
        ]

    @staticmethod
    def _suzuki_interactions(
        values: Mapping[str, Mapping[str, float]]
    ) -> list[float]:
        e = values["Electrophile"]
        nucleophile = values["Nucleophile"]
        ligand = values["Ligand"]
        base = values["Base"]
        solvent = values["Solvent"]
        return [
            e["lumo_ev"] - ligand["homo_ev"],
            e["electrophile_c_hirshfeld"] - e["leaving_group_hirshfeld"],
            e["c_leaving_mayer_bond_order"] * ligand["homo_ev"],
            e["electrophile_c_hirshfeld"]
            * ligand["phosphorus_hirshfeld_mean"],
            nucleophile["transfer_c_hirshfeld"]
            - nucleophile["boron_hirshfeld"],
            nucleophile["c_b_mayer_bond_order"]
            * base["donor_hirshfeld_min"],
            nucleophile["b_hetero_mayer_bond_order_mean"]
            * base["donor_free_valence_max"],
            solvent["dipole_debye"] * base["donor_hirshfeld_min"],
            nucleophile["transfer_c_hirshfeld"]
            * ligand["phosphorus_hirshfeld_mean"],
        ]

    @property
    def _interaction_names(self) -> list[str]:
        if self.dataset_name == "Buchwald_sub4":
            return [
                "oa_frontier_gap",
                "c_leaving_charge_polarization",
                "c_leaving_bond_x_ligand_homo",
                "electrophile_x_ligand_donor",
                "additive_ligand_frontier_competition",
                "additive_electrophile_lumo_competition",
                "c_leaving_vs_additive_n_o_bond",
                "additive_o_ligand_charge_competition",
                "base_electrophile_activation",
                "product_electrophile_lumo_delta",
            ]
        return [
            "oa_frontier_gap",
            "c_leaving_charge_polarization",
            "c_leaving_bond_x_ligand_homo",
            "electrophile_x_ligand_donor",
            "transfer_carbon_boron_polarization",
            "c_b_bond_x_base_donor",
            "b_hetero_bond_x_base_free_valence",
            "solvent_polarity_x_base_donor",
            "transfer_carbon_x_ligand_donor",
        ]

    def _main_names(self) -> list[str]:
        return [
            f"{role}__{field}_robust_z"
            for role, fields in self._fields.items()
            for field in fields
        ]

    def _availability_roles(self) -> list[str]:
        # Buchwald has complete coverage, while these Suzuki roles are
        # deliberately incomplete in the frozen 74-record snapshot.
        return (
            []
            if self.dataset_name == "Buchwald_sub4"
            else ["Nucleophile", "Ligand", "Base"]
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [role for role in self._fields if role not in frame]
        if missing:
            raise ValueError(f"Condition frame lacks mechanism roles: {missing}")
        result = []
        for _, row in frame.iterrows():
            by_role: dict[str, dict[str, float]] = {}
            availability: dict[str, float] = {}
            main: list[float] = []
            for role in self._fields:
                role_values, role_available = self._role_values(row, role)
                by_role[role] = role_values
                availability[role] = role_available
                main.extend(role_values[field] for field in self._fields[role])
            interactions = (
                self._buchwald_interactions(by_role)
                if self.dataset_name == "Buchwald_sub4"
                else self._suzuki_interactions(by_role)
            )
            flags = [
                availability[role] for role in self._availability_roles()
            ]
            if self.profile == "full":
                features = [*main, *interactions, *flags]
            elif self.profile == "compact":
                # Explicit interactions plus the reaction-centre main effects
                # provide a small, low-variance view for tiny initial tables.
                if self.dataset_name == "Buchwald_sub4":
                    selected = [
                        by_role["Reactant2"]["lumo_ev"],
                        by_role["Reactant2"]["c_leaving_mayer_bond_order"],
                        by_role["Reactant2"]["electrophile_c_hirshfeld"],
                        by_role["Ligand"]["homo_ev"],
                        by_role["Ligand"]["phosphorus_hirshfeld_mean"],
                        by_role["Additive"]["lumo_ev"],
                        by_role["Additive"]["additive_o_hirshfeld"],
                        by_role["Additive"]["additive_n_o_mayer_bond_order"],
                        by_role["Base"]["donor_hirshfeld_min"],
                    ]
                else:
                    selected = [
                        by_role["Electrophile"]["lumo_ev"],
                        by_role["Electrophile"]["c_leaving_mayer_bond_order"],
                        by_role["Nucleophile"]["transfer_c_hirshfeld"],
                        by_role["Nucleophile"]["c_b_mayer_bond_order"],
                        by_role["Ligand"]["homo_ev"],
                        by_role["Ligand"]["phosphorus_hirshfeld_mean"],
                        by_role["Solvent"]["dipole_debye"],
                    ]
                features = [*selected, *interactions, *flags]
            elif self.profile == "interactions":
                features = [*interactions, *flags]
            else:
                additive = by_role["Additive"]
                features = [
                    additive["lumo_ev"],
                    additive["additive_n_hirshfeld"],
                    additive["additive_o_hirshfeld"],
                    additive["additive_c3_hirshfeld"],
                    additive["additive_c5_hirshfeld"],
                    additive["additive_n_o_mayer_bond_order"],
                    additive["additive_n_o_distance_angstrom"],
                    interactions[4],
                    interactions[5],
                    interactions[6],
                    interactions[7],
                ]
            result.append(features)
        matrix = np.asarray(result, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Mechanistic DFT transform produced non-finite values")
        return matrix

    @property
    def feature_names(self) -> list[str]:
        flags = [f"{role}__mechanistic_dft_available" for role in self._availability_roles()]
        interactions = [f"mechanism__{name}" for name in self._interaction_names]
        if self.profile == "full":
            return [*self._main_names(), *interactions, *flags]
        if self.profile == "compact":
            main = (
                [
                    "Reactant2__lumo_ev_robust_z",
                    "Reactant2__c_leaving_mayer_bond_order_robust_z",
                    "Reactant2__electrophile_c_hirshfeld_robust_z",
                    "Ligand__homo_ev_robust_z",
                    "Ligand__phosphorus_hirshfeld_mean_robust_z",
                    "Additive__lumo_ev_robust_z",
                    "Additive__additive_o_hirshfeld_robust_z",
                    "Additive__additive_n_o_mayer_bond_order_robust_z",
                    "Base__donor_hirshfeld_min_robust_z",
                ]
                if self.dataset_name == "Buchwald_sub4"
                else [
                    "Electrophile__lumo_ev_robust_z",
                    "Electrophile__c_leaving_mayer_bond_order_robust_z",
                    "Nucleophile__transfer_c_hirshfeld_robust_z",
                    "Nucleophile__c_b_mayer_bond_order_robust_z",
                    "Ligand__homo_ev_robust_z",
                    "Ligand__phosphorus_hirshfeld_mean_robust_z",
                    "Solvent__dipole_debye_robust_z",
                ]
            )
            return [*main, *interactions, *flags]
        if self.profile == "interactions":
            return [*interactions, *flags]
        return [
            "Additive__lumo_ev_robust_z",
            "Additive__additive_n_hirshfeld_robust_z",
            "Additive__additive_o_hirshfeld_robust_z",
            "Additive__additive_c3_hirshfeld_robust_z",
            "Additive__additive_c5_hirshfeld_robust_z",
            "Additive__additive_n_o_mayer_bond_order_robust_z",
            "Additive__additive_n_o_distance_angstrom_robust_z",
            "mechanism__additive_ligand_frontier_competition",
            "mechanism__additive_electrophile_lumo_competition",
            "mechanism__c_leaving_vs_additive_n_o_bond",
            "mechanism__additive_o_ligand_charge_competition",
        ]

    def coverage(self, frame: pd.DataFrame) -> dict[str, object]:
        by_role = {}
        matrix = []
        for role in self._fields:
            present = np.asarray(
                [(role, str(value)) in self._lookup for value in frame[role]],
                dtype=bool,
            )
            matrix.append(present)
            by_role[role] = {
                "rows": int(len(present)),
                "covered_rows": int(present.sum()),
                "row_fraction": float(present.mean()) if len(present) else 0.0,
                "unique_values": int(frame[role].astype(str).nunique()),
                "covered_unique_values": int(
                    sum(
                        (role, name) in self._lookup
                        for name in frame[role].astype(str).unique()
                    )
                ),
            }
        availability = np.column_stack(matrix)
        return {
            "dataset": self.dataset_name,
            "rows": int(len(frame)),
            "roles": by_role,
            "complete_rows": int(availability.all(axis=1).sum()),
            "complete_row_fraction": float(availability.all(axis=1).mean()),
            "mean_components_available": float(availability.sum(axis=1).mean()),
            "profile": self.profile,
        }


class DFTObservationPreferenceGuide:
    """Evidence-gated DFT neighbourhood lift for an existing BO surrogate.

    The guide is not trained on an unreleased candidate outcome.  At every
    round it checks, by leave-one-observation-out prediction, whether nearby
    DFT conditions have agreed with outcomes observed so far.  Negative or
    undefined rank correlation switches the lift off.
    """

    def __init__(
        self,
        featurizer: DFTElectronicFeaturizer,
        neighbors: int = 5,
        lift_scale: float = 0.25,
    ):
        if neighbors < 2 or lift_scale <= 0:
            raise ValueError("neighbors must be at least 2 and lift_scale positive")
        self.featurizer = featurizer
        self.neighbors = int(neighbors)
        self.lift_scale = float(lift_scale)

    def _predict(
        self,
        query: np.ndarray,
        history: np.ndarray,
        y: np.ndarray,
        leave_self_out: bool,
    ) -> np.ndarray:
        distance = np.mean((query[:, None, :] - history[None, :, :]) ** 2, axis=2)
        if leave_self_out and len(query) == len(history):
            np.fill_diagonal(distance, np.inf)
        count = min(self.neighbors, len(history) - int(leave_self_out))
        nearest = np.argpartition(distance, kth=count - 1, axis=1)[:, :count]
        local_distance = np.take_along_axis(distance, nearest, axis=1)
        # Adaptive bandwidth avoids one descriptor family setting the entire
        # scale and remains deterministic for duplicated condition vectors.
        bandwidth = np.maximum(np.median(local_distance, axis=1, keepdims=True), 1e-6)
        weights = np.exp(-local_distance / bandwidth)
        local_y = y[nearest]
        return (weights * local_y).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)

    def lift(
        self,
        candidates: pd.DataFrame,
        observed_conditions: pd.DataFrame,
        observed_y: Sequence[float],
        strength: float,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        y = np.asarray(observed_y, dtype=float)
        if len(y) < max(self.neighbors + 1, 6):
            return np.zeros(len(candidates)), {
                "confidence": 0.0,
                "kendall_tau": 0.0,
                "region_size": 0,
                "max_shift": 0.0,
            }
        history = self.featurizer.transform(observed_conditions)
        loo = self._predict(history, history, y, leave_self_out=True)
        tau_result = kendalltau(loo, y, nan_policy="omit")
        tau = float(tau_result.statistic) if np.isfinite(tau_result.statistic) else 0.0
        # Positive agreement is shrunk for small histories.  Negative
        # agreement is an explicit automatic-off signal.
        confidence = max(tau, 0.0) * np.sqrt(len(y) / (len(y) + 20.0))
        if confidence <= 0 or strength <= 0:
            return np.zeros(len(candidates)), {
                "confidence": 0.0,
                "kendall_tau": tau,
                "region_size": 0,
                "max_shift": 0.0,
            }
        candidate_features = self.featurizer.transform(candidates)
        local_prediction = self._predict(
            candidate_features, history, y, leave_self_out=False
        )
        spread = max(float(np.std(local_prediction, ddof=1)), 1.0)
        score = np.clip(
            (local_prediction - float(np.median(local_prediction))) / spread,
            -2.0,
            2.0,
        )
        outcome_scale = max(float(np.std(y, ddof=1)), 1.0)
        shift = (
            float(strength)
            * confidence
            * self.lift_scale
            * outcome_scale
            * score
        )
        return shift, {
            "confidence": float(confidence),
            "kendall_tau": tau,
            "region_size": int(np.sum(score > 0)),
            "max_shift": float(np.max(np.abs(shift))) if len(shift) else 0.0,
        }


class DFTXGBoostPreferenceGuide:
    """Cross-validated DFT-XGBoost mean lift for a trusted base surrogate."""

    def __init__(
        self,
        featurizer: DFTElectronicFeaturizer,
        seed: int,
        lift_scale: float = 0.25,
        bootstrap_members: int = 8,
    ):
        if lift_scale <= 0 or bootstrap_members < 4:
            raise ValueError("lift_scale must be positive and bootstrap_members at least 4")
        self.featurizer = featurizer
        self.seed = int(seed)
        self.lift_scale = float(lift_scale)
        self.bootstrap_members = int(bootstrap_members)

    @staticmethod
    def _model(seed: int):
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError(
                "DFTXGBoostPreferenceGuide requires the optional CPU XGBoost environment"
            ) from error
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )

    def lift(
        self,
        candidates: pd.DataFrame,
        observed_conditions: pd.DataFrame,
        observed_y: Sequence[float],
        strength: float,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        y = np.asarray(observed_y, dtype=float)
        if len(y) < 10 or strength <= 0:
            return np.zeros(len(candidates)), {
                "confidence": 0.0,
                "kendall_tau": 0.0,
                "region_size": 0,
                "max_shift": 0.0,
            }
        x = self.featurizer.transform(observed_conditions)
        folds = min(5, len(y))
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.seed + len(y),
        )
        oof = np.empty(len(y), dtype=float)
        for fold_index, (fit_indices, holdout_indices) in enumerate(splitter.split(x)):
            model = self._model(self.seed + 10_000 + fold_index)
            model.fit(x[fit_indices], y[fit_indices])
            oof[holdout_indices] = model.predict(x[holdout_indices])
        tau_result = kendalltau(oof, y, nan_policy="omit")
        tau = float(tau_result.statistic) if np.isfinite(tau_result.statistic) else 0.0
        confidence = max(tau, 0.0) * np.sqrt(len(y) / (len(y) + 20.0))
        if confidence <= 0:
            return np.zeros(len(candidates)), {
                "confidence": 0.0,
                "kendall_tau": tau,
                "region_size": 0,
                "max_shift": 0.0,
            }
        candidate_x = self.featurizer.transform(candidates)
        rng = np.random.default_rng(self.seed + 100_000 + len(y))
        predictions = []
        for _ in range(self.bootstrap_members):
            indices = rng.integers(0, len(y), size=len(y))
            model = self._model(int(rng.integers(0, 2**31 - 1)))
            model.fit(x[indices], y[indices])
            predictions.append(model.predict(candidate_x))
        local_prediction = np.asarray(predictions).mean(axis=0)
        spread = max(float(np.std(local_prediction, ddof=1)), 1.0)
        score = np.clip(
            (local_prediction - float(np.median(local_prediction))) / spread,
            -2.0,
            2.0,
        )
        outcome_scale = max(float(np.std(y, ddof=1)), 1.0)
        shift = (
            float(strength)
            * confidence
            * self.lift_scale
            * outcome_scale
            * score
        )
        return shift, {
            "confidence": float(confidence),
            "kendall_tau": tau,
            "region_size": int(np.sum(score > 0)),
            "max_shift": float(np.max(np.abs(shift))) if len(shift) else 0.0,
        }
