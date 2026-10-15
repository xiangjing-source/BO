"""Small-sample categorical Bayesian optimizer with a leakage-safe interface."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, DotProduct, Hyperparameter, Kernel, Product as KernelProduct,
    RBF, WhiteKernel,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


class PerVariableCategoricalKernel(Kernel):
    """One learned similarity parameter for each one-hot categorical variable.

    For a variable, equal categories have covariance 1 while unequal
    categories have covariance ``rho``.  Multiplying the per-variable kernels
    yields an interpretable categorical similarity without accessing labels
    beyond the observations supplied to the Gaussian process.
    """

    def __init__(
        self,
        start: int = 0,
        end: int = 1,
        rho: float = 0.3,
        rho_bounds: tuple[float, float] = (1e-4, 0.999),
    ):
        self.start = start
        self.end = end
        self.rho = rho
        self.rho_bounds = rho_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        matches = X[:, self.start:self.end] @ Y[:, self.start:self.end].T
        covariance = self.rho + (1.0 - self.rho) * matches
        if eval_gradient:
            # sklearn optimizes kernel theta in log space, so this is
            # dK/dlog(rho) = rho * dK/drho.
            return covariance, (
                self.rho * (1.0 - matches)
            )[:, :, np.newaxis]
        return covariance

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

    @property
    def hyperparameter_rho(self):
        return Hyperparameter("rho", "numeric", self.rho_bounds)

    def __repr__(self):
        return f"PerVariableCategoricalKernel(cols=[{self.start}:{self.end}], rho={self.rho:.3f})"


@dataclass(frozen=True)
class Recommendation:
    query_index: int
    predicted_yield: float
    uncertainty: float
    acquisition_value: float
    selection_mode: str
    preference_confidence: float = 0.0
    preference_shift: float = 0.0
    ensemble_weights: Mapping[str, float] | None = None


class CategoricalBayesOptimizer:
    """Gaussian-process BO over a public, categorical candidate pool.

    The object intentionally never accepts a labelled candidate table.  Labels
    are supplied one observation at a time by the caller after oracle.query().
    """

    def __init__(
        self,
        variable_columns: Sequence[str],
        categories: Dict[str, List[str]],
        seed: int,
        epsilon_start: float = 0.15,
        epsilon_end: float = 0.01,
        n_restarts_optimizer: int = 3,
        chemical_featurizer: Any | None = None,
        strategy: str = "expected_improvement",
        kernel: str = "dot",
        acquisition: str = "expected_improvement",
        ucb_beta: float = 1.0,
        surrogate: str = "gaussian_process",
        forest_min_samples_leaf: int = 2,
        forest_max_features: float | str = 1.0,
        ridge_alpha: float = 10.0,
        ensemble_size: int = 100,
        hybrid_forest_weight: float = 0.5,
        hybrid_ridge_weight: float = 0.5,
        feature_mode: str = "main_effects",
        diversity_weight: float = 0.0,
        knn_neighbors: int = 5,
        uncertainty_scale: float = 1.0,
        warmup_random_rounds: int = 0,
        greedy_rounds: int = 0,
        top_probability_rounds: int = 0,
        tail_probability_weight: float = 0.0,
        tail_quantile: float = 0.75,
        gp_bootstrap: bool = False,
        gp_alpha: float = 1e-6,
        gp_noise: float = 0.05,
        product_descriptor_only: bool = False,
        hybrid_forest_weight_end: float | None = None,
        product_transfer_weight: float = 0.5,
        product_transfer_weight_end: float | None = None,
        transfer_chemical_featurizer: Any | None = None,
        product_transfer_novelty_columns: Sequence[str] | None = None,
        include_categorical_identity: bool = True,
        adaptive_models: Sequence[str] | None = None,
        adaptive_temperature: float = 0.50,
        adaptive_forgetting: float = 0.85,
        preference_provider: Any | None = None,
        preference_strength: float = 1.0,
        svr_c: float = 10.0,
        svr_epsilon: float = 0.10,
        lasso_alpha: float = 0.50,
        boosting_estimators: int = 60,
        boosting_learning_rate: float = 0.05,
        boosting_max_depth: int = 2,
        xgboost_estimators: int = 100,
        xgboost_learning_rate: float = 0.05,
        xgboost_max_depth: int = 2,
        xgboost_subsample: float = 0.8,
        xgboost_colsample_bytree: float = 0.8,
        xgboost_reg_alpha: float = 0.0,
        xgboost_reg_lambda: float = 5.0,
        catboost_members: int = 12,
        catboost_iterations: int = 80,
        catboost_learning_rate: float = 0.05,
        catboost_depth: int = 3,
        catboost_l2_leaf_reg: float = 5.0,
        categorical_rho: float = 0.3,
        optimize_categorical_kernel: bool = True,
        categorical_optimize_once: bool = False,
        kernel_switch_rounds: int = 0,
        kernel_after_switch: str | None = None,
        categorical_kernel_columns: Sequence[str] | None = None,
    ):
        self.seed = int(seed)
        self.variable_columns = list(variable_columns)
        if product_descriptor_only and "Product" not in self.variable_columns:
            raise ValueError("product_descriptor_only requires a Product variable")
        self.encoding_columns = [
            column for column in self.variable_columns
            if not (product_descriptor_only and column == "Product")
        ]
        category_lists = [categories[column] for column in self.encoding_columns]
        self.encoder = OneHotEncoder(
            categories=category_lists,
            handle_unknown="error",
            sparse_output=False,
            dtype=np.float64,
        )
        self.encoder.fit(np.asarray([[values[0] for values in category_lists]], dtype=object))
        # A second, deliberately less specific representation is used only by
        # ``dual_forest_product_transfer``.  It removes the raw product name
        # but keeps the label-independent product descriptors supplied by the
        # chemistry featurizer.  The two views distinguish local adjustment
        # from cross-product transfer without querying an unseen yield.
        self.transfer_encoding_columns = [
            column for column in self.variable_columns if column != "Product"
        ]
        transfer_category_lists = [categories[column] for column in self.transfer_encoding_columns]
        self.transfer_encoder = OneHotEncoder(
            categories=transfer_category_lists,
            handle_unknown="error",
            sparse_output=False,
            dtype=np.float64,
        )
        self.transfer_encoder.fit(
            np.asarray([[values[0] for values in transfer_category_lists]], dtype=object)
        )
        self.rng = np.random.default_rng(seed)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.chemical_featurizer = chemical_featurizer
        if strategy not in {"expected_improvement", "random", "tpe", "thompson"}:
            raise ValueError("unsupported selection strategy")
        if kernel not in {
            "rbf",
            "dot",
            "categorical_product",
            "rbf_categorical_sum",
        }:
            raise ValueError("unsupported kernel")
        if kernel_after_switch not in {
            None,
            "rbf",
            "dot",
            "categorical_product",
        }:
            raise ValueError("unsupported post-switch kernel")
        if int(kernel_switch_rounds) < 0:
            raise ValueError("kernel_switch_rounds must be non-negative")
        if int(kernel_switch_rounds) > 0 and kernel_after_switch is None:
            raise ValueError(
                "kernel_after_switch is required when kernel_switch_rounds > 0"
            )
        if not 0.0 < float(categorical_rho) < 1.0:
            raise ValueError("categorical_rho must lie in (0, 1)")
        if categorical_kernel_columns is None:
            self.categorical_kernel_columns = tuple(self.encoding_columns)
        else:
            requested_kernel_columns = tuple(str(column) for column in categorical_kernel_columns)
            unknown_kernel_columns = sorted(
                set(requested_kernel_columns).difference(self.encoding_columns)
            )
            if unknown_kernel_columns:
                raise ValueError(
                    "categorical_kernel_columns contains unknown/non-identity columns: "
                    + ", ".join(unknown_kernel_columns)
                )
            if not requested_kernel_columns:
                raise ValueError("categorical_kernel_columns must not be empty")
            self.categorical_kernel_columns = requested_kernel_columns
        if acquisition not in {"expected_improvement", "upper_confidence_bound"}:
            raise ValueError("unsupported acquisition function")
        if float(ucb_beta) < 0:
            raise ValueError("ucb_beta must be non-negative")
        if surrogate not in {
            "gaussian_process",
            "random_forest",
            "extra_trees",
            "svr_ensemble",
            "lasso_ensemble",
            "gradient_boosting_ensemble",
            "xgboost_ensemble",
            "catboost_categorical_ensemble",
            "adaptive_ensemble",
            "dual_forest_product_transfer",
            "categorical_knn",
            "ridge_ensemble",
            "ridge_pair_ensemble",
            "hybrid_gp_forest",
            "hybrid_identity_gp_forest",
            "hybrid_gp_ridge",
        }:
            raise ValueError("unsupported surrogate")
        if int(forest_min_samples_leaf) < 1:
            raise ValueError("forest_min_samples_leaf must be positive")
        if isinstance(forest_max_features, str):
            if forest_max_features not in {"sqrt", "log2"}:
                raise ValueError("forest_max_features string must be 'sqrt' or 'log2'")
        elif not 0.0 < float(forest_max_features) <= 1.0:
            raise ValueError("forest_max_features must lie in (0, 1]")
        if float(ridge_alpha) <= 0 or int(ensemble_size) < 2:
            raise ValueError("ridge_alpha must be positive and ensemble_size at least 2")
        if int(knn_neighbors) < 1:
            raise ValueError("knn_neighbors must be positive")
        if float(uncertainty_scale) <= 0:
            raise ValueError("uncertainty_scale must be positive")
        if int(warmup_random_rounds) < 0:
            raise ValueError("warmup_random_rounds must be non-negative")
        if int(greedy_rounds) < 0:
            raise ValueError("greedy_rounds must be non-negative")
        if int(top_probability_rounds) < 0:
            raise ValueError("top_probability_rounds must be non-negative")
        opening_policies = sum(
            int(value) > 0
            for value in (
                warmup_random_rounds,
                greedy_rounds,
                top_probability_rounds,
            )
        )
        if opening_policies > 1:
            raise ValueError(
                "warmup_random_rounds, greedy_rounds and "
                "top_probability_rounds are mutually exclusive"
            )
        if float(tail_probability_weight) < 0:
            raise ValueError("tail_probability_weight must be non-negative")
        if not 0.5 <= float(tail_quantile) < 1.0:
            raise ValueError("tail_quantile must lie in [0.5, 1)")
        if float(gp_alpha) <= 0 or float(gp_noise) <= 0:
            raise ValueError("gp_alpha and gp_noise must be positive")
        if not 0.0 <= float(hybrid_forest_weight) <= 1.0:
            raise ValueError("hybrid_forest_weight must be between 0 and 1")
        if hybrid_forest_weight_end is not None and not 0.0 <= float(hybrid_forest_weight_end) <= 1.0:
            raise ValueError("hybrid_forest_weight_end must be between 0 and 1")
        if not 0.0 <= float(product_transfer_weight) <= 1.0:
            raise ValueError("product_transfer_weight must be between 0 and 1")
        if product_transfer_weight_end is not None and not 0.0 <= float(product_transfer_weight_end) <= 1.0:
            raise ValueError("product_transfer_weight_end must be between 0 and 1")
        if surrogate == "dual_forest_product_transfer" and (
            "Product" not in self.variable_columns or chemical_featurizer is None
        ):
            raise ValueError("dual_forest_product_transfer requires Product and chemistry descriptors")
        if not 0.0 <= float(hybrid_ridge_weight) <= 1.0:
            raise ValueError("hybrid_ridge_weight must be between 0 and 1")
        if feature_mode not in {"main_effects", "pairwise"}:
            raise ValueError("feature_mode must be 'main_effects' or 'pairwise'")
        if float(diversity_weight) < 0:
            raise ValueError("diversity_weight must be non-negative")
        allowed_adaptive = {
            "gaussian_process",
            "random_forest",
            "extra_trees",
            "svr",
            "lasso",
            "gradient_boosting",
            "xgboost",
        }
        selected_adaptive = tuple(adaptive_models or (
            "gaussian_process",
            "random_forest",
            "extra_trees",
            "svr",
            "lasso",
            "gradient_boosting",
        ))
        if not selected_adaptive or not set(selected_adaptive) <= allowed_adaptive:
            raise ValueError("adaptive_models contains an unsupported model")
        if float(adaptive_temperature) <= 0:
            raise ValueError("adaptive_temperature must be positive")
        if not 0.0 <= float(adaptive_forgetting) < 1.0:
            raise ValueError("adaptive_forgetting must lie in [0, 1)")
        if float(preference_strength) < 0:
            raise ValueError("preference_strength must be non-negative")
        if float(svr_c) <= 0 or float(svr_epsilon) < 0:
            raise ValueError("SVR C must be positive and epsilon non-negative")
        if float(lasso_alpha) <= 0:
            raise ValueError("lasso_alpha must be positive")
        if int(boosting_estimators) < 10 or float(boosting_learning_rate) <= 0:
            raise ValueError("invalid gradient-boosting size or learning rate")
        if int(boosting_max_depth) < 1:
            raise ValueError("boosting_max_depth must be positive")
        if int(xgboost_estimators) < 10 or float(xgboost_learning_rate) <= 0:
            raise ValueError("invalid XGBoost size or learning rate")
        if int(xgboost_max_depth) < 1:
            raise ValueError("xgboost_max_depth must be positive")
        if not 0 < float(xgboost_subsample) <= 1:
            raise ValueError("xgboost_subsample must lie in (0, 1]")
        if not 0 < float(xgboost_colsample_bytree) <= 1:
            raise ValueError("xgboost_colsample_bytree must lie in (0, 1]")
        if float(xgboost_reg_alpha) < 0 or float(xgboost_reg_lambda) < 0:
            raise ValueError("XGBoost regularization must be non-negative")
        if int(catboost_members) < 2 or int(catboost_iterations) < 10:
            raise ValueError("CatBoost needs at least two members and ten iterations")
        if float(catboost_learning_rate) <= 0 or int(catboost_depth) < 1:
            raise ValueError("invalid CatBoost learning rate or depth")
        if float(catboost_l2_leaf_reg) < 0:
            raise ValueError("CatBoost L2 regularization must be non-negative")
        if surrogate == "catboost_categorical_ensemble" and not include_categorical_identity:
            raise ValueError("native categorical CatBoost requires categorical identity")
        self.strategy = strategy
        self.kernel = kernel
        self._active_kernel = kernel
        self.kernel_after_switch = kernel_after_switch
        self.kernel_switch_rounds = int(kernel_switch_rounds)
        self.categorical_rho = float(categorical_rho)
        self.optimize_categorical_kernel = bool(optimize_categorical_kernel)
        self.categorical_optimize_once = bool(categorical_optimize_once)
        self._cached_categorical_kernel: Kernel | None = None
        self.acquisition = acquisition
        self.ucb_beta = float(ucb_beta)
        self.surrogate = surrogate
        self.forest_min_samples_leaf = int(forest_min_samples_leaf)
        self.forest_max_features = forest_max_features
        self.ridge_alpha = float(ridge_alpha)
        self.ensemble_size = int(ensemble_size)
        self.hybrid_forest_weight = float(hybrid_forest_weight)
        self.hybrid_forest_weight_end = (
            None if hybrid_forest_weight_end is None else float(hybrid_forest_weight_end)
        )
        self._active_hybrid_forest_weight = self.hybrid_forest_weight
        self.hybrid_ridge_weight = float(hybrid_ridge_weight)
        self.feature_mode = feature_mode
        # This is a soft, label-free exploration bonus.  It uses only the
        # identities of conditions that have already been observed, never a
        # candidate yield.  A value of zero recovers ordinary EI/UCB.
        self.diversity_weight = float(diversity_weight)
        self.knn_neighbors = int(knn_neighbors)
        self.uncertainty_scale = float(uncertainty_scale)
        self.warmup_random_rounds = int(warmup_random_rounds)
        # Greedy opening rounds choose the highest posterior mean before the
        # ordinary epsilon/EI policy starts.  This is a train-only-selectable
        # way to target first-round yield and early best-so-far AUC without
        # changing the surrogate or reading an unreleased candidate outcome.
        self.greedy_rounds = int(greedy_rounds)
        # A train-label-only classifier estimates whether a condition belongs
        # to the observed high-yield tail.  Restricting it to the opening
        # rounds separates cold-start ranking from the later BO policy.
        self.top_probability_rounds = int(top_probability_rounds)
        self.tail_probability_weight = float(tail_probability_weight)
        self.tail_quantile = float(tail_quantile)
        # The original GP route used bootstrap resampling to represent the
        # instability caused by very few observed reactions.  It is exposed
        # here as a train-only-selectable option rather than silently relying
        # on a historical formal score.
        self.gp_bootstrap = bool(gp_bootstrap)
        self.gp_alpha = float(gp_alpha)
        self.gp_noise = float(gp_noise)
        self.product_descriptor_only = bool(product_descriptor_only)
        self.product_transfer_weight = float(product_transfer_weight)
        self.product_transfer_weight_end = (
            None
            if product_transfer_weight_end is None
            else float(product_transfer_weight_end)
        )
        self._active_product_transfer_weight = self.product_transfer_weight
        self.transfer_chemical_featurizer = transfer_chemical_featurizer
        self.product_transfer_novelty_columns = tuple(
            str(column) for column in (product_transfer_novelty_columns or ())
        )
        unknown_novelty_columns = sorted(
            set(self.product_transfer_novelty_columns).difference(self.variable_columns)
        )
        if unknown_novelty_columns:
            raise ValueError(
                "product_transfer_novelty_columns contains unknown variables: "
                + ", ".join(unknown_novelty_columns)
            )
        self.include_categorical_identity = bool(include_categorical_identity)
        self.adaptive_models = selected_adaptive
        self.adaptive_temperature = float(adaptive_temperature)
        self.adaptive_forgetting = float(adaptive_forgetting)
        self.preference_provider = preference_provider
        self.preference_strength = float(preference_strength)
        self.svr_c = float(svr_c)
        self.svr_epsilon = float(svr_epsilon)
        self.lasso_alpha = float(lasso_alpha)
        self.boosting_estimators = int(boosting_estimators)
        self.boosting_learning_rate = float(boosting_learning_rate)
        self.boosting_max_depth = int(boosting_max_depth)
        self.xgboost_estimators = int(xgboost_estimators)
        self.xgboost_learning_rate = float(xgboost_learning_rate)
        self.xgboost_max_depth = int(xgboost_max_depth)
        self.xgboost_subsample = float(xgboost_subsample)
        self.xgboost_colsample_bytree = float(xgboost_colsample_bytree)
        self.xgboost_reg_alpha = float(xgboost_reg_alpha)
        self.xgboost_reg_lambda = float(xgboost_reg_lambda)
        self.catboost_members = int(catboost_members)
        self.catboost_iterations = int(catboost_iterations)
        self.catboost_learning_rate = float(catboost_learning_rate)
        self.catboost_depth = int(catboost_depth)
        self.catboost_l2_leaf_reg = float(catboost_l2_leaf_reg)
        self._train_x: list[np.ndarray] = []
        self._train_y: list[float] = []
        self._train_conditions: list[dict[str, str]] = []
        self._model: Any | None = None
        self._scaler: StandardScaler | None = None
        self._adaptive_losses: dict[str, float] | None = None
        self._adaptive_residual_scales: dict[str, float] = {}
        self._last_adaptive_components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_adaptive_weights: dict[str, float] = {}
        self._pending_adaptive_predictions: dict[str, float] | None = None

    def _encode_with(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        encoder: OneHotEncoder,
        chemical_featurizer: Any | None,
    ) -> np.ndarray:
        values = frame.loc[:, columns].astype(str).to_numpy(dtype=object)
        categorical = encoder.transform(values)
        if self.feature_mode == "pairwise" and self.include_categorical_identity:
            boundaries = []
            start = 0
            for levels in encoder.categories_:
                end = start + len(levels)
                boundaries.append((start, end))
                start = end
            interactions = []
            for left in range(len(boundaries)):
                for right in range(left + 1, len(boundaries)):
                    a0, a1 = boundaries[left]
                    b0, b1 = boundaries[right]
                    interactions.append(
                        (categorical[:, a0:a1, None] * categorical[:, None, b0:b1]).reshape(len(frame), -1)
                    )
            categorical = np.hstack([categorical, *interactions])
        if not self.include_categorical_identity:
            categorical = np.empty((len(frame), 0), dtype=np.float64)
        if chemical_featurizer is None:
            if not self.include_categorical_identity:
                raise ValueError("No input features: enable categorical identity or a featurizer")
            return categorical
        chemistry = chemical_featurizer.transform(frame)
        return np.hstack([categorical, chemistry])

    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        return self._encode_with(
            frame,
            self.encoding_columns,
            self.encoder,
            self.chemical_featurizer,
        )

    def _encode_identity(self, frame: pd.DataFrame) -> np.ndarray:
        """One-hot condition identity without any chemistry descriptor block."""
        values = frame.loc[:, self.encoding_columns].astype(str).to_numpy(dtype=object)
        return self.encoder.transform(values)

    def _encode_transfer(self, frame: pd.DataFrame) -> np.ndarray:
        return self._encode_with(
            frame,
            self.transfer_encoding_columns,
            self.transfer_encoder,
            self.transfer_chemical_featurizer or self.chemical_featurizer,
        )

    def _training_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._train_conditions, columns=self.variable_columns)

    def _catboost_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, self.variable_columns].astype(str).reset_index(drop=True)
        if self.chemical_featurizer is not None:
            chemistry = np.asarray(
                self.chemical_featurizer.transform(frame),
                dtype=np.float64,
            )
            chemistry_frame = pd.DataFrame(
                chemistry,
                columns=[f"aux_{index}" for index in range(chemistry.shape[1])],
            )
            result = pd.concat([result, chemistry_frame], axis=1)
        return result

    def add_initial_training(self, frame: pd.DataFrame) -> None:
        if "Yield" not in frame.columns:
            raise ValueError("Initial training frame must include Yield")
        self._train_x.extend(self._encode(frame))
        self._train_y.extend(frame["Yield"].astype(float).tolist())
        self._train_conditions.extend(
            {column: str(row[column]) for column in self.variable_columns}
            for _, row in frame.iterrows()
        )
        self._model = None

    def add_feedback(self, condition: pd.DataFrame, observed_yield: float) -> None:
        if self._pending_adaptive_predictions and self._adaptive_losses is not None:
            scale = max(float(np.std(self._train_y, ddof=1)), 1.0)
            for name, prediction in self._pending_adaptive_predictions.items():
                error = min(((float(observed_yield) - prediction) / scale) ** 2, 25.0)
                self._adaptive_losses[name] = (
                    self.adaptive_forgetting * self._adaptive_losses[name]
                    + (1.0 - self.adaptive_forgetting) * error
                )
        self._pending_adaptive_predictions = None
        self._train_x.append(self._encode(condition)[0])
        self._train_y.append(float(observed_yield))
        row = condition.iloc[0]
        self._train_conditions.append({column: str(row[column]) for column in self.variable_columns})
        self._model = None

    def _fit_adaptive_component(
        self,
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        seed: int,
        compact: bool = False,
    ) -> Any:
        """Fit one independently useful small-data regression component."""
        tree_count = max(24, min(self.ensemble_size, 48 if compact else self.ensemble_size))
        bootstrap_count = 4 if compact else max(8, min(16, self.ensemble_size // 6))
        rng = np.random.default_rng(seed)
        if name == "random_forest":
            return RandomForestRegressor(
                n_estimators=tree_count,
                min_samples_leaf=self.forest_min_samples_leaf,
                max_features=self.forest_max_features,
                n_jobs=1,
                random_state=seed,
            ).fit(x_train, y_train)
        if name == "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=tree_count,
                min_samples_leaf=self.forest_min_samples_leaf,
                max_features=1.0,
                n_jobs=1,
                random_state=seed,
            ).fit(x_train, y_train)
        if name == "gaussian_process":
            kernel = (
                ConstantKernel(1.0, (1e-2, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                + WhiteKernel(noise_level=self.gp_noise, noise_level_bounds=(1e-5, 1e2))
            )
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.gp_alpha,
                normalize_y=True,
                random_state=seed,
                n_restarts_optimizer=0 if compact else min(self.n_restarts_optimizer, 1),
            ).fit(x_train, y_train)

        members = []
        for member_index in range(bootstrap_count):
            indices = rng.integers(0, len(y_train), size=len(y_train))
            member_seed = int(rng.integers(0, 2**31 - 1))
            if name == "svr":
                model = make_pipeline(
                    StandardScaler(),
                    SVR(C=self.svr_c, epsilon=self.svr_epsilon, gamma="scale", kernel="rbf"),
                )
            elif name == "lasso":
                model = make_pipeline(
                    StandardScaler(),
                    Lasso(alpha=self.lasso_alpha, max_iter=10_000, random_state=member_seed),
                )
            elif name == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=self.boosting_estimators,
                    learning_rate=self.boosting_learning_rate,
                    max_depth=self.boosting_max_depth,
                    min_samples_leaf=self.forest_min_samples_leaf,
                    loss="huber",
                    random_state=member_seed,
                )
            elif name == "xgboost":
                try:
                    from xgboost import XGBRegressor
                except ImportError as error:
                    raise RuntimeError(
                        "xgboost_ensemble was selected but the optional xgboost "
                        "package is not installed"
                    ) from error
                model = XGBRegressor(
                    n_estimators=self.xgboost_estimators,
                    learning_rate=self.xgboost_learning_rate,
                    max_depth=self.xgboost_max_depth,
                    min_child_weight=max(1, self.forest_min_samples_leaf),
                    subsample=self.xgboost_subsample,
                    colsample_bytree=self.xgboost_colsample_bytree,
                    reg_alpha=self.xgboost_reg_alpha,
                    reg_lambda=self.xgboost_reg_lambda,
                    objective="reg:squarederror",
                    tree_method="hist",
                    device="cpu",
                    n_jobs=1,
                    random_state=member_seed,
                    verbosity=0,
                )
            else:
                raise ValueError(f"Unsupported adaptive component: {name}")
            members.append(model.fit(x_train[indices], y_train[indices]))
        return members

    @staticmethod
    def _predict_adaptive_component(model: Any, name: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if name in {"random_forest", "extra_trees"}:
            values = np.asarray([tree.predict(x) for tree in model.estimators_])
            return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8)
        if name == "gaussian_process":
            mean, std = model.predict(x, return_std=True)
            return np.asarray(mean), np.maximum(np.asarray(std), 1e-8)
        values = np.asarray([member.predict(x) for member in model])
        return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8)

    def _initialize_adaptive_validation(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Set model weights from observed-label cross-validation only."""
        folds = min(5, len(y_train))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=self.seed)
        predictions = {name: np.empty(len(y_train), dtype=float) for name in self.adaptive_models}
        for fold_index, (fit_indices, holdout_indices) in enumerate(splitter.split(x_train)):
            for model_index, name in enumerate(self.adaptive_models):
                seed = self.seed + 10_000 * (fold_index + 1) + 101 * (model_index + 1)
                model = self._fit_adaptive_component(
                    name, x_train[fit_indices], y_train[fit_indices], seed, compact=True
                )
                mean, _ = self._predict_adaptive_component(model, name, x_train[holdout_indices])
                predictions[name][holdout_indices] = mean
        variance = max(float(np.var(y_train, ddof=1)), 1.0)
        self._adaptive_losses = {}
        self._adaptive_residual_scales = {}
        for name, predicted in predictions.items():
            residual = y_train - predicted
            rmse = float(np.sqrt(np.mean(residual**2)))
            self._adaptive_residual_scales[name] = max(rmse, 1.0)
            self._adaptive_losses[name] = min(float(np.mean(residual**2) / variance), 25.0)

    def _adaptive_weights(self) -> dict[str, float]:
        assert self._adaptive_losses is not None
        loss = np.asarray([self._adaptive_losses[name] for name in self.adaptive_models], dtype=float)
        logits = -(loss - np.min(loss)) / self.adaptive_temperature
        raw = np.exp(np.clip(logits, -50.0, 0.0))
        raw = raw / raw.sum()
        # A small floor preserves model diversity when a tiny validation set
        # happens to favor one family by chance.
        floor = min(0.03, 0.5 / len(raw))
        raw = (1.0 - floor * len(raw)) * raw + floor
        raw = raw / raw.sum()
        return {name: float(weight) for name, weight in zip(self.adaptive_models, raw)}

    def _make_gp_kernel(self) -> Kernel:
        if (
            self._active_kernel == "categorical_product"
            and self.categorical_optimize_once
            and self._cached_categorical_kernel is not None
        ):
            return clone(self._cached_categorical_kernel)
        if self._active_kernel == "rbf":
            return (
                ConstantKernel(1.0, (1e-2, 1e3))
                * RBF(1.0, (1e-3, 1e3))
                + WhiteKernel(self.gp_noise, (1e-5, 1e2))
            )
        if self._active_kernel == "dot":
            return (
                ConstantKernel(1.0, (1e-2, 1e3))
                + DotProduct(1.0, (1e-3, 1e3))
                + WhiteKernel(self.gp_noise, (1e-5, 1e2))
            )
        boundaries = []
        start = 0
        selected_columns = set(self.categorical_kernel_columns)
        for column, values in zip(self.encoding_columns, self.encoder.categories_):
            end = start + len(values)
            if column in selected_columns:
                boundaries.append((start, end))
            start = end
        categorical: Kernel = PerVariableCategoricalKernel(
            *boundaries[0], rho=self.categorical_rho
        )
        for left, right in boundaries[1:]:
            categorical *= PerVariableCategoricalKernel(
                left, right, rho=self.categorical_rho
            )
        if self._active_kernel == "rbf_categorical_sum":
            # Additive covariance lets the GP learn from two complementary,
            # train-label-only notions of similarity: smooth one-hot distance
            # and exact/partial categorical matches. Independent amplitudes
            # are fitted from observations released so far in the loop.
            return (
                ConstantKernel(1.0, (1e-2, 1e3))
                * RBF(1.0, (1e-3, 1e3))
                + ConstantKernel(1.0, (1e-2, 1e3)) * categorical
                + WhiteKernel(self.gp_noise, (1e-5, 1e2))
            )
        return (
            ConstantKernel(1.0, (1e-2, 1e3)) * categorical
            + WhiteKernel(self.gp_noise, (1e-5, 1e2))
        )

    def _gp_optimizer(self) -> str | None:
        """Return the sklearn optimizer for the currently active GP kernel."""
        if (
            self._active_kernel == "categorical_product"
            and (
                not self.optimize_categorical_kernel
                or (
                    self.categorical_optimize_once
                    and self._cached_categorical_kernel is not None
                )
            )
        ):
            return None
        return "fmin_l_bfgs_b"

    def _fit(self) -> None:
        if self._model is not None:
            return
        if len(self._train_y) < 2:
            raise RuntimeError("At least two observations are required for GP fitting")
        x_train = np.asarray(self._train_x, dtype=np.float64)
        y_train = np.asarray(self._train_y, dtype=np.float64)

        def gp_fit_data(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if not self.gp_bootstrap or len(target) < 3:
                return x_train, target
            indices = self.rng.integers(0, len(target), size=len(target))
            return x_train[indices], target[indices]
        component_surrogates = {
            "extra_trees": "extra_trees",
            "svr_ensemble": "svr",
            "lasso_ensemble": "lasso",
            "gradient_boosting_ensemble": "gradient_boosting",
            "xgboost_ensemble": "xgboost",
        }
        if self.surrogate in component_surrogates:
            name = component_surrogates[self.surrogate]
            self._scaler = None
            self._model = self._fit_adaptive_component(
                name,
                x_train,
                y_train,
                int(self.rng.integers(0, 2**31 - 1)),
            )
            return
        if self.surrogate == "adaptive_ensemble":
            self._scaler = None
            if self._adaptive_losses is None:
                self._initialize_adaptive_validation(x_train, y_train)
            self._model = {
                name: self._fit_adaptive_component(
                    name,
                    x_train,
                    y_train,
                    int(self.rng.integers(0, 2**31 - 1)),
                )
                for name in self.adaptive_models
            }
            return
        if self.surrogate == "catboost_categorical_ensemble":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError(
                    "catboost_categorical_ensemble requires the optional "
                    "catboost package"
                ) from exc
            self._scaler = None
            training = self._catboost_frame(self._training_frame())
            self._model = [
                CatBoostRegressor(
                    iterations=self.catboost_iterations,
                    learning_rate=self.catboost_learning_rate,
                    depth=self.catboost_depth,
                    l2_leaf_reg=self.catboost_l2_leaf_reg,
                    loss_function="RMSE",
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                    random_strength=1.0,
                    random_seed=self.seed + 810_000 + member,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                ).fit(
                    training,
                    y_train,
                    cat_features=self.variable_columns,
                )
                for member in range(self.catboost_members)
            ]
            return
        if self.surrogate == "random_forest":
            self._scaler = None
            self._model = RandomForestRegressor(
                # 100 trees is enough for a stable empirical uncertainty on
                # these tens-of-observations problems, while keeping a 40-step
                # closed loop practical on CPU.
                n_estimators=self.ensemble_size,
                min_samples_leaf=self.forest_min_samples_leaf,
                max_features=self.forest_max_features,
                n_jobs=1,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
            ).fit(x_train, y_train)
            return
        if self.surrogate == "dual_forest_product_transfer":
            # Both forests receive exactly the same observed yields.  The
            # transfer forest sees no raw Product identity, only its chemical
            # descriptor vector, which prevents it from memorising a product
            # label and makes it a genuine cross-product component.
            x_transfer = self._encode_transfer(self._training_frame())
            self._scaler = None
            self._model = {
                "local": RandomForestRegressor(
                    n_estimators=self.ensemble_size,
                    min_samples_leaf=self.forest_min_samples_leaf,
                    max_features=self.forest_max_features,
                    n_jobs=1,
                    random_state=int(self.rng.integers(0, 2**31 - 1)),
                ).fit(x_train, y_train),
                "transfer": RandomForestRegressor(
                    n_estimators=self.ensemble_size,
                    min_samples_leaf=self.forest_min_samples_leaf,
                    max_features=self.forest_max_features,
                    n_jobs=1,
                    random_state=int(self.rng.integers(0, 2**31 - 1)),
                ).fit(x_transfer, y_train),
            }
            return
        if self.surrogate == "categorical_knn":
            # No outcome-driven feature learning occurs here: a condition is
            # similar when it matches on more discrete experimental factors.
            # The observed neighbour spread supplies a conservative local
            # uncertainty for acquisition.  This is particularly useful for
            # Suzuki's compact, repeated-factor training table.
            self._scaler = None
            self._model = {
                "conditions": np.asarray(
                    [[row[column] for column in self.variable_columns] for row in self._train_conditions],
                    dtype=object,
                ),
                "y": y_train,
            }
            return
        if self.surrogate in {"ridge_ensemble", "ridge_pair_ensemble"}:
            # Bootstrap members quantify data scarcity.  The individual ridge
            # models stay deliberately simple and are selected using labels
            # from train.csv only.
            self._scaler = None
            self._model = []
            for _ in range(self.ensemble_size):
                indices = self.rng.integers(0, len(y_train), size=len(y_train))
                self._model.append(Ridge(alpha=self.ridge_alpha).fit(x_train[indices], y_train[indices]))
            return
        if self.surrogate in {"hybrid_gp_forest", "hybrid_identity_gp_forest"}:
            self._scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            y_scaled = self._scaler.transform(y_train.reshape(-1, 1)).ravel()
            kernel = self._make_gp_kernel()
            if self.surrogate == "hybrid_identity_gp_forest":
                gp_x_full = self._encode_identity(self._training_frame())
                if self.gp_bootstrap and len(y_scaled) >= 3:
                    indices = self.rng.integers(0, len(y_scaled), size=len(y_scaled))
                    gp_x, gp_y = gp_x_full[indices], y_scaled[indices]
                else:
                    gp_x, gp_y = gp_x_full, y_scaled
            else:
                gp_x, gp_y = gp_fit_data(y_scaled)
            forest = RandomForestRegressor(
                    n_estimators=self.ensemble_size,
                    min_samples_leaf=self.forest_min_samples_leaf,
                    max_features=self.forest_max_features,
                    n_jobs=1,
                    random_state=int(self.rng.integers(0, 2**31 - 1)),
                ).fit(x_train, y_train)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.gp_alpha,
                normalize_y=False,
                optimizer=self._gp_optimizer(),
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                n_restarts_optimizer=self.n_restarts_optimizer,
            ).fit(gp_x, gp_y)
            if (
                self._active_kernel == "categorical_product"
                and self.categorical_optimize_once
                and self._cached_categorical_kernel is None
            ):
                self._cached_categorical_kernel = clone(gp.kernel_)
            self._model = {"forest": forest, "gp": gp}
            return
        if self.surrogate == "hybrid_gp_ridge":
            self._scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            y_scaled = self._scaler.transform(y_train.reshape(-1, 1)).ravel()
            kernel = self._make_gp_kernel()
            gp_x, gp_y = gp_fit_data(y_scaled)
            self._model = {
                "ridge": Ridge(alpha=self.ridge_alpha).fit(x_train, y_train),
                "gp": GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.gp_alpha,
                    normalize_y=False,
                    optimizer=self._gp_optimizer(),
                    random_state=int(self.rng.integers(0, 2**31 - 1)),
                    n_restarts_optimizer=self.n_restarts_optimizer,
                ).fit(gp_x, gp_y),
            }
            return
        self._scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_scaled = self._scaler.transform(y_train.reshape(-1, 1)).ravel()
        # The RBF alternative treats one-hot distance as categorical
        # similarity.  Both kernels are retained for train-label-only model
        # selection; the default dot-product kernel is the current sequential
        # validation choice rather than a claim about test-label performance.
        kernel = self._make_gp_kernel()
        gp_x, gp_y = gp_fit_data(y_scaled)
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_alpha,
            normalize_y=False,
            optimizer=self._gp_optimizer(),
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            n_restarts_optimizer=self.n_restarts_optimizer,
        ).fit(gp_x, gp_y)

    def predict(self, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._fit()
        assert self._model is not None
        component_surrogates = {
            "extra_trees": "extra_trees",
            "svr_ensemble": "svr",
            "lasso_ensemble": "lasso",
            "gradient_boosting_ensemble": "gradient_boosting",
            "xgboost_ensemble": "xgboost",
        }
        if self.surrogate in component_surrogates:
            name = component_surrogates[self.surrogate]
            return self._predict_adaptive_component(self._model, name, self._encode(candidates))
        if self.surrogate == "catboost_categorical_ensemble":
            encoded = self._catboost_frame(candidates)
            values = np.asarray(
                [model.predict(encoded) for model in self._model],
                dtype=float,
            )
            return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8)
        if self.surrogate == "adaptive_ensemble":
            encoded = self._encode(candidates)
            weights = self._adaptive_weights()
            components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for name in self.adaptive_models:
                mean, std = self._predict_adaptive_component(self._model[name], name, encoded)
                residual_floor = 0.25 * self._adaptive_residual_scales.get(name, 1.0)
                components[name] = (mean, np.maximum(std, residual_floor))
            mixture_mean = sum(weights[name] * components[name][0] for name in self.adaptive_models)
            second_moment = sum(
                weights[name] * (components[name][1] ** 2 + components[name][0] ** 2)
                for name in self.adaptive_models
            )
            mixture_std = np.sqrt(np.maximum(second_moment - mixture_mean**2, 1e-8))
            self._last_adaptive_components = components
            self._last_adaptive_weights = weights
            return np.asarray(mixture_mean), np.asarray(mixture_std)
        if self.surrogate == "random_forest":
            # Encode the public candidate pool once per decision.  Repeating
            # name-based chemistry featurisation for every tree made the
            # ensemble needlessly slow without changing its predictions.
            encoded = self._encode(candidates)
            values = np.asarray([tree.predict(encoded) for tree in self._model.estimators_])
            return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8)
        if self.surrogate == "dual_forest_product_transfer":
            encoded = self._encode(candidates)
            transfer_encoded = self._encode_transfer(candidates)
            local_values = np.asarray([tree.predict(encoded) for tree in self._model["local"].estimators_])
            transfer_values = np.asarray(
                [tree.predict(transfer_encoded) for tree in self._model["transfer"].estimators_]
            )
            local_mean, local_std = local_values.mean(axis=0), local_values.std(axis=0)
            transfer_mean, transfer_std = transfer_values.mean(axis=0), transfer_values.std(axis=0)
            if self.product_transfer_novelty_columns:
                observed = self._training_frame()
                novelty = np.zeros(len(candidates), dtype=float)
                for column in self.product_transfer_novelty_columns:
                    seen = set(observed[column].astype(str))
                    novelty = np.maximum(
                        novelty,
                        ~candidates[column].astype(str).isin(seen).to_numpy(),
                    )
                weight = self._active_product_transfer_weight * novelty
            else:
                weight = self._active_product_transfer_weight
            mean = (1.0 - weight) * local_mean + weight * transfer_mean
            variance = (
                (1.0 - weight) * local_std**2
                + weight * transfer_std**2
                + weight * (1.0 - weight) * (local_mean - transfer_mean) ** 2
            )
            return mean, np.maximum(np.sqrt(variance), 1e-8)
        if self.surrogate == "categorical_knn":
            conditions = candidates.loc[:, self.variable_columns].astype(str).to_numpy(dtype=object)
            history = self._model["conditions"]
            y = self._model["y"]
            distance = (conditions[:, None, :] != history[None, :, :]).mean(axis=2)
            count = min(self.knn_neighbors, len(y))
            nearest = np.argpartition(distance, kth=count - 1, axis=1)[:, :count]
            near_distance = np.take_along_axis(distance, nearest, axis=1)
            near_y = y[nearest]
            weights = 1.0 / (near_distance + 0.15)
            mean = (weights * near_y).sum(axis=1) / weights.sum(axis=1)
            weighted_variance = (weights * (near_y - mean[:, None]) ** 2).sum(axis=1) / weights.sum(axis=1)
            # A small floor prevents EI from becoming spuriously deterministic
            # when a candidate happens to have near-identical neighbours.
            return mean, np.maximum(np.sqrt(weighted_variance), 1.0)
        if self.surrogate in {"ridge_ensemble", "ridge_pair_ensemble"}:
            encoded = self._encode(candidates)
            values = np.asarray([model.predict(encoded) for model in self._model])
            return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8)
        if self.surrogate in {"hybrid_gp_forest", "hybrid_identity_gp_forest"}:
            assert self._scaler is not None
            encoded = self._encode(candidates)
            forest_values = np.asarray([tree.predict(encoded) for tree in self._model["forest"].estimators_])
            forest_mean, forest_std = forest_values.mean(axis=0), forest_values.std(axis=0)
            gp_encoded = (
                self._encode_identity(candidates)
                if self.surrogate == "hybrid_identity_gp_forest"
                else encoded
            )
            gp_mean, gp_std = self._model["gp"].predict(gp_encoded, return_std=True)
            gp_mean = self._scaler.inverse_transform(gp_mean.reshape(-1, 1)).ravel()
            gp_std = gp_std * float(self._scaler.scale_[0])
            weight = self._active_hybrid_forest_weight
            mean = weight * forest_mean + (1.0 - weight) * gp_mean
            variance = weight * forest_std**2 + (1.0 - weight) * gp_std**2 + weight * (1.0 - weight) * (forest_mean - gp_mean)**2
            return mean, np.maximum(np.sqrt(variance), 1e-8)
        if self.surrogate == "hybrid_gp_ridge":
            assert self._scaler is not None
            encoded = self._encode(candidates)
            ridge_mean = self._model["ridge"].predict(encoded)
            gp_mean, gp_std = self._model["gp"].predict(encoded, return_std=True)
            gp_mean = self._scaler.inverse_transform(gp_mean.reshape(-1, 1)).ravel()
            gp_std = gp_std * float(self._scaler.scale_[0])
            weight = self.hybrid_ridge_weight
            mean = weight * ridge_mean + (1.0 - weight) * gp_mean
            # Ridge disagreement adds epistemic uncertainty without letting a
            # high-variance bootstrap dominate the acquisition function.
            variance = gp_std**2 + weight * (1.0 - weight) * (ridge_mean - gp_mean)**2
            return mean, np.maximum(np.sqrt(variance), 1e-8)
        assert self._scaler is not None
        mean, std = self._model.predict(self._encode(candidates), return_std=True)
        mean = self._scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
        std = std * float(self._scaler.scale_[0])
        return mean, np.maximum(std, 1e-8)

    def _epsilon(self, round_index: int, total_rounds: int) -> float:
        if total_rounds <= 1:
            return self.epsilon_end
        fraction = min(max(round_index, 0), total_rounds - 1) / (total_rounds - 1)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _tpe_score(self, candidates: pd.DataFrame) -> np.ndarray:
        """Categorical TPE ratio with transparent main-effect and pair terms.

        The top quartile of observed yields is the ``good`` density and the
        remainder is the ``bad`` density.  Laplace smoothing makes unseen
        combinations exploratory rather than impossible.  This uses only
        labels already observed in the current trajectory.
        """
        y = np.asarray(self._train_y, dtype=float)
        good_count = max(1, int(math.ceil(0.25 * len(y))))
        order = np.argsort(y)[::-1]
        good_indices = set(int(index) for index in order[:good_count])
        good = [self._train_conditions[index] for index in range(len(y)) if index in good_indices]
        bad = [self._train_conditions[index] for index in range(len(y)) if index not in good_indices]
        if not bad:
            bad = good
        alpha = 0.5
        main_good: dict[tuple[str, str], int] = {}
        main_bad: dict[tuple[str, str], int] = {}
        pair_good: dict[tuple[str, str, str, str], int] = {}
        pair_bad: dict[tuple[str, str, str, str], int] = {}
        pairs = list(combinations(self.variable_columns, 2))
        for rows, main, pair in ((good, main_good, pair_good), (bad, main_bad, pair_bad)):
            for row in rows:
                for column in self.variable_columns:
                    key = (column, row[column])
                    main[key] = main.get(key, 0) + 1
                for left, right in pairs:
                    key = (left, row[left], right, row[right])
                    pair[key] = pair.get(key, 0) + 1
        values = []
        for _, row in candidates.iterrows():
            score = 0.0
            for column in self.variable_columns:
                levels = len(self.encoder.categories_[self.variable_columns.index(column)])
                key = (column, str(row[column]))
                score += math.log((main_good.get(key, 0) + alpha) / (len(good) + alpha * levels))
                score -= math.log((main_bad.get(key, 0) + alpha) / (len(bad) + alpha * levels))
            # Pair effects help distinguish, for example, ligand/base and
            # electrophile/nucleophile compatibility while retaining strong
            # smoothing in this very small-data regime.
            for left, right in pairs:
                levels = len(self.encoder.categories_[self.variable_columns.index(left)]) * len(self.encoder.categories_[self.variable_columns.index(right)])
                key = (left, str(row[left]), right, str(row[right]))
                score += 0.35 * math.log((pair_good.get(key, 0) + alpha) / (len(good) + alpha * levels))
                score -= 0.35 * math.log((pair_bad.get(key, 0) + alpha) / (len(bad) + alpha * levels))
            values.append(score)
        return np.asarray(values, dtype=float)

    def _nearest_history_diversity(self, candidates: pd.DataFrame) -> np.ndarray:
        """Return categorical distance to the closest observed condition.

        One means that a candidate differs in every optimised variable from
        every completed experiment; zero means its exact condition is already
        in the observation history.  The latter should normally be unavailable
        in a no-repeat run, but retaining the definition makes this utility
        safe for arbitrary public candidate pools.
        """
        if not self._train_conditions:
            return np.ones(len(candidates), dtype=float)
        history = np.asarray(
            [[row[column] for column in self.variable_columns] for row in self._train_conditions],
            dtype=object,
        )
        values = candidates.loc[:, self.variable_columns].astype(str).to_numpy(dtype=object)
        distances = (values[:, None, :] != history[None, :, :]).mean(axis=2)
        return distances.min(axis=1)

    def _tail_probability(self, candidates: pd.DataFrame) -> np.ndarray:
        """Estimate membership in the observed high-yield tail.

        The threshold and classifier use only outcomes already released to the
        optimizer.  A deterministic seed derived from trajectory state keeps
        this auxiliary ranker from perturbing the legacy surrogate RNG stream.
        """
        y = np.asarray(self._train_y, dtype=float)
        threshold = float(np.quantile(y, self.tail_quantile))
        labels = (y >= threshold).astype(int)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives < 2 or negatives < 2:
            return np.zeros(len(candidates), dtype=float)
        model = ExtraTreesClassifier(
            n_estimators=max(48, min(self.ensemble_size, 100)),
            min_samples_leaf=max(1, self.forest_min_samples_leaf),
            max_features=1.0,
            class_weight="balanced",
            n_jobs=1,
            random_state=self.seed + 700_000 + len(y),
        ).fit(np.asarray(self._train_x, dtype=float), labels)
        classes = list(model.classes_)
        positive_column = classes.index(1)
        return model.predict_proba(self._encode(candidates))[:, positive_column]

    def _apply_preference(
        self,
        candidates: pd.DataFrame,
        mean: np.ndarray,
        candidate_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Mapping[str, float | int]]:
        if self.preference_provider is None or self.preference_strength <= 0:
            return mean, np.zeros(len(mean), dtype=float), {
                "confidence": 0.0,
                "kendall_tau": 0.0,
                "region_size": 0,
                "max_shift": 0.0,
            }
        full_shift, diagnostics = self.preference_provider.lift(
            candidates=candidates,
            observed_conditions=self._training_frame(),
            observed_y=self._train_y,
            strength=self.preference_strength,
        )
        full_shift = np.asarray(full_shift, dtype=float)
        shift = full_shift if candidate_indices is None else full_shift[candidate_indices]
        if shift.shape != mean.shape or not np.isfinite(shift).all():
            raise ValueError("Preference provider returned an invalid mean shift")
        # Deliberately preserve the surrogate uncertainty: only its mean is
        # lifted, matching the tractable LGBO construction.
        return mean + shift, shift, diagnostics

    def _capture_adaptive_pending(self, local_index: int) -> None:
        if self.surrogate != "adaptive_ensemble" or not self._last_adaptive_components:
            self._pending_adaptive_predictions = None
            return
        self._pending_adaptive_predictions = {
            name: float(values[0][local_index])
            for name, values in self._last_adaptive_components.items()
        }

    def recommend(
        self,
        candidates: pd.DataFrame,
        queried_indices: Iterable[int],
        target_incumbent: float,
        round_index: int,
        total_rounds: int,
    ) -> Recommendation:
        queried = set(int(index) for index in queried_indices)
        desired_kernel = (
            self.kernel_after_switch
            if (
                self.kernel_after_switch is not None
                and round_index >= self.kernel_switch_rounds
            )
            else self.kernel
        )
        if desired_kernel != self._active_kernel:
            self._active_kernel = desired_kernel
            self._model = None
        if self.hybrid_forest_weight_end is None or total_rounds <= 1:
            self._active_hybrid_forest_weight = self.hybrid_forest_weight
        else:
            progress = min(max(round_index, 0), total_rounds - 1) / (total_rounds - 1)
            self._active_hybrid_forest_weight = self.hybrid_forest_weight + progress * (
                self.hybrid_forest_weight_end - self.hybrid_forest_weight
            )
        if self.product_transfer_weight_end is None or total_rounds <= 1:
            self._active_product_transfer_weight = self.product_transfer_weight
        else:
            progress = min(max(round_index, 0), total_rounds - 1) / (total_rounds - 1)
            self._active_product_transfer_weight = self.product_transfer_weight + progress * (
                self.product_transfer_weight_end - self.product_transfer_weight
            )
        available = np.asarray([index not in queried for index in range(len(candidates))])
        if not available.any():
            raise RuntimeError("No unqueried candidates remain")
        available_indices = np.flatnonzero(available)

        if self.strategy == "random":
            chosen = int(self.rng.choice(available_indices))
            return Recommendation(chosen, float("nan"), float("nan"), float("nan"), "random_baseline")

        if round_index < self.warmup_random_rounds:
            chosen = int(self.rng.choice(available_indices))
            mean, std = self.predict(candidates.iloc[[chosen]])
            self._capture_adaptive_pending(0)
            std = std * self.uncertainty_scale
            return Recommendation(chosen, float(mean[0]), float(std[0]), float("nan"), "warmup_random")

        # Consume the policy draw at the same point for ordinary and greedy
        # routes.  Otherwise a greedy ablation would also change downstream
        # forest/bootstrap seeds because model fitting and policy sampling
        # intentionally share this legacy deterministic RNG stream.
        epsilon = self._epsilon(round_index, total_rounds)
        explore_random = float(self.rng.random())

        if round_index < self.top_probability_rounds:
            opening_candidates = candidates.iloc[available_indices]
            probability = self._tail_probability(opening_candidates)
            if np.max(probability) > 0:
                mean, std = self.predict(opening_candidates)
                mean, preference_shift, preference_diagnostics = self._apply_preference(
                    candidates, mean, available_indices
                )
                std = std * self.uncertainty_scale
                best_probability = np.max(probability)
                probability_ties = np.flatnonzero(
                    np.isclose(
                        probability,
                        best_probability,
                        rtol=1e-10,
                        atol=1e-12,
                    )
                )
                best_mean = np.max(mean[probability_ties])
                ties = probability_ties[
                    np.flatnonzero(
                        np.isclose(
                            mean[probability_ties],
                            best_mean,
                            rtol=1e-10,
                            atol=1e-12,
                        )
                    )
                ]
                local = int(self.rng.choice(ties))
                self._capture_adaptive_pending(local)
                return Recommendation(
                    query_index=int(available_indices[local]),
                    predicted_yield=float(mean[local]),
                    uncertainty=float(std[local]),
                    acquisition_value=float(probability[local]),
                    selection_mode="top_tail_probability",
                    preference_confidence=float(
                        preference_diagnostics["confidence"]
                    ),
                    preference_shift=float(preference_shift[local]),
                    ensemble_weights=dict(self._last_adaptive_weights) or None,
                )

        if round_index < self.greedy_rounds:
            mean, std = self.predict(candidates.iloc[available_indices])
            mean, preference_shift, preference_diagnostics = self._apply_preference(
                candidates, mean, available_indices
            )
            std = std * self.uncertainty_scale
            best = np.max(mean)
            ties = np.flatnonzero(np.isclose(mean, best, rtol=1e-10, atol=1e-12))
            local = int(self.rng.choice(ties))
            self._capture_adaptive_pending(local)
            return Recommendation(
                query_index=int(available_indices[local]),
                predicted_yield=float(mean[local]),
                uncertainty=float(std[local]),
                acquisition_value=float(mean[local]),
                selection_mode=(
                    "posterior_mean_preference_lift"
                    if float(preference_diagnostics["confidence"]) > 0
                    else "posterior_mean"
                ),
                preference_confidence=float(preference_diagnostics["confidence"]),
                preference_shift=float(preference_shift[local]),
                ensemble_weights=dict(self._last_adaptive_weights) or None,
            )

        if explore_random < epsilon:
            chosen = int(self.rng.choice(available_indices))
            mean, std = self.predict(candidates.iloc[[chosen]])
            self._capture_adaptive_pending(0)
            std = std * self.uncertainty_scale
            return Recommendation(chosen, float(mean[0]), float(std[0]), float("nan"), "random_exploration")

        if self.strategy == "tpe":
            score = self._tpe_score(candidates.iloc[available_indices])
            best = np.max(score)
            ties = np.flatnonzero(np.isclose(score, best, rtol=1e-10, atol=1e-12))
            local = int(self.rng.choice(ties))
            return Recommendation(
                query_index=int(available_indices[local]),
                predicted_yield=float("nan"),
                uncertainty=float("nan"),
                acquisition_value=float(score[local]),
                selection_mode="tpe",
            )

        if self.strategy == "thompson":
            mean, std = self.predict(candidates.iloc[available_indices])
            mean, preference_shift, preference_diagnostics = self._apply_preference(
                candidates, mean, available_indices
            )
            std = std * self.uncertainty_scale
            sampled = mean + self.rng.normal(size=len(mean)) * std
            best = np.max(sampled)
            ties = np.flatnonzero(np.isclose(sampled, best, rtol=1e-10, atol=1e-12))
            local = int(self.rng.choice(ties))
            self._capture_adaptive_pending(local)
            return Recommendation(
                query_index=int(available_indices[local]),
                predicted_yield=float(mean[local]),
                uncertainty=float(std[local]),
                acquisition_value=float(sampled[local]),
                selection_mode=(
                    "thompson_preference_lift"
                    if float(preference_diagnostics["confidence"]) > 0
                    else "thompson"
                ),
                preference_confidence=float(preference_diagnostics["confidence"]),
                preference_shift=float(preference_shift[local]),
                ensemble_weights=dict(self._last_adaptive_weights) or None,
            )

        mean, std = self.predict(candidates.iloc[available_indices])
        mean, preference_shift, preference_diagnostics = self._apply_preference(
            candidates, mean, available_indices
        )
        std = std * self.uncertainty_scale
        if self.acquisition == "expected_improvement":
            improvement = mean - float(target_incumbent)
            z = improvement / std
            score = improvement * norm.cdf(z) + std * norm.pdf(z)
        else:
            score = mean + self.ucb_beta * std
        if self.diversity_weight:
            # EI/UCB is expressed in yield units.  Scale the dimensionless
            # Hamming-distance bonus by the observed yield spread so the same
            # setting remains meaningful for both reaction datasets.
            observed_scale = max(float(np.std(self._train_y)), 1.0)
            score = score + self.diversity_weight * observed_scale * self._nearest_history_diversity(
                candidates.iloc[available_indices]
            )
        if self.tail_probability_weight:
            observed_scale = max(float(np.std(self._train_y)), 1.0)
            tail_probability = self._tail_probability(
                candidates.iloc[available_indices]
            )
            score = (
                score
                + self.tail_probability_weight
                * observed_scale
                * tail_probability
            )
        max_score = np.max(score)
        ties = np.flatnonzero(np.isclose(score, max_score, rtol=1e-10, atol=1e-12))
        local = int(self.rng.choice(ties))
        self._capture_adaptive_pending(local)
        preference_active = float(preference_diagnostics["confidence"]) > 0
        mode = f"{self.acquisition}_diverse" if self.diversity_weight else self.acquisition
        if self.tail_probability_weight:
            mode = f"{mode}_tail_probability"
        if preference_active:
            mode = f"{mode}_preference_lift"
        return Recommendation(
            query_index=int(available_indices[local]),
            predicted_yield=float(mean[local]),
            uncertainty=float(std[local]),
            acquisition_value=float(score[local]),
            selection_mode=mode,
            preference_confidence=float(preference_diagnostics["confidence"]),
            preference_shift=float(preference_shift[local]),
            ensemble_weights=dict(self._last_adaptive_weights) or None,
        )
