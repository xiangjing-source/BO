#!/usr/bin/env python3
"""Generate post-hoc final figures and supplementary metrics.

This script is evaluation-only. It reads completed trajectories and the oracle
only after all model configurations and 40-round paths have been frozen. No
output is returned to an optimizer or used for candidate selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import norm
from scipy.stats import t as student_t


SEEDS = list(range(100, 2001, 100))
DATASET_FILES = {
    "Suzuki": {
        "train": "suzuki_train.csv",
        "test": "suzuki_test.csv",
        "variables": ["Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"],
    },
    "Buchwald_sub4": {
        "train": "buchwald_sub4_train.csv",
        "test": "buchwald_sub4_test.csv",
        "variables": ["Reactant2", "Ligand", "Additive", "Base", "Product"],
    },
}
DATASET_TITLES = {
    "Suzuki": "Suzuki–Miyaura C–C coupling",
    "Buchwald_sub4": "Buchwald–Hartwig C–N coupling",
}
STAGES = {
    "Suzuki": {
        "Random search": "results/model_evolution/internal_stage_runs/suzuki_stage1_random/optimization_trajectories/Suzuki",
        "Conventional mixed-kernel Bayesian optimization": "results/model_selection/dual_benchmark_acquisition_v4_20x40/Suzuki/optimization_trajectories/Suzuki",
        "Random forest + tensor-product categorical-kernel Gaussian process": "results/final_external_evaluation_20x40/Suzuki/optimization_trajectories/Suzuki",
    },
    "Buchwald_sub4": {
        "Random search": "results/model_evolution/internal_stage_runs/buchwald_stage1_random/optimization_trajectories/Buchwald_sub4",
        "Fused forest with random exploration": "results/model_selection/dual_benchmark_v2_20x40/Buchwald_sub4/optimization_trajectories/Buchwald_sub4",
        "Random forest with categorical, chemical-knowledge, and structural features": "results/final_external_evaluation_20x40/Buchwald_sub4/optimization_trajectories/Buchwald_sub4",
    },
}
COLORS = {
    "Random search": "#9AA4AD",
    "Conventional mixed-kernel Bayesian optimization": "#E6A23C",
    "Fused forest with random exploration": "#E6A23C",
    "Random forest + tensor-product categorical-kernel Gaussian process": "#3B5CCC",
    "Random forest with categorical, chemical-knowledge, and structural features": "#3B5CCC",
}
METRICS = (
    ("initial_round_found_best", "Initial best", True),
    ("best_found", "Best found", True),
    ("round_to_95_global_best", "Rounds to 95% (lower is better)", False),
    ("AUC_best_so_far", "Best-so-far AUC", True),
)


def read_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "trajectory" not in payload:
        raise ValueError(f"Unexpected trajectory payload: {path}")
    return payload


def load_stage(root: Path, relative: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    curves = []
    payloads = []
    trajectory_root = root / relative
    for seed in SEEDS:
        path = trajectory_root / f"seed_{seed:04d}.pt"
        payload = read_payload(path)
        records = payload["trajectory"]
        if len(records) != 40:
            raise ValueError(f"{path}: expected 40 steps")
        curves.append([float(record["best_so_far"]) for record in records])
        payloads.append(payload)
    return np.asarray(curves, dtype=float), payloads


def official_metrics(curve: np.ndarray, global_best: float) -> dict[str, float]:
    hits = np.flatnonzero(curve >= 0.95 * global_best)
    return {
        "initial_round_found_best": float(curve[0]),
        "best_found": float(curve[-1]),
        "round_to_95_global_best": float(hits[0] + 1) if len(hits) else 41.0,
        "AUC_best_so_far": float(curve.mean()),
    }


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def wilson_ci(count: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a binomial hit rate."""
    z = float(norm.ppf(0.975))
    rate = count / n
    denominator = 1.0 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / n + z**2 / (4 * n**2)) / denominator
    return float(center - half), float(center + half)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.0,
        "figure.dpi": 130,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.color": "#80909A",
        "axes.facecolor": "#FBFCFD",
    })


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Keep a moderately sized, high-resolution derivative for the submitted
    # Word/PDF reports. The HTML renderer sets its *display* width separately,
    # so the embedded bitmap remains sharp when readers zoom or print it.
    with Image.open(png_path) as image:
        report_width = min(1600, image.width)
        if image.width != report_width:
            height = max(1, round(image.height * report_width / image.width))
            image.resize((report_width, height), Image.Resampling.LANCZOS).save(
                output_dir / f"{name}_report.png",
                optimize=True,
                dpi=(240, 240),
            )
        else:
            image.save(
                output_dir / f"{name}_report.png",
                optimize=True,
                dpi=(240, 240),
            )


def plot_framework(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def group(x: float, y: float, w: float, h: float, title: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.16",
            linewidth=1.25, edgecolor=color, facecolor=mpl.colors.to_rgba(color, 0.025),
            linestyle=(0, (4, 3)),
        )
        ax.add_patch(patch)
        ax.text(x + 0.18, y + h - 0.28, title, ha="left", va="center",
                fontsize=11.2, fontweight="bold", color=color)

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        color: str,
        fontsize: float = 9.2,
        face_alpha: float = 0.11,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.35, edgecolor=color, facecolor=mpl.colors.to_rgba(color, face_alpha),
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#52636b",
        rad: float = 0.0,
    ) -> None:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.35, color=color, connectionstyle=f"arc3,rad={rad}",
        ))

    group(0.25, 0.85, 4.0, 6.35, "1  Evidence and guardrails", "#596A75")
    group(4.55, 0.85, 4.45, 6.35, "2  Frozen reaction-family route", "#7A5EB3")
    group(9.3, 0.85, 5.45, 6.35, "3  Auditable sequential optimization", "#2F6F73")

    box(0.60, 4.95, 3.30, 1.35,
        "Public optimization inputs\ninitial labelled experiments\noutcome-free candidate table",
        "#4C78A8")
    box(0.60, 3.10, 3.30, 1.35,
        "Label-free chemical evidence\nmechanisms · molecular structures\nliterature · audited ORCA calculations",
        "#6E8B3D")
    box(0.60, 1.40, 3.30, 1.10,
        "Safety gate\nschema check · outcome boundary\nindependent seed state · audit log",
        "#596A75")

    box(4.95, 5.15, 3.65, 1.15,
        "Static router\nreaction identity and declared schema\nfixed before the first recommendation",
        "#7A5EB3")
    box(4.95, 3.45, 3.65, 1.20,
        "Suzuki–Miyaura surrogate\nrole-wise categorical kernel\nweighted with a random forest",
        "#3B5CCC")
    box(4.95, 1.70, 3.65, 1.20,
        "Buchwald–Hartwig surrogate\nidentity, chemical knowledge and\nmolecular structure in one forest",
        "#D66A4A")

    box(9.70, 5.05, 2.00, 1.15, "① Fit surrogate\nmean and uncertainty", "#2F6F73")
    box(12.20, 5.05, 2.00, 1.15, "② Rank candidates\nExpected Improvement", "#2F6F73")
    box(12.20, 3.05, 2.00, 1.15, "③ Recommend one\nexperiment / oracle", "#2F6F73")
    box(9.70, 3.05, 2.00, 1.15, "④ Append feedback\nqueried outcome only", "#2F6F73")
    box(9.55, 1.35, 2.35, 1.00,
        "Post-hoc evaluation\nmetrics · confidence intervals\ntrajectory-level audit",
        "#8A8F94", fontsize=8.3)
    box(12.15, 1.35, 2.35, 1.00,
        "Post-selection DFT explainer\nrole-wise percentiles\nsupport · risk · confidence",
        "#6E8B3D", fontsize=8.2)

    arrow(3.90, 5.62, 4.95, 5.72)
    arrow(3.90, 3.77, 4.95, 4.05)
    arrow(3.90, 1.95, 4.95, 2.20)
    arrow(6.78, 5.15, 6.78, 4.65)
    arrow(5.05, 5.15, 4.95, 2.30, rad=0.32)
    arrow(8.60, 4.05, 9.70, 5.45)
    arrow(8.60, 2.30, 9.70, 5.30, rad=-0.30)
    arrow(11.70, 5.62, 12.20, 5.62)
    arrow(13.20, 5.05, 13.20, 4.20)
    arrow(12.20, 3.62, 11.70, 3.62)
    arrow(10.70, 4.20, 10.70, 5.05)
    arrow(10.70, 3.05, 10.70, 2.35)
    arrow(13.20, 3.05, 13.30, 2.35)

    ax.text(7.5, 7.58, "CatalystFlow: one safe backbone, two chemistry-aware surrogate routes",
            ha="center", va="center", fontsize=16.5, fontweight="bold", color="#173847")
    ax.text(
        7.5, 0.35,
        "Only queried feedback updates the model. DFT explanations are generated after selection and never alter ranking.",
        ha="center", va="center", fontsize=9.4, color="#6B2F2F",
    )
    save_figure(fig, output_dir, "framework_overview")


def plot_best_so_far(
    stage_curves: dict[str, dict[str, np.ndarray]],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharex=True)
    rounds = np.arange(1, 41)
    for ax, dataset in zip(axes, ("Suzuki", "Buchwald_sub4")):
        for label, curves in stage_curves[dataset].items():
            mean = curves.mean(axis=0)
            half = student_t.ppf(0.975, len(curves) - 1) * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            color = COLORS[label]
            ax.plot(rounds, mean, label=f"{label} (AUC={mean.mean():.2f})", color=color, linewidth=2.2)
            ax.fill_between(rounds, mean - half, mean + half, color=color, alpha=0.15, linewidth=0)
        final_label = list(stage_curves[dataset])[-1]
        final_mean = stage_curves[dataset][final_label].mean(axis=0)
        lower_bound = 70 if dataset == "Suzuki" else 45
        ax.fill_between(rounds, lower_bound, final_mean, color=COLORS[final_label], alpha=0.035)
        ax.axhline(global_bests[dataset], color="#263238", linestyle="--", linewidth=1.0,
                   label=f"Candidate-pool maximum ({global_bests[dataset]:.2f}%)")
        ax.axhline(0.95 * global_bests[dataset], color="#8A6D3B", linestyle=":", linewidth=1.1,
                   label="95% target")
        ax.set_title(DATASET_TITLES[dataset])
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("Best-so-far yield (%)")
        ax.set_xlim(1, 40)
        ax.set_ylim(lower_bound, 102 if dataset == "Suzuki" else 90)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.suptitle(
        "Best-so-far optimization curves (mean and pointwise 95% CI across 20 seeds)",
        fontsize=14, fontweight="bold",
    )
    fig.text(0.5, -0.01, "AUC_best_so_far = arithmetic mean of the 40 recorded best-so-far values.",
             ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_dir, "best_so_far_curves")


def plot_attainment(
    stage_curves: dict[str, dict[str, np.ndarray]],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True, sharey=True)
    rounds = np.arange(1, 41)
    for ax, dataset in zip(axes, ("Suzuki", "Buchwald_sub4")):
        threshold = 0.95 * global_bests[dataset]
        for label, curves in stage_curves[dataset].items():
            attained = (curves >= threshold).any(axis=1)
            first = np.full(len(curves), 41, dtype=int)
            for i, curve in enumerate(curves):
                hits = np.flatnonzero(curve >= threshold)
                if len(hits):
                    first[i] = int(hits[0] + 1)
            rates = np.asarray([(first <= r).mean() for r in rounds])
            ax.step(rounds, rates, where="post", linewidth=2.2, color=COLORS[label], label=label)
        ax.set_title(f"{DATASET_TITLES[dataset]}\n95% target = {threshold:.2f}%")
        ax.set_xlabel("Optimization round")
        ax.set_ylabel("Fraction of seeds reaching 95% global best")
        ax.set_xlim(1, 40)
        ax.set_ylim(0, 1.03)
        ax.legend(loc="lower right")
    fig.suptitle("Empirical 95%-attainment curves across 20 independent seeds",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "t95_attainment_curves")


def plot_core_metric_intervals(
    stage_curves: dict[str, dict[str, np.ndarray]],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for row, dataset in enumerate(("Suzuki", "Buchwald_sub4")):
        stage_labels = list(stage_curves[dataset])
        for col, (metric, title, _) in enumerate(METRICS):
            ax = axes[row, col]
            for y, label in enumerate(stage_labels):
                values = np.asarray([
                    official_metrics(curve, global_bests[dataset])[metric]
                    for curve in stage_curves[dataset][label]
                ])
                mean, low, high = mean_ci(values)
                ax.errorbar(
                    mean, y, xerr=[[mean - low], [high - mean]],
                    fmt="o", markersize=6.5, capsize=3, color=COLORS[label],
                )
                ax.text(mean, y + 0.18, f"{mean:.2f}", ha="center", fontsize=7.5)
            ax.set_yticks(range(len(stage_labels)), stage_labels if col == 0 else [""] * len(stage_labels))
            ax.invert_yaxis()
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(dataset)
    fig.suptitle("Core metrics with 95% confidence intervals (20 seeds)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "core_metrics_confidence_intervals")


def plot_seed_heatmaps(
    final_curves: dict[str, np.ndarray],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    image = None
    for ax, dataset in zip(axes, ("Suzuki", "Buchwald_sub4")):
        normalized = final_curves[dataset] / global_bests[dataset]
        image = ax.imshow(normalized, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.65, vmax=1.0)
        ax.set_title(f"{DATASET_TITLES[dataset]}\nfinal method: best-so-far / candidate-pool maximum")
        ax.set_ylabel("Seed")
        ax.set_yticks(range(len(SEEDS)), [str(seed) for seed in SEEDS], fontsize=7)
    axes[-1].set_xlabel("Optimization round")
    axes[-1].set_xticks(np.arange(0, 40, 4), [str(i) for i in range(1, 41, 4)])
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of global best")
    fig.suptitle("Per-seed stability of the frozen final models", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "per_seed_best_so_far_heatmaps")


def plot_performance_overview(
    stage_curves: dict[str, dict[str, np.ndarray]],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    """Paper-style convergence traces paired with per-seed trajectory heatmaps."""
    fig = plt.figure(figsize=(14.8, 8.3))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.55, 1.0), hspace=0.38, wspace=0.18,
        left=0.07, right=0.92, top=0.88, bottom=0.09,
    )
    rounds = np.arange(1, 41)
    image = None
    for row, dataset in enumerate(("Suzuki", "Buchwald_sub4")):
        ax_curve = fig.add_subplot(grid[row, 0])
        ax_heat = fig.add_subplot(grid[row, 1])
        for label, curves in stage_curves[dataset].items():
            mean = curves.mean(axis=0)
            half = (
                student_t.ppf(0.975, len(curves) - 1)
                * curves.std(axis=0, ddof=1)
                / np.sqrt(len(curves))
            )
            color = COLORS[label]
            width = 2.6 if label == list(stage_curves[dataset])[-1] else 1.8
            ax_curve.plot(rounds, mean, color=color, linewidth=width, label=label)
            ax_curve.fill_between(
                rounds, mean - half, mean + half,
                color=color, alpha=0.14 if width > 2 else 0.09, linewidth=0,
            )
        global_best = global_bests[dataset]
        ax_curve.axhline(global_best, color="#263238", linestyle="--", linewidth=1.0)
        ax_curve.axhline(0.95 * global_best, color="#A36B15", linestyle=":", linewidth=1.2)
        ax_curve.text(
            39.7, global_best - 0.35, "candidate-pool maximum",
            ha="right", va="top", fontsize=7.4, color="#263238",
        )
        ax_curve.text(
            39.7, 0.95 * global_best + 0.30, "95% target",
            ha="right", va="bottom", fontsize=7.4, color="#8A5C16",
        )
        ax_curve.set_xlim(1, 40)
        ax_curve.set_ylim((70, 102) if dataset == "Suzuki" else (45, 90))
        ax_curve.set_xlabel("Optimization round")
        ax_curve.set_ylabel("Best-so-far yield (%)")
        ax_curve.set_title(f"{chr(65 + row * 2)}  {DATASET_TITLES[dataset]} — convergence",
                           loc="left", fontweight="bold")
        ax_curve.legend(loc="lower right", frameon=True, framealpha=0.95)

        final_label = list(stage_curves[dataset])[-1]
        final_curves = stage_curves[dataset][final_label]
        normalized = final_curves / global_best
        image = ax_heat.imshow(
            normalized, aspect="auto", interpolation="nearest",
            cmap="viridis", vmin=0.50, vmax=1.0,
        )
        ax_heat.set_title(f"{chr(66 + row * 2)}  Per-seed convergence stability",
                          loc="left", fontweight="bold")
        ax_heat.set_ylabel("Independent seed")
        ax_heat.set_xlabel("Optimization round")
        ax_heat.set_yticks(range(len(SEEDS)), [str(seed) for seed in SEEDS], fontsize=6.7)
        ax_heat.set_xticks(np.arange(0, 40, 4), [str(i) for i in range(1, 41, 4)])
        ax_heat.grid(False)

        final_metric_values = [
            official_metrics(curve, global_best)
            for curve in final_curves
        ]
        metric_text = (
            f"Initial {np.mean([m['initial_round_found_best'] for m in final_metric_values]):.2f}   "
            f"Best {np.mean([m['best_found'] for m in final_metric_values]):.2f}   "
            f"t95 {np.mean([m['round_to_95_global_best'] for m in final_metric_values]):.2f}   "
            f"AUC {np.mean([m['AUC_best_so_far'] for m in final_metric_values]):.2f}"
        )
        ax_heat.text(
            0.5, 0.025, metric_text, transform=ax_heat.transAxes,
            ha="center", va="bottom", fontsize=8.0, color="#263238",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F6F8",
                  "edgecolor": "#B8C7CC", "linewidth": 0.8, "alpha": 0.94},
        )

    assert image is not None
    colorbar_ax = fig.add_axes([0.94, 0.18, 0.015, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Best-so-far / candidate-pool maximum")
    fig.suptitle(
        "Optimization performance and robustness across 20 independent runs",
        fontsize=16, fontweight="bold", color="#173847",
    )
    fig.text(
        0.5, 0.015,
        "Lines show mean best-so-far yield; shaded bands are pointwise 95% confidence intervals. "
        "Heatmaps retain every seed and round.",
        ha="center", fontsize=8.8, color="#52636B",
    )
    save_figure(fig, output_dir, "optimization_performance_overview")


def build_supplementary_tables(
    project_root: Path,
    data_root: Path,
    stage_curves: dict[str, dict[str, np.ndarray]],
    final_payloads: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    per_seed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    required = {"step", "query_index", "condition", "observed_yield", "best_so_far"}
    optional = {
        "predicted_yield", "uncertainty", "acquisition_value", "selection_mode",
        "preference_confidence", "preference_shift", "ensemble_weights",
        "dft_components_available", "mechanistic_dft_components_available",
    }

    schema = {
        "serialization": "Python standard-library pickle with .pt suffix; no PyTorch dependency",
        "top_level": {
            "seed": "official random seed",
            "dataset": "dataset identifier",
            "trajectory": "ordered list of exactly 40 round records",
        },
        "round_fields": {
            "step": "one-indexed optimization round",
            "query_index": "row index in the public *_test_features.csv candidate table",
            "condition": "all public condition variables at query_index",
            "observed_yield": "oracle/experiment feedback released after selection",
            "best_so_far": "maximum observed target value through this round",
            "predicted_yield": "pre-query surrogate mean at the selected candidate",
            "uncertainty": "pre-query surrogate empirical/posterior standard deviation",
            "acquisition_value": "pre-query acquisition score",
            "selection_mode": "expected_improvement or frozen random-exploration mode",
            "preference_confidence": "pre-query optional preference confidence; zero when disabled",
            "preference_shift": "pre-query optional mean shift; zero when disabled",
            "ensemble_weights": "optional adaptive-ensemble weights; null for final models",
            "dft_components_available": "optional generic DFT coverage count; null when not in final ranker",
            "mechanistic_dft_components_available": "optional mechanistic DFT coverage count; null when not in final ranker",
        },
        "metric_definitions": {
            "initial_round_found_best": "best_so_far at step 1",
            "best_found": "best_so_far at step 40",
            "round_to_95_global_best": "first one-indexed step with best_so_far >= 0.95*global_best; 41 if not reached",
            "AUC_best_so_far": "arithmetic mean of the 40 best_so_far values",
        },
    }
    (output_dir / "trajectory_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for dataset in ("Suzuki", "Buchwald_sub4"):
        files = DATASET_FILES[dataset]
        train = pd.read_csv(data_root / dataset / files["train"])
        test = pd.read_csv(data_root / dataset / files["test"])
        target_product = test["Product"].iloc[0] if dataset == "Buchwald_sub4" else None
        target_train = train.loc[train["Product"] == target_product] if target_product is not None else train
        global_best = float(test["Yield"].max())
        test_min = float(test["Yield"].min())
        test_max_indices = test.index[np.isclose(test["Yield"], global_best)].tolist()
        final_curves = stage_curves[dataset][list(stage_curves[dataset])[-1]]

        source_rows.append({
            "dataset": dataset,
            "initial_train_rows": len(train),
            "target_initial_train_rows": len(target_train),
            "candidate_rows": len(test),
            "initial_train_yield_min": float(train["Yield"].min()),
            "initial_train_yield_max": float(train["Yield"].max()),
            "target_initial_train_yield_max": float(target_train["Yield"].max()),
            "candidate_yield_min": test_min,
            "candidate_global_best": global_best,
            "global_best_candidate_count": len(test_max_indices),
            "global_best_candidate_indices": ";".join(map(str, test_max_indices)),
        })

        for seed, curve, payload in zip(SEEDS, final_curves, final_payloads[dataset]):
            records = payload["trajectory"]
            best = float(curve[-1])
            metrics = official_metrics(curve, global_best)
            predictions = np.asarray([float(r["predicted_yield"]) for r in records])
            observed = np.asarray([float(r["observed_yield"]) for r in records])
            uncertainty = np.asarray([float(r["uncertainty"]) for r in records])
            per_seed_rows.append({
                "dataset": dataset,
                "seed": seed,
                **metrics,
                "global_best": global_best,
                "final_simple_regret": max(global_best - best, 0.0),
                "final_fraction_of_global_best": best / global_best,
                "hit_exact_global_best": int(np.isclose(best, global_best, atol=1e-9)),
                "hit_99_percent_by_budget": int(best >= 0.99 * global_best),
                "hit_95_percent_by_round_5": int(np.any(curve[:5] >= 0.95 * global_best)),
                "hit_95_percent_by_round_10": int(np.any(curve[:10] >= 0.95 * global_best)),
                "hit_95_percent_by_round_20": int(np.any(curve[:20] >= 0.95 * global_best)),
                "hit_95_percent_by_budget": int(best >= 0.95 * global_best),
                "early_AUC_5": float(curve[:5].mean()),
                "early_AUC_10": float(curve[:10].mean()),
                "early_AUC_20": float(curve[:20].mean()),
                "selected_prediction_MAE": float(np.mean(np.abs(observed - predictions))),
                "selected_prediction_RMSE": float(np.sqrt(np.mean((observed - predictions) ** 2))),
                "predictive_95_coverage": float(np.mean(np.abs(observed - predictions) <= 1.96 * np.maximum(uncertainty, 1e-8))),
                "mean_selected_uncertainty": float(uncertainty.mean()),
            })
            all_fields = set.intersection(*(set(record) for record in records))
            audit_rows.append({
                "dataset": dataset,
                "seed": seed,
                "rounds": len(records),
                "consecutive_steps": all(int(r["step"]) == i for i, r in enumerate(records, 1)),
                "unique_query_indices": len({int(r["query_index"]) for r in records}),
                "all_required_fields_present": required <= all_fields,
                "optional_fields_present_all_rounds": ";".join(sorted(optional & all_fields)),
                "top_level_seed_matches": int(payload["seed"]) == seed,
                "top_level_dataset_matches": payload["dataset"] == dataset,
            })

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(output_dir / "supplementary_metrics_per_seed.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "trajectory_field_audit.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "dataset_and_global_best_summary.csv", index=False)

    summary_rows = []
    count_metrics = (
        "hit_exact_global_best", "hit_99_percent_by_budget",
        "hit_95_percent_by_round_5", "hit_95_percent_by_round_10",
        "hit_95_percent_by_round_20", "hit_95_percent_by_budget",
    )
    continuous = (
        "final_simple_regret", "final_fraction_of_global_best",
        "early_AUC_5", "early_AUC_10", "early_AUC_20",
        "selected_prediction_MAE", "selected_prediction_RMSE",
        "predictive_95_coverage", "mean_selected_uncertainty",
    )
    for dataset, group in per_seed.groupby("dataset", sort=False):
        for metric in count_metrics:
            count = int(group[metric].sum())
            low, high = wilson_ci(count, len(group))
            summary_rows.append({
                "dataset": dataset, "metric": metric, "n": len(group),
                "mean_or_rate": count / len(group),
                "std": float(group[metric].std(ddof=1)),
                "ci95_lower": low, "ci95_upper": high,
                "count": count,
            })
        for metric in continuous:
            mean, low, high = mean_ci(group[metric].to_numpy(float))
            summary_rows.append({
                "dataset": dataset, "metric": metric, "n": len(group),
                "mean_or_rate": mean, "std": float(group[metric].std(ddof=1)),
                "ci95_lower": low, "ci95_upper": high, "count": np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "supplementary_metrics_summary.csv", index=False)


def write_model_progression_tables(
    stage_curves: dict[str, dict[str, np.ndarray]],
    global_bests: dict[str, float],
    output_dir: Path,
) -> None:
    """Persist the exact plotted values so figures remain independently auditable."""
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for dataset, methods in stage_curves.items():
        for method, curves in methods.items():
            for seed, curve in zip(SEEDS, curves):
                for round_index, best in enumerate(curve, 1):
                    curve_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "round": round_index,
                        "best_so_far": float(best),
                    })
                metric_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    **official_metrics(curve, global_bests[dataset]),
                })
    pd.DataFrame(curve_rows).to_csv(
        output_dir / "model_progression_curves.csv", index=False
    )
    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(output_dir / "model_progression_metrics_per_seed.csv", index=False)
    summary_rows = []
    for (dataset, method), group in per_seed.groupby(["dataset", "method"], sort=False):
        for metric, _, _ in METRICS:
            values = group[metric].to_numpy(float)
            mean, low, high = mean_ci(values)
            summary_rows.append({
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": float(values.std(ddof=1)),
                "ci95_lower": low,
                "ci95_upper": high,
            })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "model_progression_metrics_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    stage_curves: dict[str, dict[str, np.ndarray]] = {}
    final_payloads: dict[str, list[dict[str, Any]]] = {}
    global_bests: dict[str, float] = {}
    for dataset in ("Suzuki", "Buchwald_sub4"):
        stage_curves[dataset] = {}
        for label, relative in STAGES[dataset].items():
            curves, payloads = load_stage(project_root, relative)
            stage_curves[dataset][label] = curves
            if label == list(STAGES[dataset])[-1]:
                final_payloads[dataset] = payloads
        test = pd.read_csv(data_root / dataset / DATASET_FILES[dataset]["test"])
        global_bests[dataset] = float(test["Yield"].max())

    plot_framework(output_dir)
    plot_best_so_far(stage_curves, global_bests, output_dir)
    plot_attainment(stage_curves, global_bests, output_dir)
    plot_core_metric_intervals(stage_curves, global_bests, output_dir)
    plot_performance_overview(stage_curves, global_bests, output_dir)
    final_curves = {
        dataset: curves[list(curves)[-1]]
        for dataset, curves in stage_curves.items()
    }
    plot_seed_heatmaps(final_curves, global_bests, output_dir)
    build_supplementary_tables(
        project_root, data_root, stage_curves, final_payloads, output_dir
    )
    write_model_progression_tables(stage_curves, global_bests, output_dir)
    print(f"Generated final figures and tables in {output_dir}")


if __name__ == "__main__":
    main()
