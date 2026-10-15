#!/usr/bin/env python3
"""Verify the CatalystFlow competition submission."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import tarfile
import zipfile
import xml.etree.ElementTree as ET


DATASETS = ("Suzuki", "Buchwald_sub4")
SEEDS = tuple(range(100, 2001, 100))
ROUND_FIELDS = {
    "step",
    "query_index",
    "condition",
    "observed_yield",
    "best_so_far",
    "predicted_yield",
    "uncertainty",
    "acquisition_value",
    "selection_mode",
}
REQUIRED = (
    "README.md",
    "MANIFEST.json",
    "data/README.md",
    "data/Suzuki/README.md",
    "data/Suzuki/options.json",
    "data/Suzuki/suzuki_train.csv",
    "data/Suzuki/suzuki_test_features.csv",
    "data/Suzuki/suzuki_test.csv",
    "data/Suzuki/suzuki_searchspace.csv",
    "data/Buchwald_sub4/README.md",
    "data/Buchwald_sub4/options.json",
    "data/Buchwald_sub4/buchwald_sub4_train.csv",
    "data/Buchwald_sub4/buchwald_sub4_test_features.csv",
    "data/Buchwald_sub4/buchwald_sub4_test.csv",
    "data/Buchwald_sub4/buchwald_sub4_searchspace.csv",
    "runtime/README.md",
    "runtime/bootstrap_portable_env.sh",
    "runtime/bootstrap_openscience.sh",
    "runtime/prepare_openscience_workspace.sh",
    "runtime/openscience_project_excludes",
    "runtime/configure_deepseek.sh",
    "runtime/verify_deepseek.sh",
    "runtime/openscience_cli.sh",
    "runtime/run_openscience.sh",
    "runtime/run_reproduction.sh",
    "runtime/run_reproduction_linux_arm64.sh",
    "runtime/run_reproduction_macos.sh",
    "runtime/run_reproduction_windows.ps1",
    "runtime/smoke_test_environment.py",
    "runtime/portable_environment.json",
    "runtime/catalystflow-linux-x86_64.tar.gz",
    "runtime/verify_openscience_runtime.sh",
    "runtime/verify_offline_bundle.sh",
    "runtime/openscience/README.md",
    "runtime/openscience/LICENSE",
    "runtime/openscience/NOTICE",
    "runtime/openscience/runtime.json",
    "runtime/openscience/openscience-linux-x64-2.0.23.tgz",
    "code/README.md",
    "code/Dockerfile",
    "code/environment.yml",
    "code/environment-runtime.yml",
    "code/main/run_selected.py",
    "code/config/production_config.json",
    "code/config/buchwald_structures.json",
    "code/config/mechanistic_preference_priors.json",
    "code/logs/README.md",
    "code/logs/reproduction_verification.log",
    "code/protocol.py",
    "code/run_compliant.py",
    "code/compliant_bo.py",
    "code/dft_explainer.py",
    "code/scripts/explain_dft_condition.py",
    "code/scripts/generate_dft_explanations.py",
    "code/scripts/validate_submission.py",
    "code/scripts/verify_final_submission.py",
    "code/scripts/apply_top5_evidence_snapshot.py",
    "doc/method_description.md",
    "doc/method_description.pdf",
    "doc/method_description.docx",
    "doc/final_delivery_guide.md",
    "doc/reproduction_guide.md",
    "doc/resume.pdf",
    "results/results_and_analysis.md",
    "results/results_and_analysis.pdf",
    "results/results_and_analysis.docx",
    "results/results_and_analysis.json",
    "results/top5_external_evidence.json",
    "results/summary_metrics.csv",
    "results/figures/framework_overview.png",
    "results/figures/optimization_performance_overview.png",
    "results/figures/core_metrics_confidence_intervals.png",
    "results/figures/model_progression_curves.csv",
    "results/figures/model_progression_metrics_per_seed.csv",
    "results/figures/model_progression_metrics_summary.csv",
    "results/figures/supplementary_metrics_per_seed.csv",
    "results/figures/supplementary_metrics_summary.csv",
    "results/figures/trajectory_schema.json",
    "results/audits/dft_component_features.csv",
    "results/audits/mechanistic_dft_component_features.csv",
    "results/dft_explanations/dft_explanation_examples.json",
    "results/dft_explanations/dft_explanation_examples.md",
    "results/dft_explanations/dft_coverage_summary.csv",
    "results/dft_explanations/dft_posthoc_associations.csv",
    "dft/README.md",
    "dft/orca_electronic_structure_database/summary.json",
    "dft/orca_electronic_structure_database/calculation_index.csv",
    "skill/optimize-coupling-reactions/INTERNAL_ADAPTER.md",
    "skill/catalystflow/SKILL.md",
    "skill/catalystflow/assets/feasibility_assessment_template.json",
    ".openscience/agent/catalystflow.md",
    "agent/README.md",
    "agent/mcp_server.py",
    "openscience.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        "__pycache__" in relative.parts
        or path.suffix == ".pyc"
        or relative.parts[0] in {
            ".catalystflow-agent",
            ".git",
            ".pytest_cache",
            "reproduced_results",
        }
        or (
            relative.parts[0] == "runtime"
            and len(relative.parts) > 1
            and relative.parts[1].startswith(
                (".portable_env", ".platform_env_", ".openscience_runtime")
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    args = parser.parse_args()
    root = Path(args.package_root).resolve()
    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    recorded = {item["path"]: item for item in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json" and not ignored(path, root)
    }
    if set(recorded) != actual:
        raise ValueError("MANIFEST.json does not exactly match the package")
    for relative, item in recorded.items():
        path = root / relative
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Manifest mismatch: {relative}")

    misplaced_outcome_tables = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower().endswith("_test.csv")
            or "searchspace.csv" in path.name.lower()
        )
        and "data" not in path.relative_to(root).parts[:1]
    ]
    if misplaced_outcome_tables:
        raise ValueError(
            "Outcome/search-space tables must remain inside the isolated data directory: "
            f"{misplaced_outcome_tables}"
        )

    data_specs = {
        "Suzuki": {
            "prefix": "suzuki",
            "variables": (
                "Electrophile",
                "Nucleophile",
                "Ligand",
                "Base",
                "Solvent",
            ),
            "expected_train": 29,
            "expected_test": 5731,
            "expected_searchspace": 5760,
        },
        "Buchwald_sub4": {
            "prefix": "buchwald_sub4",
            "variables": ("Reactant2", "Ligand", "Additive", "Base", "Product"),
            "expected_train": 35,
            "expected_test": 783,
            "expected_searchspace": 790,
        },
    }
    for dataset, spec in data_specs.items():
        data_root = root / "data" / dataset
        prefix = spec["prefix"]
        with (data_root / f"{prefix}_train.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            train_rows = list(csv.DictReader(handle))
        with (data_root / f"{prefix}_test_features.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            feature_rows = list(csv.DictReader(handle))
        with (data_root / f"{prefix}_test.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            oracle_rows = list(csv.DictReader(handle))
        with (data_root / f"{prefix}_searchspace.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            search_rows = list(csv.DictReader(handle))
        if (
            len(train_rows) != spec["expected_train"]
            or len(feature_rows) != spec["expected_test"]
            or len(oracle_rows) != spec["expected_test"]
            or len(search_rows) != spec["expected_searchspace"]
        ):
            raise ValueError(f"Official public data row counts disagree for {dataset}")
        if any("Yield" in row for row in feature_rows):
            raise ValueError(f"Candidate feature table contains Yield: {dataset}")
        variables = spec["variables"]
        feature_conditions = [
            tuple(row[column] for column in variables) for row in feature_rows
        ]
        oracle_conditions = [
            tuple(row[column] for column in variables) for row in oracle_rows
        ]
        if feature_conditions != oracle_conditions or any(
            "Yield" not in row for row in oracle_rows
        ):
            raise ValueError(f"Oracle and outcome-free candidates are misaligned: {dataset}")

    protocol_tree = ast.parse((root / "code" / "protocol.py").read_text(encoding="utf-8"))
    dataset_files = None
    for node in protocol_tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "DATASET_FILES":
            dataset_files = ast.literal_eval(node.value)
            break
    if dataset_files is None or any(
        set(files) != {"train", "candidates", "oracle"}
        or any("searchspace" in value.lower() for value in files.values())
        for files in dataset_files.values()
    ):
        raise ValueError("Production data mapping must not expose searchspace.csv")

    runtime_metadata = json.loads(
        (root / "runtime" / "portable_environment.json").read_text(encoding="utf-8")
    )
    runtime_archive = root / "runtime" / runtime_metadata["archive"]
    if (
        runtime_metadata.get("platform") != "linux-x86_64"
        or runtime_metadata.get("offline") is not True
        or set(runtime_metadata.get("supported_launchers", {}))
        != {
            "linux_x86_64",
            "linux_aarch64",
            "macos_x86_64_or_arm64",
            "windows_x86_64",
        }
        or runtime_archive.stat().st_size != int(runtime_metadata["bytes"])
        or sha256(runtime_archive) != runtime_metadata["sha256"]
    ):
        raise ValueError("Portable runtime metadata or archive hash is invalid")

    openscience_metadata = json.loads(
        (root / "runtime" / "openscience" / "runtime.json").read_text(encoding="utf-8")
    )
    openscience_archive = root / "runtime" / "openscience" / openscience_metadata["archive"]
    if (
        openscience_metadata.get("name") != "@synsci/openscience-linux-x64"
        or openscience_metadata.get("version") != "2.0.23"
        or openscience_metadata.get("platform") != "linux-x86_64"
        or openscience_metadata.get("offline_bootstrap") is not True
        or openscience_archive.stat().st_size
        != int(openscience_metadata["archive_bytes"])
        or sha256(openscience_archive) != openscience_metadata["archive_sha256"]
    ):
        raise ValueError("Open Science runtime metadata or archive hash is invalid")
    with tarfile.open(openscience_archive, "r:gz") as archive:
        package_json = json.loads(
            archive.extractfile("package/package.json").read().decode("utf-8")
        )
        binary_member = archive.getmember(openscience_metadata["binary_member"])
        binary_digest = hashlib.sha256()
        with archive.extractfile(binary_member) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                binary_digest.update(chunk)
    if (
        package_json.get("name") != openscience_metadata["name"]
        or package_json.get("version") != openscience_metadata["version"]
        or binary_member.size != int(openscience_metadata["binary_bytes"])
        or binary_digest.hexdigest() != openscience_metadata["binary_sha256"]
    ):
        raise ValueError("Open Science package contents failed verification")
    openscience_config = json.loads(
        (root / "openscience.json").read_text(encoding="utf-8")
    )
    if (
        openscience_config.get("default_agent") != "catalystflow"
        or "deepseek" not in openscience_config.get("provider", {})
        or openscience_config.get("mcp", {}).get("catalystflow", {}).get("enabled")
        is not True
    ):
        raise ValueError("Open Science default Agent, provider, or MCP binding is invalid")
    launcher_checks = {
        "runtime/run_reproduction.sh": (
            "bootstrap_portable_env.sh",
            "code/main/run_selected.py",
        ),
        "runtime/run_openscience.sh": (
            "cd \"$package_root\"",
            "bootstrap_openscience.sh",
            "exec \"$engine\" web",
        ),
        "runtime/configure_deepseek.sh": (
            "bootstrap_openscience.sh",
            "keys add deepseek",
            "verify_deepseek.sh",
        ),
        "runtime/verify_deepseek.sh": (
            "deepseek/deepseek-v4-flash",
            "catalystflow_agent_status",
            "AUTH_OK",
        ),
        "runtime/openscience_cli.sh": (
            "bootstrap_openscience.sh",
            "exec \"$engine\" \"$@\"",
        ),
        "runtime/verify_openscience_runtime.sh": (
            "agent list",
            "catalystflow (primary)",
        ),
        "runtime/verify_offline_bundle.sh": (
            "bootstrap_portable_env.sh",
            "verify_openscience_runtime.sh",
        ),
        "runtime/run_reproduction_linux_arm64.sh": (
            "linux-aarch64",
            "environment-runtime.yml",
            "smoke_test_environment.py",
        ),
        "runtime/run_reproduction_macos.sh": (
            "osx-arm64",
            "osx-64",
            "environment-runtime.yml",
            "smoke_test_environment.py",
        ),
        "runtime/run_reproduction_windows.ps1": (
            "win-64",
            "environment-runtime.yml",
            "smoke_test_environment.py",
            "code/main/run_selected.py",
        ),
    }
    for relative, required_tokens in launcher_checks.items():
        path = root / relative
        content = path.read_text(encoding="utf-8")
        if any(token not in content for token in required_tokens):
            raise ValueError(f"Cross-platform launcher is incomplete: {relative}")
        if path.suffix == ".sh" and not os.access(path, os.X_OK):
            raise ValueError(f"Shell launcher is not executable: {relative}")

    for dataset in DATASETS:
        trajectory_root = root / "results" / "optimization_trajectories" / dataset
        paths = sorted(trajectory_root.glob("seed_*.pt"))
        if len(paths) != 20:
            raise ValueError(f"{dataset}: expected 20 trajectories, found {len(paths)}")
        for seed in SEEDS:
            path = trajectory_root / f"seed_{seed}.pt"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("seed") != seed or payload.get("dataset") != dataset:
                raise ValueError(f"Trajectory metadata mismatch: {path}")
            records = payload.get("trajectory")
            if not isinstance(records, list) or len(records) != 40:
                raise ValueError(f"Trajectory must contain exactly 40 rounds: {path}")
            if [record.get("step") for record in records] != list(range(1, 41)):
                raise ValueError(f"Non-consecutive steps: {path}")
            if len({record.get("query_index") for record in records}) != 40:
                raise ValueError(f"Repeated query index: {path}")
            if any(set(record) != ROUND_FIELDS for record in records):
                raise ValueError(f"Unexpected or missing trajectory field: {path}")

    with (root / "results" / "summary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 8:
        raise ValueError("summary_metrics.csv must contain 2 datasets × 4 metrics")

    unified_report = json.loads(
        (root / "results" / "results_and_analysis.json").read_text(encoding="utf-8")
    )
    trajectory_audit = unified_report.get("reproduction", {}).get("trajectory_audit", {})
    comparison = unified_report.get("reproduction", {}).get("reference_comparison", {})
    feasibility = unified_report.get("feasibility_assessment", {})
    if (
        trajectory_audit.get("status") != "valid"
        or trajectory_audit.get("trajectory_files") != 40
        or trajectory_audit.get("query_records") != 1600
        or comparison.get("status") != "identical"
        or comparison.get("identical") != 40
        or comparison.get("different") != 0
        or feasibility.get("owner") != "DeepSeek"
        or feasibility.get("ranking_effect") != "none"
        or len(feasibility.get("dimensions", [])) != 6
        or set(unified_report.get("final_recommended_reaction_conditions", {}))
        != set(DATASETS)
    ):
        raise ValueError("Unified results and agent analysis report is incomplete or inconsistent")
    external_evidence = json.loads(
        (root / "results" / "top5_external_evidence.json").read_text(encoding="utf-8")
    )
    sources = external_evidence.get("sources", [])
    source_ids = {item.get("source_id") for item in sources}
    candidate_source_map = external_evidence.get("candidate_source_map", {})
    if (
        len(sources) != 20
        or len(source_ids) != 20
        or external_evidence.get("ranking_effect") != "none"
        or set(candidate_source_map) != set(DATASETS)
        or any(
            set(candidate_source_map[dataset]) != {"1", "2", "3", "4", "5"}
            for dataset in DATASETS
        )
        or any(
            source_id not in source_ids
            for dataset in DATASETS
            for ids in candidate_source_map[dataset].values()
            for source_id in ids
        )
        or feasibility.get("evidence_retrieval", {}).get("source_count") != 20
    ):
        raise ValueError("Top-5 external evidence snapshot is incomplete or inconsistent")

    dft_root = root / "dft" / "orca_electronic_structure_database"
    dft_summary = json.loads((dft_root / "summary.json").read_text(encoding="utf-8"))
    with (dft_root / "calculation_index.csv").open(newline="", encoding="utf-8") as handle:
        dft_rows = sum(1 for _ in csv.DictReader(handle))
    if dft_rows != int(dft_summary["completed_total"]):
        raise ValueError("DFT calculation index and summary disagree")

    for audit_name in ("dft_boundary_audit.json", "mechanistic_dft_boundary_audit.json"):
        audit = json.loads((root / "results" / "audits" / audit_name).read_text(encoding="utf-8"))
        snapshot = root / audit["snapshot"]["path"]
        if not snapshot.is_file() or sha256(snapshot) != audit["snapshot"]["sha256"]:
            raise ValueError(f"DFT feature snapshot hash mismatch: {audit_name}")

    explanations = json.loads(
        (
            root
            / "results"
            / "dft_explanations"
            / "dft_explanation_examples.json"
        ).read_text(encoding="utf-8")
    )
    if (
        explanations.get("analysis_phase")
        != "post_hoc_after_model_and_trajectories_were_frozen"
        or explanations.get("ranking_effect") != "none"
    ):
        raise ValueError("DFT explanation boundary metadata is missing or invalid")
    explanation_text = json.dumps(explanations, ensure_ascii=False).lower()
    if '"yield"' in explanation_text:
        raise ValueError("DFT explanation feature payload must not contain a Yield field")
    with (
        root / "results" / "dft_explanations" / "dft_coverage_summary.csv"
    ).open(newline="", encoding="utf-8") as handle:
        coverage_rows = list(csv.DictReader(handle))
    complete = {
        row["dataset"]: float(row["coverage_fraction"])
        for row in coverage_rows
        if row["role"] == "complete_condition"
    }
    if set(complete) != set(DATASETS) or any(
        not 0.0 <= fraction <= 1.0 for fraction in complete.values()
    ):
        raise ValueError("DFT explanation coverage summary is incomplete")
    with (
        root / "results" / "dft_explanations" / "dft_posthoc_associations.csv"
    ).open(newline="", encoding="utf-8") as handle:
        association_rows = list(csv.DictReader(handle))
    if not association_rows or any(
        row["interpretation_scope"]
        != "post_hoc_descriptive_only_not_used_for_ranking"
        or not 0.0 <= float(row["q_value_bh"]) <= 1.0
        for row in association_rows
    ):
        raise ValueError("DFT post-hoc association boundary or q-values are invalid")

    for relative in (
        "doc/method_description.docx",
        "results/results_and_analysis.docx",
    ):
        with zipfile.ZipFile(root / relative) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt DOCX member {bad}: {relative}")
            media = [
                name for name in archive.namelist()
                if name.startswith("word/media/")
            ]
            relationships = archive.read("word/_rels/document.xml.rels").decode("utf-8")
            relationship_root = ET.fromstring(relationships)
            external_images = [
                relation
                for relation in relationship_root
                if relation.get("Type", "").endswith("/image")
                and relation.get("TargetMode") == "External"
            ]
            if external_images:
                raise ValueError(f"DOCX images must be embedded, not externally linked: {relative}")
            # A document without image references legitimately has no media
            # directory. If image relationships exist, every one must resolve
            # to an embedded media member.
            if "relationships/image" in relationships and not media:
                raise ValueError(f"DOCX image relationships lack embedded media: {relative}")
    for relative in (
        "doc/method_description.pdf",
        "results/results_and_analysis.pdf",
        "doc/resume.pdf",
    ):
        if not (root / relative).read_bytes().startswith(b"%PDF"):
            raise ValueError(f"Invalid PDF: {relative}")

    print(f"Verified CatalystFlow submission: {root}")
    print(
        f"Manifest files: {len(recorded)}; primary trajectories: 40; "
        f"DFT calculations: {dft_rows}; official public datasets: complete and isolated; "
        "numerical and Open Science runtime archives: verified"
    )


if __name__ == "__main__":
    main()
