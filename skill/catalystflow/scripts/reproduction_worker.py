#!/usr/bin/env python3
"""Durable worker for a full two-dataset CatalystFlow reproduction job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_status(job_dir: Path, **fields: Any) -> None:
    path = job_dir / "status.json"
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(fields)
    current["updated_at_utc"] = utc_now()
    current["pid"] = os.getpid()
    atomic_json(path, current)


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    heartbeat: Callable[[], None] | None = None,
) -> None:
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                return_code = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if heartbeat is not None:
                    heartbeat()
    if return_code:
        raise RuntimeError(f"command failed with exit code {return_code}: {command[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--compare-reference", choices=("yes", "no"), default="yes")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    data_root = Path(args.data_root).resolve()
    try:
        write_status(
            job_dir,
            status="running",
            stage="environment_bootstrap",
            started_at_utc=utc_now(),
        )
        runtime_root = project_root / "runtime" / ".portable_env"
        runtime_python = runtime_root / "bin" / "python"
        if not runtime_python.is_file():
            run_logged(
                [
                    "bash",
                    str(project_root / "runtime" / "bootstrap_portable_env.sh"),
                    str(runtime_root),
                ],
                project_root,
                job_dir / "environment.log",
                heartbeat=lambda: write_status(job_dir),
            )
        if not runtime_python.is_file():
            raise RuntimeError("portable environment was not created")

        write_status(job_dir, status="running", stage="optimization_2x20x40")
        run_root = job_dir / "run"
        run_logged(
            [
                str(runtime_python),
                str(project_root / "code" / "main" / "run_selected.py"),
                "--data-root", str(data_root),
                "--results-root", str(run_root),
                "--dataset", "all",
                "--rounds", "40",
                "--seeds", "20",
            ],
            project_root,
            job_dir / "optimizer.log",
            heartbeat=lambda: write_status(job_dir),
        )

        write_status(job_dir, status="running", stage="trajectory_audit_and_report")
        report_root = job_dir / "report"
        command = [
            str(runtime_python),
            str(Path(__file__).with_name("report_builder.py")),
            "--project-root", str(project_root),
            "--data-root", str(data_root),
            "--run-root", str(run_root),
            "--report-root", str(report_root),
        ]
        if args.compare_reference == "yes":
            command += ["--reference-root", str(project_root / "results")]
        run_logged(
            command,
            project_root,
            job_dir / "report.log",
            heartbeat=lambda: write_status(job_dir),
        )
        report = json.loads((report_root / "report.json").read_text(encoding="utf-8"))
        write_status(
            job_dir,
            status="completed",
            stage="completed",
            completed_at_utc=utc_now(),
            trajectory_files=report["trajectory_audit"]["trajectory_files"],
            query_records=report["trajectory_audit"]["query_records"],
            comparison_status=report["reference_comparison"]["status"],
            feasibility_status="awaiting_llm_assessment",
            next_action="catalystflow_prepare_feasibility_review",
        )
    except Exception as exc:
        traceback_path = job_dir / "failure.txt"
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_status(
            job_dir,
            status="failed",
            stage="failed",
            failed_at_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
            failure_log=str(traceback_path),
        )
        raise


if __name__ == "__main__":
    main()
