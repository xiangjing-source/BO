#!/usr/bin/env python3
"""Atomically regenerate the exact package SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
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
        or (
            relative.parts[:2] == (".openscience", "handoffs")
            and path.suffix == ".md"
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or ignored(path):
            continue
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "package": "CatalystFlow",
        "format": "competition submission with Open Science Agent",
        "files": files,
    }
    destination = ROOT / "MANIFEST.json"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ROOT, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    print(f"Wrote {destination} with {len(files)} files")


if __name__ == "__main__":
    main()
