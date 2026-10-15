#!/usr/bin/env python3
"""Build a resumable, provenance-rich structure map without model guesses."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem


PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OPSIN = "https://opsin.ch.cam.ac.uk/opsin"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def query_hash(row: dict[str, str]) -> str:
    raw = "\x1f".join(
        [row["variable"], row["value"], row["identifier_type"], row["identifier"]]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_identifier(value: str) -> None:
    if not value or len(value) > 1000:
        raise ValueError("identifier must contain 1..1000 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError("identifier contains control characters")


def molecule_from(identifier_type: str, identifier: str) -> Chem.Mol:
    if identifier_type == "smiles":
        molecule = Chem.MolFromSmiles(identifier)
    elif identifier_type == "inchi":
        molecule = Chem.MolFromInchi(identifier)
    else:
        raise ValueError("direct parsing accepts smiles or inchi")
    if molecule is None:
        raise ValueError(f"RDKit could not parse {identifier_type}")
    return molecule


def canonical_record(molecule: Chem.Mol) -> dict[str, str]:
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
        "inchi": Chem.MolToInchi(molecule),
        "inchikey": Chem.MolToInchiKey(molecule),
    }


def fetch_json(url: str, timeout: float, retries: int) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={"User-Agent": "CatalystFlow-CouplingResolver/1.0"},
            )
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"structure service failed after bounded retries: {error}")


def from_pubchem(name: str, timeout: float, retries: int) -> tuple[Chem.Mol, str]:
    encoded = quote(name, safe="")
    cid_payload = fetch_json(
        f"{PUBCHEM}/compound/name/{encoded}/cids/JSON", timeout, retries
    )
    cids = cid_payload.get("IdentifierList", {}).get("CID", [])
    if len(cids) != 1:
        raise ValueError(f"PubChem name returned {len(cids)} CIDs; review ambiguity")
    cid = str(cids[0])
    properties = fetch_json(
        (
            f"{PUBCHEM}/compound/cid/{cid}/property/"
            "SMILES,ConnectivitySMILES,InChI/JSON"
        ),
        timeout,
        retries,
    )
    rows = properties.get("PropertyTable", {}).get("Properties", [])
    if len(rows) != 1:
        raise ValueError("PubChem property response was not unique")
    record = rows[0]
    identifier = (
        record.get("SMILES")
        or record.get("IsomericSMILES")
        or record.get("ConnectivitySMILES")
        or record.get("InChI")
    )
    if not identifier:
        raise ValueError("PubChem returned no parseable structure")
    molecule = (
        Chem.MolFromInchi(identifier)
        if str(identifier).startswith("InChI=")
        else Chem.MolFromSmiles(identifier)
    )
    if molecule is None:
        raise ValueError("RDKit could not parse the PubChem structure")
    return molecule, cid


def from_opsin(name: str, timeout: float, retries: int) -> Chem.Mol:
    payload = fetch_json(f"{OPSIN}/{quote(name, safe='')}.json", timeout, retries)
    smiles = payload.get("smiles")
    molecule = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if molecule is None:
        raise ValueError("OPSIN returned no parseable structure")
    return molecule


def load_input(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        if not {"variable", "value"}.issubset(headers):
            raise ValueError("input requires variable,value columns")
        rows = []
        seen: set[tuple[str, str]] = set()
        for raw in reader:
            row = {key: str(value or "").strip() for key, value in raw.items()}
            row["identifier_type"] = row.get("identifier_type", "name").lower()
            row["identifier"] = row.get("identifier") or row["value"]
            if row["identifier_type"] not in {"name", "smiles", "inchi"}:
                raise ValueError("identifier_type must be name, smiles, or inchi")
            if not row["variable"] or not row["value"]:
                raise ValueError("variable and value cannot be empty")
            validate_identifier(row["identifier"])
            key = (row["variable"], row["value"])
            if key in seen:
                raise ValueError(f"duplicate structure key: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("structure input is empty")
    return rows


def load_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def resolve_record(
    row: dict[str, str],
    cache: dict[str, dict[str, object]],
    *,
    offline: bool,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    digest = query_hash(row)
    base: dict[str, object] = {
        "variable": row["variable"],
        "value": row["value"],
        "identifier_type": row["identifier_type"],
        "query_sha256": digest,
        "retrieved_at": utc_now(),
    }
    cached = cache.get(digest)
    if cached and str(cached.get("status", "")).startswith("resolved"):
        cached_review = str(cached.get("review_status", "")).lower()
        if cached_review in {"approved", "approved_before_agent_run"}:
            return {**cached, "status": "resolved_cache"}
        return {
            **cached,
            "status": "resolved_pending_review",
            "review_status": "pending_human_review",
        }

    errors: list[str] = []
    molecule: Chem.Mol | None = None
    source = ""
    source_id = ""
    try:
        if row["identifier_type"] in {"smiles", "inchi"}:
            molecule = molecule_from(row["identifier_type"], row["identifier"])
            source = "input"
        elif offline:
            errors.append("offline mode: name was not found in the successful cache")
        else:
            try:
                molecule, source_id = from_pubchem(
                    row["identifier"], timeout, retries
                )
                source = "pubchem"
            except (RuntimeError, ValueError) as exc:
                if "review ambiguity" in str(exc):
                    raise
                errors.append(str(exc))
                molecule = from_opsin(row["identifier"], timeout, retries)
                source = "opsin"
        if molecule is not None:
            canonical = canonical_record(molecule)
            if "." in canonical["canonical_smiles"]:
                return {
                    **base,
                    **canonical,
                    "source": source,
                    "source_id": source_id,
                    "status": "review_required",
                    "errors": errors
                    + ["disconnected fragments may represent a salt, solvate, or mixture"],
                }
            return {
                **base,
                **canonical,
                "source": source,
                "source_id": source_id,
                "status": f"resolved_{source}",
                "review_status": (
                    "approved_before_agent_run" if source == "input" else "pending_human_review"
                ),
                "errors": errors,
            }
    except (RuntimeError, ValueError) as exc:
        errors.append(str(exc))
    return {
        **base,
        "canonical_smiles": "",
        "inchi": "",
        "inchikey": "",
        "source": "",
        "source_id": "",
        "status": "unresolved",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-unresolved", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 60:
        raise ValueError("--timeout must be in 1..60")
    if not 0 <= args.retries <= 3:
        raise ValueError("--retries must be in 0..3")

    output = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    cache = load_cache(cache_path)
    records = []
    for row in load_input(Path(args.input).resolve()):
        record = resolve_record(
            row,
            cache,
            offline=args.offline,
            timeout=args.timeout,
            retries=args.retries,
        )
        records.append(record)
        cache[str(record["query_sha256"])] = record
        atomic_json(
            cache_path,
            {"schema_version": 1, "updated_at": utc_now(), "entries": cache},
        )

    resolved = sum(
        str(record["status"]).startswith("resolved") for record in records
    )
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "generator": "optimize-coupling-reactions/resolve_structures.py",
        "records": records,
        "audit": {
            "total": len(records),
            "resolved": resolved,
            "unresolved": len(records) - resolved,
            "offline": args.offline,
        },
    }
    atomic_json(output, payload)
    print(json.dumps(payload["audit"], indent=2, ensure_ascii=False))
    if resolved != len(records) and not args.allow_unresolved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
