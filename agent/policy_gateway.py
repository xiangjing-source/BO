#!/usr/bin/env python3
"""Policy-enforced structure gateway for the CatalystFlow Open Science agent.

Queried Top-5 feasibility assessment belongs to ``WorkflowGateway`` so that official
reproduction and live optimization share one DeepSeek-led report contract.  This
module now has one bounded responsibility: resolve and validate structures.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROUNDS = 40
SUPPORTED_DATASETS = {"Suzuki", "Buchwald_sub4"}
ALLOWED_STRUCTURE_ROLES = {
    "Suzuki": {"Electrophile", "Nucleophile", "Ligand", "Base", "Solvent", "Product"},
    "Buchwald_sub4": {"Reactant2", "Ligand", "Additive", "Base", "Product"},
}


class PolicyError(ValueError):
    """A deterministic boundary or validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _plain(value: Any) -> Any:
    """Convert common scalar containers to strict, finite JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return _plain(value.item())
    raise PolicyError("non_json_value", f"Unsupported value type: {type(value).__name__}")


class PolicyGateway:
    """Bounded, auditable operations offered to the Open Science agent."""

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        runtime_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runtime_root = Path(
            runtime_root or self.project_root / ".catalystflow-agent"
        ).resolve()
        self.requests_dir = self.runtime_root / "requests"
        self.outputs_dir = self.runtime_root / "validated_outputs"
        self.audit_path = self.runtime_root / "audit.jsonl"

    def status(self) -> dict[str, Any]:
        return {
            "agent": "catalystflow",
            "policy_version": "1.0.0",
            "expected_rounds": EXPECTED_ROUNDS,
            "supported_datasets": sorted(SUPPORTED_DATASETS),
            "modes": ["structure_resolution"],
            "core_tools_disabled": True,
            "oracle_path_access": "denied",
            "runtime_root": str(self.runtime_root.relative_to(self.project_root)),
        }

    def _audit(self, action: str, request_id: str, input_data: Any, output: Any) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": utc_now(),
            "request_id": request_id,
            "action": action,
            "input_sha256": _sha256(input_data),
            "output_sha256": _sha256(output),
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _store_request(self, request_id: str, packet: Mapping[str, Any]) -> None:
        _atomic_json(self.requests_dir / f"{request_id}.json", dict(packet))

    def _load_request(self, request_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise PolicyError("invalid_request_id", "request_id format is invalid")
        path = self.requests_dir / f"{request_id}.json"
        if not path.is_file():
            raise PolicyError("request_not_found", "No immutable request packet exists")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PolicyError("invalid_request_packet", "Stored request is not an object")
        return payload

    def _load_fixed_structures(self) -> dict[tuple[str, str], dict[str, Any]]:
        path = self.project_root / "code" / "config" / "buchwald_structures.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return {
            (str(row.get("variable")), str(row.get("value"))): row
            for row in records
            if isinstance(row, dict)
            and str(row.get("status", "")).startswith("resolved")
            and str(row.get("canonical_smiles", "")).strip()
            and (
                str(row.get("source_url", "")).strip()
                or str(row.get("status", "")) == "resolved_direct_smiles"
            )
        }

    @staticmethod
    def _resolver_module(project_root: Path) -> Any:
        path = (
            project_root
            / "skill"
            / "optimize-coupling-reactions"
            / "scripts"
            / "resolve_structures.py"
        )
        spec = importlib.util.spec_from_file_location("catalystflow_structure_resolver", path)
        if spec is None or spec.loader is None:
            raise PolicyError("resolver_unavailable", "Packaged structure resolver is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def resolve_structures(
        self,
        reaction_family: str,
        roles: Sequence[Mapping[str, Any]],
        offline: bool = False,
    ) -> dict[str, Any]:
        if reaction_family not in SUPPORTED_DATASETS:
            raise PolicyError("unsupported_dataset", "Unsupported reaction family")
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)) or not roles:
            raise PolicyError("invalid_roles", "roles must be a non-empty array")
        if len(roles) > 32:
            raise PolicyError("too_many_roles", "At most 32 structures may be resolved per request")
        fixed = self._load_fixed_structures()
        resolver = self._resolver_module(self.project_root)
        results: list[dict[str, Any]] = []
        allowed = ALLOWED_STRUCTURE_ROLES[reaction_family]
        for item in roles:
            if not isinstance(item, Mapping):
                raise PolicyError("invalid_role", "Each role must be an object")
            variable = str(item.get("variable", "")).strip()
            value = str(item.get("value", "")).strip()
            identifier_type = str(item.get("identifier_type", "name")).strip().lower()
            identifier = str(item.get("identifier", value)).strip()
            if variable not in allowed:
                raise PolicyError("invalid_role", f"Role {variable!r} is not valid for {reaction_family}")
            if not value or len(value) > 1000 or not identifier:
                raise PolicyError("invalid_identifier", "Structure value/identifier is invalid")
            if identifier_type not in {"name", "smiles", "inchi"}:
                raise PolicyError("invalid_identifier_type", "Use name, smiles, or inchi")
            preapproved = fixed.get((variable, value))
            if preapproved:
                molecule = resolver.Chem.MolFromSmiles(preapproved["canonical_smiles"])
                if molecule is None:
                    raise PolicyError(
                        "invalid_approved_cache",
                        "An approved cache structure failed current RDKit validation",
                    )
                results.append(
                    {
                        "variable": variable,
                        "role": variable,
                        "value": value,
                        "raw_name": value,
                        "normalized_name": value,
                        "status": "resolved_approved_cache",
                        "canonical_smiles": preapproved["canonical_smiles"],
                        "source": preapproved.get("source_url") or "packaged_direct_smiles",
                        "source_identifier": str(preapproved.get("cid", "")),
                        "retrieved_at": "packaged_before_agent_run",
                        "validation": {
                            "parse_ok": True,
                            "formal_charge": sum(
                                atom.GetFormalCharge() for atom in molecule.GetAtoms()
                            ),
                            "disconnected_fragments": "." in preapproved["canonical_smiles"],
                            "validator": f"RDKit {resolver.Chem.rdBase.rdkitVersion}",
                        },
                        "confidence": "high",
                        "review_status": "approved_before_agent_run",
                        "cache_event": "approved_cache_hit",
                    }
                )
                continue
            row = {
                "variable": variable,
                "value": value,
                "identifier_type": identifier_type,
                "identifier": identifier,
            }
            resolved = resolver.resolve_record(
                row,
                {},
                offline=bool(offline),
                timeout=10.0,
                retries=1,
            )
            status = str(resolved.get("status", "unresolved"))
            if status.startswith("resolved"):
                review_status = "pending_human_review"
                outward_status = "resolved_pending_review"
            else:
                review_status = "required"
                outward_status = status
            canonical_smiles = str(resolved.get("canonical_smiles", ""))
            molecule = resolver.Chem.MolFromSmiles(canonical_smiles) if canonical_smiles else None
            confidence = (
                "high"
                if outward_status == "resolved_pending_review" and identifier_type in {"smiles", "inchi"}
                else "medium"
                if outward_status == "resolved_pending_review"
                else "low"
            )
            results.append(
                {
                    "variable": variable,
                    "role": variable,
                    "value": value,
                    "raw_name": value,
                    "normalized_name": value,
                    "status": outward_status,
                    "canonical_smiles": canonical_smiles,
                    "inchi": resolved.get("inchi", ""),
                    "inchikey": resolved.get("inchikey", ""),
                    "source": resolved.get("source", ""),
                    "source_identifier": resolved.get("source_id", ""),
                    "retrieved_at": resolved.get("retrieved_at"),
                    "validation": {
                        "parse_ok": molecule is not None,
                        "formal_charge": (
                            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
                            if molecule is not None
                            else None
                        ),
                        "disconnected_fragments": "." in canonical_smiles,
                        "validator": f"RDKit {resolver.Chem.rdBase.rdkitVersion}",
                    },
                    "confidence": confidence,
                    "errors": resolved.get("errors", []),
                    "query_sha256": resolved.get("query_sha256"),
                    "review_status": review_status,
                    "cache_event": (
                        "pending_cache_write"
                        if outward_status == "resolved_pending_review"
                        else "not_cached"
                    ),
                }
            )
        request_id = uuid.uuid4().hex
        all_usable = all(row["status"] == "resolved_approved_cache" for row in results)
        all_resolved = all(
            row["status"] in {"resolved_approved_cache", "resolved_pending_review"}
            for row in results
        )
        if all_usable:
            overall_status = "cache_hit"
            next_action = "run_existing_feature_builder"
        elif all_resolved:
            overall_status = "resolved_pending_cache_review"
            next_action = "request_human_review"
        else:
            overall_status = "manual_review_required"
            next_action = "request_human_review"
        blocking_reasons = [
            {
                "role": row["role"],
                "value": row["value"],
                "status": row["status"],
                "errors": row.get("errors", []),
            }
            for row in results
            if row["status"] != "resolved_approved_cache"
        ]
        packet = {
            "schema_version": "1.0",
            "request_id": request_id,
            "mode": "structure_resolution",
            "reaction_family": reaction_family,
            "status": overall_status,
            "resolved_roles": results,
            "all_usable": all_usable,
            "blocking_reasons": blocking_reasons,
            "next_action": next_action,
            "rule": "Newly retrieved structures remain pending until human review",
            "created_at_utc": utc_now(),
        }
        pending = [row for row in results if row["cache_event"] == "pending_cache_write"]
        if pending:
            _atomic_json(
                self.runtime_root / "pending_structures" / f"{request_id}.json",
                {
                    "request_id": request_id,
                    "review_status": "pending_human_review",
                    "records": pending,
                },
            )
        self._store_request(request_id, packet)
        self._audit(
            "resolve_structures",
            request_id,
            {"reaction_family": reaction_family, "roles": _plain(roles), "offline": offline},
            packet,
        )
        return packet

    @staticmethod
    def _assert_exact(actual: Any, expected: Any, field: str) -> None:
        if actual != expected:
            raise PolicyError("immutable_field_changed", f"Immutable field changed: {field}")

    def _validate_structure(self, request: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        if set(payload) != {
            "mode",
            "request_id",
            "status",
            "resolved_roles",
            "all_usable",
            "blocking_reasons",
            "next_action",
        }:
            raise PolicyError("invalid_output_fields", "Structure output fields do not match schema")
        if payload.get("mode") != "structure_resolution":
            raise PolicyError("invalid_mode", "Output mode does not match request")
        self._assert_exact(payload.get("request_id"), request["request_id"], "request_id")
        self._assert_exact(
            payload.get("resolved_roles"), request.get("resolved_roles"), "resolved_roles"
        )
        self._assert_exact(payload.get("all_usable"), request.get("all_usable"), "all_usable")
        self._assert_exact(payload.get("status"), request.get("status"), "status")
        self._assert_exact(
            payload.get("blocking_reasons"), request.get("blocking_reasons"), "blocking_reasons"
        )
        self._assert_exact(payload.get("next_action"), request.get("next_action"), "next_action")

    def validate_output(
        self, mode: str, request_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise PolicyError("invalid_output", "payload must be a JSON object")
        request = self._load_request(request_id)
        if request.get("mode") != mode:
            raise PolicyError("invalid_mode", "Requested validation mode does not match packet")
        if mode == "structure_resolution":
            self._validate_structure(request, payload)
        else:
            raise PolicyError("invalid_mode", "Unsupported Agent mode")
        validated = {
            "valid": True,
            "request_id": request_id,
            "mode": mode,
            "validated_at_utc": utc_now(),
            "payload_sha256": _sha256(payload),
            "payload": _plain(payload),
        }
        _atomic_json(self.outputs_dir / f"{request_id}.json", validated)
        self._audit("validate_output", request_id, payload, validated)
        return validated


def error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PolicyError):
        return {"error": exc.code, "message": str(exc)}
    return {"error": "internal_error", "message": f"{type(exc).__name__}: operation failed"}
