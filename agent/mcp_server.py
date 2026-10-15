#!/usr/bin/env python3
"""Stdio MCP server exposing CatalystFlow Skill v2 policy-gated workflows."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

from policy_gateway import PolicyGateway, error_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = PROJECT_ROOT / "skill" / "catalystflow" / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from workflow_gateway import (  # noqa: E402
    FEASIBILITY_DIMENSIONS,
    WorkflowGateway,
    error_payload as workflow_error_payload,
)


POLICY = PolicyGateway()
WORKFLOW = WorkflowGateway()
PROTOCOL_VERSION = "2024-11-05"


def _source_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["type", "title", "locator", "retrieved_at", "claim"],
        "properties": {
            "type": {"type": "string", "enum": [
                "official_sds_or_ghs", "primary_literature", "chemical_registry",
                "supplier_catalog", "project_artifact",
            ]},
            "title": {"type": "string", "minLength": 1, "maxLength": 160},
            "locator": {"type": "string", "minLength": 1, "maxLength": 300},
            "retrieved_at": {"type": "string", "minLength": 1},
            "claim": {"type": "string", "minLength": 1, "maxLength": 160},
        },
        "additionalProperties": False,
    }


def _dimension_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["level", "basis", "confidence", "rationale", "sources"],
        "properties": {
            "level": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "basis": {"type": "string", "enum": ["verified_evidence", "llm_inference", "unknown"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 96},
            "sources": {"type": "array", "items": _source_schema()},
        },
        "additionalProperties": False,
    }


def _context_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["target_scale", "location", "assessment_date"],
        "properties": {
            "target_scale": {"type": "string", "minLength": 1},
            "location": {"type": "string", "minLength": 1},
            "assessment_date": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"},
        },
        "additionalProperties": False,
    }


def _candidate_schema(*, official: bool) -> dict[str, Any]:
    required = ["optimizer_rank", *FEASIBILITY_DIMENSIONS, "candidate_summary", "critical_safety_flags"]
    properties: dict[str, Any] = {
        "optimizer_rank": {"type": "integer", "minimum": 1, "maximum": 5},
        **{dimension: _dimension_schema() for dimension in FEASIBILITY_DIMENSIONS},
        "candidate_summary": {"type": "string", "minLength": 1, "maxLength": 160},
        "critical_safety_flags": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 96},
        },
    }
    if official:
        required.insert(0, "dataset")
        properties["dataset"] = {"type": "string", "enum": ["Suzuki", "Buchwald_sub4"]}
    return {"type": "object", "required": required, "properties": properties, "additionalProperties": False}


def _final_schema(*, official: bool) -> dict[str, Any]:
    required = [
        "recommendation_status", "selected_optimizer_rank", "rationale", "tradeoffs",
        "alternative_optimizer_rank", "alternative_rationale", "unresolved_information",
        "required_checks",
    ]
    nullable_rank = {"anyOf": [
        {"type": "integer", "minimum": 1, "maximum": 5}, {"type": "null"}
    ]}
    properties: dict[str, Any] = {
        "recommendation_status": {"type": "string", "enum": ["recommended", "no_condition_recommended"]},
        "selected_optimizer_rank": nullable_rank,
        "rationale": {"type": "string", "minLength": 1, "maxLength": 200},
        "tradeoffs": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "minLength": 1, "maxLength": 160}},
        "alternative_optimizer_rank": nullable_rank,
        "alternative_rationale": {"type": "string", "minLength": 1, "maxLength": 160},
        "unresolved_information": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string", "minLength": 1, "maxLength": 160}},
        "required_checks": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string", "minLength": 1, "maxLength": 160}},
    }
    if official:
        required.insert(0, "dataset")
        properties["dataset"] = {"type": "string", "enum": ["Suzuki", "Buchwald_sub4"]}
    return {"type": "object", "required": required, "properties": properties, "additionalProperties": False}


TOOLS = [
    {
        "name": "agent_status",
        "description": (
            "Load the CatalystFlow workflow context and return Skill v2 capabilities, the official 2x20x40 protocol, "
            "runtime readiness, and the DeepSeek/oracle data boundary."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "inspect_input",
        "description": (
            "Classify and audit a project-relative read-only input path. Use packaged data for "
            "the official run; project-contained unknown CSV directories return metadata for a "
            "DeepSeek manifest draft. Oracle values are never returned."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["input_path"],
            "properties": {"input_path": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
    },
    {
        "name": "start_reproduction",
        "description": (
            "Start a fresh reproduction using the final fixed model configuration: Suzuki and Buchwald, "
            "20 fixed seeds each, 40 sequential rounds each. Inline mode (default) keeps "
            "the tool call alive through report completion; detached mode requires a host "
            "that preserves child processes after the client exits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_path": {"type": "string", "default": "data"},
                "compare_reference": {"type": "boolean", "default": True},
                "execution_mode": {
                    "type": "string",
                    "enum": ["inline", "detached"],
                    "default": "inline",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "reproduction_status",
        "description": (
            "Poll a reproduction job. Completion is reported only after 40 trajectory files, "
            "1600 records, metrics, report generation, and optional reference comparison. "
            "Completed jobs directly return the bounded B1, B40, t95, and AUC summary "
            "(20-run mean, sample standard deviation, and 95% confidence interval)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
    },
    {
        "name": "prepare_feasibility_review",
        "description": (
            "Return the immutable queried Top-5, DFT context, and the required template "
            "for a DeepSeek-led six-dimension real-world feasibility assessment."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
    },
    {
        "name": "new_dataset",
        "description": (
            "Accept a DeepSeek manifest draft for a scanned directory, profile a manifest, or perform "
            "one stateful init/recommend/feedback/status operation. Deterministic validation precedes init."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["draft_manifest", "profile", "init", "recommend", "feedback", "status"],
                },
                "manifest_path": {"type": "string"},
                "input_path": {"type": "string"},
                "manifest_draft": {"type": "object"},
                "run_id": {"type": "string"},
                "observed_yield": {"type": "number", "minimum": 0, "maximum": 100},
                "transaction_id": {"type": "string"},
                "rounds": {"type": "integer", "minimum": 5, "maximum": 200, "default": 40},
                "seed": {"type": "integer", "minimum": 0, "default": 100},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize_live_feasibility",
        "description": (
            "Validate a DeepSeek-led six-dimension assessment for a completed new-dataset "
            "queried Top-5 and persist the final recommended reaction condition."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["run_id", "assessment"],
            "properties": {
                "run_id": {"type": "string", "minLength": 1},
                "assessment": {
                    "type": "object",
                    "required": ["assessment_context", "candidate_assessments", "final_recommendation"],
                    "properties": {
                        "assessment_context": _context_schema(),
                        "candidate_assessments": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": _candidate_schema(official=False),
                        },
                        "final_recommendation": _final_schema(official=False),
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize_report",
        "description": (
            "Incrementally validate and save one queried Top-5 six-dimension assessment per call. "
            "After all ten are saved, call again with the two final_recommendations to assemble "
            "the enriched report without changing optimizer ranks. A legacy one-call assessment "
            "is accepted but should not be used by language models."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {
                "job_id": {"type": "string", "minLength": 1},
                "assessment_context": _context_schema(),
                "candidate_assessment": _candidate_schema(official=True),
                "final_recommendations": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": _final_schema(official=True),
                },
                "assessment": {
                    "type": "object",
                    "required": ["assessment_context", "candidate_assessments", "final_recommendations"],
                    "properties": {
                        "assessment_context": _context_schema(),
                        "candidate_assessments": {
                            "type": "array",
                            "minItems": 10,
                            "maxItems": 10,
                            "items": _candidate_schema(official=True),
                        },
                        "final_recommendations": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": _final_schema(official=True),
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "oneOf": [
                {"required": ["job_id", "assessment"]},
                {"required": ["job_id", "assessment_context", "candidate_assessment"]},
                {"required": ["job_id", "final_recommendations"]},
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve_structures",
        "description": (
            "Resolve reaction-role structures through approved cache/registry/RDKit. "
            "New registry results remain pending human review; never invent identifiers."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["reaction_family", "roles"],
            "properties": {
                "reaction_family": {"type": "string", "enum": ["Suzuki", "Buchwald_sub4"]},
                "roles": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "required": ["variable", "value"],
                        "properties": {
                            "variable": {"type": "string"},
                            "value": {"type": "string"},
                            "identifier_type": {"type": "string", "enum": ["name", "smiles", "inchi"]},
                            "identifier": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "offline": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    },
]


def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    calls: dict[str, Callable[..., dict[str, Any]]] = {
        "agent_status": WORKFLOW.status,
        "inspect_input": WORKFLOW.inspect_input,
        "start_reproduction": WORKFLOW.start_reproduction,
        "reproduction_status": WORKFLOW.reproduction_status,
        "prepare_feasibility_review": WORKFLOW.prepare_feasibility_review,
        "new_dataset": WORKFLOW.new_dataset,
        "finalize_live_feasibility": WORKFLOW.finalize_live_feasibility,
        "finalize_report": WORKFLOW.finalize_report,
        "resolve_structures": POLICY.resolve_structures,
    }
    if name not in calls:
        raise ValueError("unknown MCP tool")
    return calls[name](**arguments)


def _response(request_id: Any, result: Any = None, error: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        return _response(
            request_id,
            {
                "protocolVersion": requested or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "catalystflow-skill-gateway", "version": "2.2.1"},
            },
        )
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method == "ping":
        return _response(request_id, {})
    if method == "tools/list":
        return _response(request_id, {"tools": TOOLS})
    if method == "tools/call":
        try:
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            output = _dispatch(str(params.get("name", "")), arguments)
            text = json.dumps(output, ensure_ascii=False, allow_nan=False)
            return _response(
                request_id,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": output,
                    "isError": False,
                },
            )
        except Exception as exc:
            output = (
                workflow_error_payload(exc)
                if exc.__class__.__module__ == "workflow_gateway"
                else error_payload(exc)
            )
            return _response(
                request_id,
                {
                    "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False)}],
                    "structuredContent": output,
                    "isError": True,
                },
            )
    if request_id is None:
        return None
    return _response(request_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("JSON-RPC message must be an object")
            answer = handle(message)
            if answer is not None:
                sys.stdout.write(json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            error = _response(None, error={"code": -32700, "message": f"Invalid request: {type(exc).__name__}"})
            sys.stdout.write(json.dumps(error, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
