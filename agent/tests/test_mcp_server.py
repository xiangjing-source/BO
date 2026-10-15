from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import unittest


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from mcp_server import handle  # noqa: E402


class MCPServerTests(unittest.TestCase):
    def test_initialize(self) -> None:
        response = handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
        self.assertEqual(response["result"]["serverInfo"]["name"], "catalystflow-skill-gateway")

    def test_only_bounded_skill_tools_are_listed(self) -> None:
        response = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertEqual(
            names,
            {
                "agent_status",
                "inspect_input",
                "start_reproduction",
                "reproduction_status",
                "prepare_feasibility_review",
                "new_dataset",
                "finalize_live_feasibility",
                "finalize_report",
                "resolve_structures",
            },
        )

    def test_status_tool_call(self) -> None:
        response = handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "agent_status", "arguments": {}},
            }
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["oracle_path_access"], "denied")

    def test_stdio_protocol_exposes_and_calls_status(self) -> None:
        messages = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "agent_status", "arguments": {}},
            },
        ]
        completed = subprocess.run(
            [sys.executable, str(AGENT_ROOT / "mcp_server.py")],
            input="\n".join(json.dumps(item) for item in messages) + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(responses[1]["result"]["tools"]), 9)
        self.assertEqual(
            responses[2]["result"]["structuredContent"]["skill_version"],
            "2.2.1",
        )


if __name__ == "__main__":
    unittest.main()
