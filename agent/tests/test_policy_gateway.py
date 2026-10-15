from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest


AGENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGENT_ROOT.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from policy_gateway import PolicyGateway  # noqa: E402


class PolicyGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = Path(tempfile.mkdtemp(prefix=".agent-unit-", dir=PROJECT_ROOT))
        self.gateway = PolicyGateway(PROJECT_ROOT, self.runtime)

    def tearDown(self) -> None:
        shutil.rmtree(self.runtime)

    def test_only_structure_resolution_mode_remains(self) -> None:
        self.assertEqual(self.gateway.status()["modes"], ["structure_resolution"])

    def test_direct_smiles_is_validated_but_held_for_review(self) -> None:
        packet = self.gateway.resolve_structures(
            "Suzuki",
            [
                {
                    "variable": "Ligand",
                    "value": "ethanol-example",
                    "identifier_type": "smiles",
                    "identifier": "CCO",
                }
            ],
            offline=True,
        )
        self.assertFalse(packet["all_usable"])
        self.assertEqual(packet["resolved_roles"][0]["canonical_smiles"], "CCO")
        self.assertEqual(packet["resolved_roles"][0]["review_status"], "pending_human_review")
        output = {
            key: packet[key]
            for key in (
                "mode",
                "request_id",
                "status",
                "resolved_roles",
                "all_usable",
                "blocking_reasons",
                "next_action",
            )
        }
        accepted = self.gateway.validate_output(
            "structure_resolution", packet["request_id"], output
        )
        self.assertTrue(accepted["valid"])

    def test_disconnected_structure_requires_review(self) -> None:
        packet = self.gateway.resolve_structures(
            "Suzuki",
            [
                {
                    "variable": "Base",
                    "value": "salt-example",
                    "identifier_type": "smiles",
                    "identifier": "[Na+].[OH-]",
                }
            ],
            offline=True,
        )
        self.assertEqual(packet["resolved_roles"][0]["status"], "review_required")
        self.assertFalse(packet["all_usable"])


if __name__ == "__main__":
    unittest.main()
