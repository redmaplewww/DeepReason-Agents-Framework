import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.evidence import EvidenceLedger
from reasoning_agent_template.gates import GatePolicy


class EvidenceAndGateTests(unittest.TestCase):
    def test_evidence_items_are_hashed_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(Path(tmp) / "evidence.jsonl")

            item = ledger.record(
                source_type="file",
                uri="knowledge/source.md",
                locator="lines 1-3",
                content="Evidence-first reasoning needs stable source traces.",
                summary="Evidence-first reasoning needs source traces.",
                confidence=0.91,
                used_for=["claim:traceability"],
            )

            self.assertTrue(item.id.startswith("ev_"))
            self.assertEqual(len(item.content_hash), 64)
            self.assertEqual(item.used_for, ["claim:traceability"])
            self.assertEqual(ledger.list()[0].id, item.id)

    def test_duplicate_evidence_identity_is_not_appended_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(Path(tmp) / "evidence.jsonl")
            first = ledger.record(
                source_type="web",
                uri="https://example.test/source",
                locator="section 1",
                content="same source content",
                summary="summary",
                confidence=0.8,
            )
            second = ledger.record(
                source_type="web",
                uri="https://example.test/source",
                locator="section 1",
                content="same source content",
                summary="different summary",
                confidence=0.9,
            )
            rows = ledger.list()

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(rows), 1)

    def test_high_risk_actions_interrupt_without_evidence_and_allow_with_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            ledger = EvidenceLedger(workspace / "evidence.jsonl")
            policy = GatePolicy(
                workspace_root=workspace,
                min_evidence_by_risk={"high": 1},
                approval_required_actions={"write_file"},
            )

            blocked = policy.evaluate(
                action="write_file",
                risk_level="high",
                evidence=[],
                target_path=workspace / "agent.yaml",
            )

            self.assertEqual(blocked.status, "interrupt")
            self.assertIn("requires at least 1 evidence item", " ".join(blocked.reasons))

            evidence = ledger.record(
                source_type="tool",
                uri="tests",
                locator="test fixture",
                content="A human-approved config change is requested.",
                summary="Approved config change evidence.",
                confidence=1.0,
                used_for=["gate:write_file"],
            )
            allowed = policy.evaluate(
                action="write_file",
                risk_level="high",
                evidence=[evidence],
                target_path=workspace / "agent.yaml",
                approved_by="human",
            )

            self.assertEqual(allowed.status, "allow")
            self.assertEqual(allowed.approved_by, "human")

    def test_gate_denies_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            policy = GatePolicy(workspace_root=workspace)

            decision = policy.evaluate(
                action="write_file",
                risk_level="medium",
                evidence=[],
                target_path=workspace.parent / "outside.txt",
            )

            self.assertEqual(decision.status, "deny")
            self.assertIn("outside workspace", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
