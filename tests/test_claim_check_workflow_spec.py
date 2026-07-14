import json
import unittest
from pathlib import Path

from reasoning_agent_template.workflow_spec import WorkflowSpec


class ClaimCheckWorkflowSpecTests(unittest.TestCase):
    def test_claim_check_is_between_reason_and_evidence_audit(self):
        base_path = Path(
            "configs/workflows/default.workflow.json"
        )
        candidate_path = Path(
            "configs/workflows/learning.claim-check.workflow.json"
        )

        base = WorkflowSpec.from_dict(
            json.loads(base_path.read_text(encoding="utf-8"))
        )
        candidate = WorkflowSpec.from_dict(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        )

        validation = candidate.validate(base=base)

        self.assertTrue(
            validation.ok,
            msg={
                "errors": validation.errors,
                "warnings": validation.warnings,
                "requires_code": validation.requires_code,
            },
        )

        node_ids = candidate.node_ids()

        reason_index = node_ids.index("reason")
        claim_index = node_ids.index("claim_check")
        audit_index = node_ids.index("evidence_audit")

        self.assertEqual(claim_index, reason_index + 1)
        self.assertEqual(audit_index, claim_index + 1)

        claim_node = candidate.node_map()["claim_check"]

        self.assertEqual(claim_node.agent, "critic")
        self.assertEqual(claim_node.handler_kind, "builtin")
        self.assertEqual(claim_node.handler, "claim_check")
        self.assertTrue(claim_node.checkpoint)

        edge_pairs = {
            (edge.from_node, edge.to_node)
            for edge in candidate.edges
        }

        self.assertNotIn(
            ("reason", "evidence_audit"),
            edge_pairs,
        )
        self.assertIn(
            ("reason", "claim_check"),
            edge_pairs,
        )
        self.assertIn(
            ("claim_check", "evidence_audit"),
            edge_pairs,
        )


if __name__ == "__main__":
    unittest.main()
