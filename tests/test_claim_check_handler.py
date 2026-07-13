import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.workflow import TemplateCoordinator


class ClaimCheckHandlerTests(unittest.TestCase):
    def test_claim_check_records_gaps_when_no_evidence_exists(self):
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmp:
            knowledge_dir = Path(tmp) / "knowledge"
            knowledge_dir.mkdir()

            config = AgentConfig.default(
                workspace_root=project_root
            )
            config.runtime["workflow_spec"] = (
                "configs/workflows/"
                "learning.claim-check.workflow.json"
            )
            config.knowledge["directory"] = str(
                knowledge_dir
            )
            config.knowledge["min_score"] = 0.0

            routing = {
                "source": "test",
                "difficulty": "medium",
                "workflow": "evidence_soft",
                "evidence_mode": "required",
                "evidence_strictness": "soft",
                "risk_level": "low",
                "category": "technical_claim",
                "sources": ["rag"],
                "reasons": [
                    "测试无证据时的结论检查"
                ],
                "confidence": 1.0,
            }

            result = TemplateCoordinator(
                config=config,
                workspace_root=project_root,
            ).run(
                "请根据可靠证据解释证据门禁机制。",
                routing_decision=routing,
            )

        self.assertIn(
            "claim_check",
            result.stage_trace,
        )

        claim_review = getattr(
            result.state,
            "claim_review",
            {},
        )

        self.assertTrue(
            claim_review,
            msg="claim_check 没有生成结构化检查结果",
        )

        self.assertTrue(
            claim_review.get("claims"),
            msg="没有从回答草稿中提取出结论",
        )

        self.assertEqual(
            claim_review.get(
                "claims_with_candidate_evidence"
            ),
            [],
            msg="没有检索结果时不应声称存在候选证据",
        )

        self.assertTrue(
            claim_review.get("unsupported_claims"),
            msg="无证据时应标记未支持结论",
        )

        self.assertTrue(
            claim_review.get("required_follow_up"),
            msg="无证据时应给出后续检索或审计动作",
        )


if __name__ == "__main__":
    unittest.main()
