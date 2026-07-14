import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.models import (
    AgentState,
    KnowledgeChunk,
)
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

    def test_claim_check_does_not_bind_unrelated_claims_to_any_chunk(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "火星有两颗天然卫星。"
                "Python 列表是可变序列。"
            ),
            evidence_mode="required",
            retrieval_results=[
                KnowledgeChunk(
                    source="test-knowledge.md",
                    span="1-2",
                    text=(
                        "火星有两颗天然卫星，"
                        "分别是火卫一和火卫二。"
                    ),
                    content_hash="mars-hash",
                    score=1.0,
                    evidence_id="ev-mars",
                )
            ],
        )

        coordinator._claim_check(state)

        claim_review = state.claim_review

        self.assertEqual(
            claim_review.get("claims"),
            [
                "火星有两颗天然卫星",
                "Python 列表是可变序列",
            ],
        )

        self.assertEqual(
            claim_review.get(
                "claims_with_candidate_evidence"
            ),
            ["火星有两颗天然卫星"],
            msg=(
                "与火星有关的证据不应被绑定到 "
                "Python 列表结论"
            ),
        )

        self.assertEqual(
            claim_review.get("unsupported_claims"),
            ["Python 列表是可变序列"],
            msg="无相关候选证据的结论应标记为 unsupported",
        )
    def test_claim_check_preserves_decimal_points_inside_claims(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "模型准确率为 92.5%。"
                "火星有两颗天然卫星。"
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "模型准确率为 92.5%",
                "火星有两颗天然卫星",
            ],
            msg=(
                "句子切分不应把数字内部的小数点"
                "误认为句号"
            ),
        )

    def test_claim_check_preserves_dots_inside_domain_names(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "Official docs are at docs.python.org. "
                "Mars has two moons."
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "Official docs are at docs.python.org",
                "Mars has two moons",
            ],
            msg=(
                "Claim splitting should preserve dots "
                "inside domain names"
            ),
        )

    def test_claim_check_preserves_dots_inside_abbreviations(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "Use e.g. Python for scripting. "
                "Mars has two moons."
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "Use e.g. Python for scripting",
                "Mars has two moons",
            ],
            msg=(
                "Claim splitting should preserve dots "
                "inside abbreviations"
            ),
        )

    def test_claim_check_preserves_dots_inside_initialisms(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "U.S. policy changed recently. "
                "Mars has two moons."
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "U.S. policy changed recently",
                "Mars has two moons",
            ],
            msg=(
                "Claim splitting should preserve dots "
                "inside initialisms"
            ),
        )

    def test_claim_check_splits_after_initialism_at_sentence_end(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "The office is in the U.S. "
                "Mars has two moons."
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "The office is in the U.S",
                "Mars has two moons",
            ],
            msg=(
                "An initialism at sentence end should "
                "still allow claim splitting"
            ),
        )

    def test_claim_check_preserves_initialism_before_proper_noun(
        self,
    ):
        project_root = Path(__file__).resolve().parents[1]

        config = AgentConfig.default(
            workspace_root=project_root
        )

        coordinator = TemplateCoordinator(
            config=config,
            workspace_root=project_root,
        )

        state = AgentState(
            answer=(
                "The U.S. Army uses this system. "
                "Mars has two moons."
            ),
            evidence_mode="required",
        )

        coordinator._claim_check(state)

        self.assertEqual(
            state.claim_review.get("claims"),
            [
                "The U.S. Army uses this system",
                "Mars has two moons",
            ],
            msg=(
                "An initialism before a proper noun "
                "should stay in the same claim"
            ),
        )


if __name__ == "__main__":
    unittest.main()
