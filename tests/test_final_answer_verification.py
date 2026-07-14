import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.llm import ChatResult
from reasoning_agent_template.multiagent import MultiAgentOrchestrator


class DeterministicDeepSeekClient:
    """返回固定路由和固定错误回答，避免真实网络调用。"""

    model = "deepseek-v4-flash"

    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        prompt_text = "\n".join(message.content for message in messages)
        self.calls.append(prompt_text)

        if "ROUTING_DECISION_JSON" in prompt_text:
            route = {
                "difficulty": "medium",
                "workflow": "evidence_soft",
                "evidence_mode": "required",
                "evidence_strictness": "soft",
                "risk_level": "low",
                "category": "technical_claim",
                "sources": ["rag", "web"],
                "reasons": ["用户要求基于可靠证据解释技术机制"],
                "confidence": 0.9,
                "retrieval_query": "证据门禁机制",
            }
            return ChatResult(
                content=json.dumps(route, ensure_ascii=False),
                model=self.model,
                raw={"id": "route"},
            )

        if "REVIEW_DECISION_JSON" in prompt_text:
            review = {
                "review_status": "approve",
                "findings": ["软证据工作流适用于该问题"],
                "confidence": 0.9,
            }
            return ChatResult(
                content=json.dumps(review, ensure_ascii=False),
                model=self.model,
                raw={"id": "review"},
            )

        # 故意模拟一个违反 Gate 限制的最终回答。
        return ChatResult(
            content="可靠证据已经充分证明该机制完全正确。",
            model=self.model,
            raw={"id": "final"},
        )


class FinalAnswerVerificationTests(unittest.TestCase):
    def test_unqualified_evidence_requires_limited_final_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            knowledge = root / "knowledge"
            knowledge.mkdir()

            # 创建一条可以被检索到、但强制不能通过 Gate 的本地资料。
            (knowledge / "gate.md").write_text(
                "证据门禁机制会在回答前检查证据数量、风险和审批要求。",
                encoding="utf-8",
            )

            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)
            config.knowledge["min_score"] = 0.0

            # 设置极高门槛，确保本地资料被标记为不合格证据。
            config.gates["local_evidence_min_score"] = 999.0
            config.gates["local_evidence_min_semantic_score"] = 999.0

            client = DeterministicDeepSeekClient()
            orchestrator = MultiAgentOrchestrator(
                config=config,
                workspace_root=root,
                llm_client_factory=lambda _config: client,
            )

            # 禁止真实外部检索。
            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ):
                payload = orchestrator.run(
                    "请根据可靠证据解释这个框架中的证据门禁机制。"
                )

        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["strictness"], "soft")
        self.assertEqual(payload["evidence"]["status"], "unqualified")
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")

        limitation_markers = (
            "证据不足",
            "受限回答",
            "尚未通过",
            "不能视为确定结论",
        )

        self.assertTrue(
            any(marker in payload["answer"] for marker in limitation_markers),
            msg=f"最终回答没有声明证据不足：{payload['answer']}",
        )

        certainty_markers = (
            "充分证明",
            "可靠证据已经",
            "完全正确",
        )

        self.assertFalse(
            any(marker in payload["answer"] for marker in certainty_markers),
            msg=f"最终回答包含不允许的确定性表述：{payload['answer']}",
        )


if __name__ == "__main__":
    unittest.main()
