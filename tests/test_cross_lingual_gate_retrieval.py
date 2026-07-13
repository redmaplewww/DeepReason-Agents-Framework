import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.knowledge import LocalKnowledgeBase


class CrossLingualGateRetrievalTests(unittest.TestCase):
    def test_chinese_gate_query_reaches_semantic_gate_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()

            target = knowledge_dir / "evidence-gate.md"
            target.write_text(
                "The evidence gate checks evidence counts, "
                "approval requirements, and workspace boundaries "
                "before the agent answers or acts.",
                encoding="utf-8",
            )

            (knowledge_dir / "memory.md").write_text(
                "Long-term memory stores durable project facts "
                "after an approved memory write.",
                encoding="utf-8",
            )

            kb = LocalKnowledgeBase(knowledge_dir)

            results = kb.retrieve(
                "证据门禁机制 框架 解释 依据",
                top_k=2,
                methods=["bm25", "semantic", "graph"],
                min_score=0.0,
            )

        target_result = next(
            (
                result
                for result in results
                if result.source == str(target)
            ),
            None,
        )

        self.assertIsNotNone(
            target_result,
            msg="中文查询没有检索到英文证据门禁文档",
        )

        semantic_score = target_result.score_breakdown.get(
            "semantic",
            0.0,
        )

        self.assertGreaterEqual(
            target_result.score,
            0.45,
            msg=(
                "目标文档综合分没有达到本地证据门槛："
                f"{target_result.score}"
            ),
        )

        self.assertGreaterEqual(
            semantic_score,
            0.25,
            msg=(
                "目标文档虽然相关，但跨语言语义分没有达到 "
                f"Gate 门槛：{semantic_score}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
