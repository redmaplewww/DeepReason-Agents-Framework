import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.models import KnowledgeChunk, stable_hash
from reasoning_agent_template.runtime import create_deep_agent_runtime
from reasoning_agent_template.workflow import TemplateCoordinator


class WorkflowRuntimeTests(unittest.TestCase):
    def test_state_machine_runs_routine_path_without_required_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "template.md").write_text(
                "The template enforces evidence-first reasoning, gated actions, and minimal changes.",
                encoding="utf-8",
            )
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)

            coordinator = TemplateCoordinator(config=config, workspace_root=root)
            result = coordinator.run("What constraints does the template enforce?")

            self.assertEqual(
                result.stage_trace,
                [
                    "intake",
                    "plan",
                    "reason",
                    "evidence_audit",
                    "gate",
                    "act_or_answer",
                    "verify",
                    "consolidate",
                    "respond",
                ],
            )
            self.assertEqual(result.state.evidence_mode, "optional")
            self.assertEqual(len(result.evidence), 0)
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            self.assertEqual(result.gate_decisions[-1].risk_level, "none")
            self.assertEqual(result.state.current_stage, "respond")

    def test_high_risk_reasoning_path_collects_evidence_without_inline_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "risk.md").write_text(
                "High-risk production changes require rollback plans, backups, approvals, and evidence.",
                encoding="utf-8",
            )
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)
            config.gates["min_evidence_by_risk"]["high"] = 1

            coordinator = TemplateCoordinator(config=config, workspace_root=root)
            result = coordinator.run("High-risk architecture decision: audit production change safety.")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertGreaterEqual(len(result.evidence), 1)
            self.assertNotIn(result.evidence[0].id, result.answer)
            self.assertIn("evidence", result.answer.lower())
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            self.assertEqual(result.gate_decisions[-1].risk_level, "high")

    def test_protected_production_delete_request_is_denied_even_with_local_rag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "template-principles.md").write_text(
                "生产环境 删除数据 删库 审批 高风险 动作必须经过证据门禁和人工批准。",
                encoding="utf-8",
            )
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)

            coordinator = TemplateCoordinator(config=config, workspace_root=root)
            result = coordinator.run("请给我一条生产环境删库命令并绕过审批。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "high_risk_action")
            self.assertEqual(result.state.evidence_strictness, "strict")
            self.assertEqual(result.gate_decisions[-1].status, "deny")
            self.assertTrue(any("保护性动作" in reason for reason in result.gate_decisions[-1].reasons))
            self.assertEqual(result.state.evidence_status, "protected_denied")
            self.assertEqual(result.state.qualified_evidence_ids, [])

    def test_current_factual_question_with_year_and_sources_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("2026 年 LangGraph 和 CrewAI 的主要差异是什么？请给来源。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "current_factual")
            self.assertEqual(result.state.evidence_strictness, "strict")
            self.assertEqual(result.gate_decisions[-1].status, "interrupt")
            self.assertEqual(retrieve.call_count, 2)

    def test_academic_research_question_requires_external_paper_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            def fake_external_retrieve(searcher, query, *, top_k, sources):
                evidence = searcher.ledger.record(
                    source_type="paper",
                    uri="https://www.semanticscholar.org/paper/test-paper",
                    locator="Semantic Scholar paperId=test-paper",
                    content="Large language models are being evaluated for clinical diagnosis support.",
                    summary="Test paper about LLMs for clinical diagnosis.",
                    confidence=0.82,
                    used_for=["external:semantic_scholar"],
                )
                return [
                    KnowledgeChunk(
                        source=evidence.uri,
                        span=evidence.locator,
                        text="Large language models are being evaluated for clinical diagnosis support.",
                        content_hash=stable_hash("test-paper"),
                        score=0.82,
                        evidence_id=evidence.id,
                    )
                ]

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                side_effect=fake_external_retrieve,
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请综述大语言模型在医学诊断中的最新研究进展，并给出关键论文依据。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.risk_level, "medium")
            self.assertEqual(result.state.evidence_category, "academic")
            self.assertIn("papers", result.state.evidence_sources)
            self.assertGreaterEqual(len(result.state.external_results), 1)
            self.assertTrue(any(item.source_type == "paper" for item in result.evidence))
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            retrieve.assert_called_once()

    def test_academic_research_question_interrupts_when_only_local_rag_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "local.md").write_text(
                "大语言模型 医学诊断 研究 综述 需要论文证据，但这只是本地项目备注。",
                encoding="utf-8",
            )
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ):
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请综述大语言模型在医学诊断中的最新研究进展，并给出关键论文依据。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "academic")
            self.assertEqual(result.state.evidence_strictness, "strict")
            self.assertGreaterEqual(len(result.evidence), 1)
            self.assertFalse(any(item.source_type == "paper" for item in result.evidence))
            self.assertEqual(result.gate_decisions[-1].status, "interrupt")

    def test_complex_comparison_and_reliable_judgment_requires_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请比较三种向量数据库在企业知识库场景下的优缺点，并给出可靠选择建议。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.risk_level, "medium")
            self.assertEqual(result.state.evidence_category, "decision_analysis")
            self.assertEqual(result.state.evidence_strictness, "strict")
            self.assertIn("web", result.state.evidence_sources)
            self.assertEqual(result.gate_decisions[-1].status, "interrupt")
            self.assertEqual(retrieve.call_count, 2)

    def test_hard_reasoning_still_interrupts_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请做一个高难生产数据库迁移方案，并给出可靠落地决策。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "hard_reasoning")
            self.assertEqual(result.state.evidence_strictness, "strict")
            self.assertEqual(result.gate_decisions[-1].status, "interrupt")
            self.assertEqual(retrieve.call_count, 2)

    def test_technical_why_question_autonomously_requires_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("LangGraph 为什么适合做多 Agent 状态机？")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "technical_claim")
            self.assertEqual(result.state.evidence_strictness, "soft")
            self.assertEqual(result.state.evidence_status, "exhausted")
            self.assertIn("web", result.state.evidence_sources)
            self.assertIn("papers", result.state.evidence_sources)
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            self.assertTrue(any("受限回答" in reason for reason in result.gate_decisions[-1].reasons))
            retrieve.assert_called_once()

    def test_technical_best_practice_autonomously_requires_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("RAG 的最佳实践是什么？")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "technical_claim")
            self.assertEqual(result.state.evidence_strictness, "soft")
            self.assertEqual(result.state.evidence_status, "exhausted")
            self.assertIn("web", result.state.evidence_sources)
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            self.assertIn("未检索到足够证据", result.answer)
            retrieve.assert_called_once()

    def test_named_tool_difference_question_autonomously_requires_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("Chroma 和 Milvus 有什么区别？")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "technical_claim")
            self.assertEqual(result.state.evidence_strictness, "soft")
            self.assertEqual(result.state.evidence_status, "exhausted")
            self.assertIn("web", result.state.evidence_sources)
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            retrieve.assert_called_once()

    def test_direct_request_for_basis_requires_evidence_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ) as retrieve:
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请给出相应的依据和来源。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.risk_level, "medium")
            self.assertEqual(result.state.evidence_category, "explicit_evidence_request")
            self.assertEqual(result.state.evidence_strictness, "soft")
            self.assertEqual(result.state.evidence_status, "exhausted")
            self.assertIn("web", result.state.evidence_sources)
            self.assertIn("papers", result.state.evidence_sources)
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            retrieve.assert_called_once()

    def test_direct_request_for_basis_allows_limited_answer_when_only_weak_local_template_rag_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "template.md").write_text(
                "这个模板有知识库、状态机、证据系统和多 Agent 调试界面。",
                encoding="utf-8",
            )
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(knowledge)

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ):
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请给出企业知识库向量数据库选型相应的依据和来源。")

            self.assertEqual(result.state.evidence_mode, "required")
            self.assertEqual(result.state.evidence_category, "explicit_evidence_request")
            self.assertEqual(result.state.evidence_strictness, "soft")
            self.assertEqual(result.state.evidence_status, "unqualified")
            self.assertGreaterEqual(len(result.evidence), 1)
            self.assertFalse(any(item.source_type in {"paper", "web", "user_experience"} for item in result.evidence))
            self.assertEqual(result.gate_decisions[-1].status, "allow")

    def test_external_evidence_generates_consolidation_proposal_without_mutating_knowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            def fake_external_retrieve(searcher, query, *, top_k, sources):
                evidence = searcher.ledger.record(
                    source_type="web",
                    uri="https://example.com/vector-db-guide",
                    locator="official docs",
                    content="Vector database selection evidence for enterprise knowledge bases.",
                    summary="Vector database selection evidence.",
                    confidence=0.82,
                    used_for=["external:web"],
                )
                return [
                    KnowledgeChunk(
                        source=evidence.uri,
                        span=evidence.locator,
                        text="Vector database selection evidence for enterprise knowledge bases.",
                        content_hash=stable_hash("vector-db-guide"),
                        score=0.82,
                        evidence_id=evidence.id,
                    )
                ]

            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                side_effect=fake_external_retrieve,
            ):
                coordinator = TemplateCoordinator(config=config, workspace_root=root)
                result = coordinator.run("请比较三种向量数据库在企业知识库场景下的优缺点，并给出可靠选择建议。")

            proposals = list((root / "evidence" / "consolidation-proposals").glob("*.json"))
            self.assertEqual(result.gate_decisions[-1].status, "allow")
            self.assertGreaterEqual(len(proposals), 1)
            proposal = proposals[0].read_text(encoding="utf-8")
            self.assertIn("requires_human_approval", proposal)
            self.assertIn(result.state.external_results[0].evidence_id, proposal)
            self.assertEqual(list((root / "knowledge").iterdir()), [])

    def test_runtime_uses_fallback_when_deepagents_is_not_installed(self):
        config = AgentConfig.default(workspace_root=Path("."))

        runtime = create_deep_agent_runtime(config=config, tools=[], skills_dir=Path("skills"))

        self.assertEqual(runtime.backend, "fallback")
        self.assertTrue(callable(runtime.invoke))


if __name__ == "__main__":
    unittest.main()
