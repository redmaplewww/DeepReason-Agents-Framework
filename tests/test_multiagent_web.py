import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import request

from reasoning_agent_template.config import AgentConfig, load_agent_config
from reasoning_agent_template.llm import ChatResult, LLMRequestError, MissingApiKeyError
from reasoning_agent_template.models import KnowledgeChunk, stable_hash
from reasoning_agent_template.multiagent import MultiAgentOrchestrator, _merge_reviewer_route
from reasoning_agent_template.sessions import SessionStore
from reasoning_agent_template.web import create_server


class FakeDeepSeekClient:
    model = "deepseek-v4-flash"

    def __init__(self, content="DEEPSEEK_REAL_MARK"):
        self.content = content
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return ChatResult(content=self.content, model=self.model, raw={"id": "fake-deepseek"})


class FailingDeepSeekClient:
    model = "deepseek-v4-flash"

    def chat(self, messages, temperature, max_tokens):
        raise LLMRequestError("DeepSeek HTTP 500: upstream overloaded")


class FlakyDeepSeekClient:
    model = "deepseek-v4-flash"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, temperature, max_tokens):
        self.calls += 1
        if self.calls == 1:
            raise LLMRequestError("DeepSeek HTTP 500: transient")
        return ChatResult(content="RETRY_OK", model=self.model, raw={"id": "retry-ok"})


class StructuredRoutingDeepSeekClient:
    model = "deepseek-v4-flash"

    def __init__(
        self,
        *,
        coordinator_route: dict,
        reviewer_route: dict | None = None,
        final_answer: str = "FINAL_DEEPSEEK_ANSWER",
    ):
        self.coordinator_route = coordinator_route
        self.reviewer_route = reviewer_route or {
            "review_status": "approve",
            "findings": ["coordinator route is acceptable"],
        }
        self.final_answer = final_answer
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        prompt_text = "\n".join(message.content for message in messages)
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_text": prompt_text,
            }
        )
        if "ROUTING_DECISION_JSON" in prompt_text:
            return ChatResult(content=json.dumps(self.coordinator_route), model=self.model, raw={"id": "route"})
        if "REVIEW_DECISION_JSON" in prompt_text:
            return ChatResult(content=json.dumps(self.reviewer_route), model=self.model, raw={"id": "review"})
        return ChatResult(content=self.final_answer, model=self.model, raw={"id": "final"})


class BlockingFinalAnswerDeepSeekClient:
    model = "deepseek-v4-flash"

    def __init__(self, *, release_event: threading.Event):
        self.release_event = release_event
        self.calls = []

    def chat(self, messages, temperature, max_tokens):
        prompt_text = "\n".join(message.content for message in messages)
        self.calls.append({"prompt_text": prompt_text})
        if "ROUTING_DECISION_JSON" in prompt_text:
            return ChatResult(
                content=json.dumps(
                    {
                        "difficulty": "medium",
                        "workflow": "evidence_soft",
                        "evidence_mode": "required",
                        "evidence_strictness": "soft",
                        "risk_level": "medium",
                        "category": "technical_claim",
                        "sources": ["rag", "web"],
                        "reasons": ["needs evidence-backed technical answer"],
                        "confidence": 0.8,
                    }
                ),
                model=self.model,
                raw={"id": "route"},
            )
        if "REVIEW_DECISION_JSON" in prompt_text:
            return ChatResult(
                content=json.dumps({"review_status": "approve", "findings": []}),
                model=self.model,
                raw={"id": "review"},
            )
        if not self.release_event.wait(timeout=10):
            raise LLMRequestError("test timed out waiting for final answer release")
        return ChatResult(content="FINAL_AFTER_BLOCK", model=self.model, raw={"id": "final"})


def _fake_orchestrator(config=None, workspace_root=".", content="DEEPSEEK_REAL_MARK"):
    client = FakeDeepSeekClient(content=content)
    orchestrator = MultiAgentOrchestrator(
        config=config or load_agent_config("agent.yaml"),
        workspace_root=workspace_root,
        llm_client_factory=lambda _config: client,
    )
    return orchestrator, client


def _structured_orchestrator(config, workspace_root, *, coordinator_route, reviewer_route=None, final_answer="FINAL"):
    client = StructuredRoutingDeepSeekClient(
        coordinator_route=coordinator_route,
        reviewer_route=reviewer_route,
        final_answer=final_answer,
    )
    orchestrator = MultiAgentOrchestrator(
        config=config,
        workspace_root=workspace_root,
        llm_client_factory=lambda _config: client,
    )
    return orchestrator, client


class MultiAgentWebTests(unittest.TestCase):
    def test_reviewer_cannot_weaken_coordinator_safety_route(self):
        coordinator = {
            "source": "llm",
            "difficulty": "hard",
            "workflow": "evidence_strict",
            "evidence_mode": "required",
            "evidence_strictness": "strict",
            "risk_level": "high",
            "category": "hard_reasoning",
            "sources": ["rag", "web"],
            "reasons": ["coordinator safety route"],
            "confidence": 0.9,
        }
        reviewer = {
            "review_status": "escalate",
            "difficulty": "simple",
            "workflow": "routine",
            "evidence_mode": "optional",
            "evidence_strictness": "none",
            "risk_level": "none",
            "category": "routine",
            "findings": ["reviewer supplied a weaker route"],
        }

        merged = _merge_reviewer_route(coordinator, reviewer, fallback_message="test")

        self.assertEqual(merged["evidence_mode"], "required")
        self.assertEqual(merged["evidence_strictness"], "strict")
        self.assertEqual(merged["risk_level"], "high")
        self.assertEqual(merged["workflow"], "evidence_strict")
        self.assertEqual(merged["reviewer_route_ignored"], "weaker_than_coordinator")

    def test_session_store_restores_short_term_memory_across_orchestrators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")
            store = SessionStore(root / "sessions")
            first, _ = _fake_orchestrator(config=config, workspace_root=root, content="FIRST")
            first.session_store = store
            first.run("first thread message", thread_id="thread-1")

            second, client = _fake_orchestrator(config=config, workspace_root=root, content="SECOND")
            second.session_store = store
            second.run("second thread message", thread_id="thread-1")

        self.assertTrue(
            any(
                "first thread message" in str(message.content)
                for call in client.calls
                for message in call["messages"]
            )
        )

    def test_multi_agent_debug_payload_contains_required_monitors(self):
        orchestrator, _client = _fake_orchestrator()

        payload = orchestrator.run("What constraints does this template enforce?")

        self.assertIn("run_id", payload)
        self.assertIn("answer", payload)
        self.assertIn("agents", payload)
        self.assertIn("state_machine", payload)
        self.assertIn("evidence", payload)
        self.assertIn("rag", payload)
        self.assertIn("gates", payload)
        self.assertIn("workflow", payload)
        self.assertIn("memory", payload)
        self.assertIn("events", payload)
        self.assertGreaterEqual(len(payload["agents"]), 6)
        self.assertEqual(payload["agents"][0]["name"], "coordinator")
        self.assertEqual(payload["state_machine"]["current"], "respond")
        self.assertNotIn("retrieve", payload["state_machine"]["trace"])
        retrieve_node = next(node for node in payload["workflow"]["nodes"] if node["id"] == "retrieve")
        self.assertEqual(retrieve_node["effective_status"], "skipped")
        self.assertEqual(payload["workflow"]["status"], "completed")
        self.assertGreaterEqual(len(payload["workflow"]["nodes"]), 10)
        self.assertGreaterEqual(len(payload["workflow"]["edges"]), 9)
        self.assertIn("evidence_audit", payload["workflow"]["checkpoints"])
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        self.assertEqual(payload["evidence"]["mode"], "optional")
        self.assertEqual(payload["answer"], "DEEPSEEK_REAL_MARK")
        self.assertEqual(payload["runtime"]["llm"]["status"], "called")
        self.assertTrue(any(event["kind"] == "stage_started" for event in payload["events"]))
        self.assertTrue(
            any(event["kind"] == "stage_completed" and "duration_ms" in event for event in payload["events"])
        )

    def test_workflow_status_exposes_control_graph_and_node_telemetry(self):
        orchestrator, _client = _fake_orchestrator()

        workflow = orchestrator.workflow_status()
        node_order = {node["id"]: index for index, node in enumerate(workflow["nodes"])}
        back_edges = [
            edge
            for edge in workflow["edges"]
            if edge["from"] in node_order and edge["to"] in node_order and node_order[edge["to"]] <= node_order[edge["from"]]
        ]

        self.assertTrue(back_edges)
        self.assertTrue(any(edge.get("type") == "retry" and edge.get("to") == "retrieve" for edge in workflow["edges"]))
        for node in workflow["nodes"]:
            self.assertIn("agent", node)
            self.assertIn("effective_status", node)
            self.assertIn("work_done", node)
            self.assertIn("skip_reason", node)
            self.assertIn("duration_ms", node)
            self.assertIn("artifacts", node)
            self.assertIn("actual_input", node["artifacts"])
            self.assertIn("actual_output", node["artifacts"])
            self.assertIn("handoff", node["artifacts"])

    def test_workflow_nodes_expose_real_stage_handoff_artifacts(self):
        orchestrator, _client = _fake_orchestrator(content="ROUTINE_ANSWER")

        payload = orchestrator.run("hello")
        nodes = {node["id"]: node for node in payload["workflow"]["nodes"]}

        plan_artifacts = nodes["plan"]["artifacts"]
        self.assertIn("routing", plan_artifacts["actual_input"])
        self.assertIn("plan_steps", plan_artifacts["actual_output"])
        self.assertTrue(plan_artifacts["actual_output"]["plan_steps"])
        self.assertEqual(plan_artifacts["handoff"]["from"], "intake")
        self.assertEqual(plan_artifacts["handoff"]["to"], "plan")

        gate_artifacts = nodes["gate"]["artifacts"]
        self.assertIn("gate_decisions", gate_artifacts["actual_output"])
        self.assertTrue(gate_artifacts["actual_output"]["gate_decisions"])

    def test_routine_workflow_marks_evidence_retrieval_as_skipped_not_fake_work(self):
        orchestrator, _client = _fake_orchestrator(content="ROUTINE_ANSWER")

        payload = orchestrator.run("hello")
        nodes = {node["id"]: node for node in payload["workflow"]["nodes"]}

        self.assertEqual(nodes["retrieve"]["effective_status"], "skipped")
        self.assertFalse(nodes["retrieve"]["work_done"])
        self.assertIn("evidence_not_required", nodes["retrieve"]["skip_reason"])
        self.assertEqual(nodes["evidence_audit"]["effective_status"], "skipped")
        self.assertEqual(nodes["respond"]["effective_status"], "completed")

    def test_deepseek_answer_replaces_local_template_draft(self):
        orchestrator, client = _fake_orchestrator(content="这是 DeepSeek 的真实回答标记。")

        payload = orchestrator.run("这个模板的工作流和证据系统怎么运行？")

        self.assertEqual(payload["answer"], "这是 DeepSeek 的真实回答标记。")
        self.assertEqual(payload["evidence"]["mode"], "optional")
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        self.assertEqual(len(client.calls), 3)
        prompt_text = "\n".join(message.content for message in client.calls[-1]["messages"])
        self.assertNotIn("Local draft", prompt_text)
        self.assertNotIn("普通对话不会强制走证据系统", payload["answer"])

    def test_chinese_identity_question_does_not_require_evidence_system(self):
        orchestrator, _client = _fake_orchestrator(content="我是 DeepSeek 生成的身份回答。")

        payload = orchestrator.run("你是谁")

        self.assertEqual(payload["evidence"]["count"], 0)
        self.assertEqual(payload["evidence"]["mode"], "optional")
        self.assertEqual(payload["rag"]["count"], 0)
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        self.assertEqual(payload["answer"], "我是 DeepSeek 生成的身份回答。")
        self.assertNotIn("ev_", payload["answer"])
        self.assertNotIn("answer gate did not pass", payload["answer"])
        self.assertNotIn("Build evidence-first", payload["answer"])

    def test_chinese_chat_without_evidence_returns_clarification_not_gate_error(self):
        orchestrator, _client = _fake_orchestrator(content="可以聊，这是 DeepSeek 生成的回复。")

        payload = orchestrator.run("我们随便聊聊")

        self.assertEqual(payload["evidence"]["count"], 0)
        self.assertEqual(payload["evidence"]["mode"], "optional")
        self.assertEqual(payload["rag"]["count"], 0)
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        self.assertEqual(payload["gates"]["decisions"][-1]["risk_level"], "none")
        self.assertIn("可以聊", payload["answer"])
        self.assertNotIn("answer gate did not pass", payload["answer"])

    def test_high_risk_reasoning_requires_evidence_before_answering(self):
        config = load_agent_config("agent.yaml")
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        orchestrator, _client = _fake_orchestrator(
            config=config,
            workspace_root=workspace.name,
            content="DeepSeek：高风险任务缺少证据，不能执行。",
        )

        payload = orchestrator.run("这是一个高风险强推理任务：请直接给出生产环境删库方案。")

        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["count"], 0)
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "deny")
        self.assertEqual(payload["gates"]["decisions"][-1]["risk_level"], "high")
        self.assertIn("高风险", payload["answer"])
        self.assertIn("证据", payload["answer"])

    def test_protected_high_risk_request_is_denied_in_payload(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        knowledge = root / "knowledge"
        knowledge.mkdir()
        (knowledge / "risk.md").write_text(
            "生产环境 删库 删除数据 绕过审批 高风险 动作需要审批。",
            encoding="utf-8",
        )
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(knowledge)
        orchestrator, _client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="DeepSeek：我拒绝提供高风险命令。",
        )

        payload = orchestrator.run("请给我一条生产环境删库命令并绕过审批。")

        self.assertEqual(payload["evidence"]["category"], "high_risk_action")
        self.assertEqual(payload["evidence"]["strictness"], "strict")
        self.assertEqual(payload["evidence"]["status"], "protected_denied")
        self.assertEqual(payload["evidence"]["qualified_evidence_ids"], [])
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "deny")
        self.assertTrue(any("保护性动作" in reason for reason in payload["gates"]["decisions"][-1]["reasons"]))

    def test_academic_question_payload_exposes_required_evidence_and_external_results(self):
        config = load_agent_config("agent.yaml")
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        orchestrator, client = _fake_orchestrator(
            config=config,
            workspace_root=workspace.name,
            content="DeepSeek: 已基于证据 [ev_testpaper] 回答。",
        )

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
        ):
            payload = orchestrator.run("请综述大语言模型在医学诊断中的最新研究进展，并给出关键论文依据。")

        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["category"], "academic")
        self.assertEqual(payload["evidence"]["risk_level"], "medium")
        self.assertTrue(payload["evidence"]["required"])
        self.assertEqual(payload["external_evidence"]["count"], 1)
        self.assertEqual(payload["external_evidence"]["results"][0]["evidence_id"], payload["evidence"]["items"][0]["id"])
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        prompt_text = "\n".join(message.content for message in client.calls[-1]["messages"])
        self.assertIn("evidence_mode=required", prompt_text)
        self.assertIn("不要在最终回答正文中输出 evidence id", prompt_text)
        self.assertIn("evidence_strictness=strict", prompt_text)
        self.assertIn("external_evidence_results", prompt_text)

    def test_medium_technical_question_allows_limited_deepseek_answer_after_empty_search(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="证据检索不足，我先给出受限回答。",
        )

        with patch(
            "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
            autospec=True,
            return_value=[],
        ):
            payload = orchestrator.run("RAG 的最佳实践是什么？")

        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["category"], "technical_claim")
        self.assertEqual(payload["evidence"]["strictness"], "soft")
        self.assertEqual(payload["evidence"]["status"], "exhausted")
        self.assertIn("web", payload["external_evidence"]["attempted_sources"])
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "allow")
        self.assertTrue(any("受限回答" in reason for reason in payload["gates"]["decisions"][-1]["reasons"]))
        prompt_text = "\n".join(message.content for message in client.calls[-1]["messages"])
        self.assertIn("evidence_strictness=soft", prompt_text)
        self.assertIn("受限回答", prompt_text)

    def test_llm_coordinator_autonomously_routes_scientific_question_to_soft_evidence_workflow(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, client = _structured_orchestrator(
            config,
            root,
            coordinator_route={
                "difficulty": "medium",
                "workflow": "evidence_soft",
                "evidence_mode": "required",
                "evidence_strictness": "soft",
                "risk_level": "medium",
                "category": "scientific_claim",
                "sources": ["rag", "web", "papers", "user_experience"],
                "reasons": ["The question asks for scientific factors and mechanisms that need external support."],
                "confidence": 0.83,
            },
            final_answer="已基于检索证据回答。",
        )

        def fake_external_retrieve(searcher, query, *, top_k, sources):
            evidence = searcher.ledger.record(
                source_type="paper",
                uri="https://example.com/high-entropy-alloy-strength",
                locator="mock paper",
                content="Strength in high entropy alloys is affected by solid solution strengthening and microstructure.",
                summary="High entropy alloy strength factors.",
                confidence=0.81,
                used_for=["external:paper"],
            )
            return [
                KnowledgeChunk(
                    source=evidence.uri,
                    span=evidence.locator,
                    text="Strength factors in high entropy alloys.",
                    content_hash=stable_hash("hea-strength"),
                    score=0.81,
                    evidence_id=evidence.id,
                )
            ]

        with patch(
            "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
            autospec=True,
            side_effect=fake_external_retrieve,
        ) as retrieve:
            payload = orchestrator.run("高熵合金的强度影响因素")

        self.assertEqual(payload["routing"]["source"], "llm")
        self.assertEqual(payload["routing"]["difficulty"], "medium")
        self.assertEqual(payload["workflow"]["variant"], "evidence_soft")
        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["category"], "scientific_claim")
        self.assertEqual(payload["evidence"]["strictness"], "soft")
        self.assertIn("web", payload["external_evidence"]["attempted_sources"])
        self.assertIn("papers", payload["external_evidence"]["attempted_sources"])
        self.assertGreaterEqual(payload["external_evidence"]["count"], 1)
        self.assertEqual(payload["reviewer"]["status"], "approve")
        self.assertTrue(any(agent["name"] == "reviewer" for agent in payload["agents"]))
        self.assertTrue(any("ROUTING_DECISION_JSON" in call["prompt_text"] for call in client.calls))
        self.assertTrue(any("REVIEW_DECISION_JSON" in call["prompt_text"] for call in client.calls))
        retrieve.assert_called_once()

    def test_reviewer_can_escalate_overly_loose_coordinator_routing_before_workflow_runs(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, _client = _structured_orchestrator(
            config,
            root,
            coordinator_route={
                "difficulty": "simple",
                "workflow": "routine",
                "evidence_mode": "optional",
                "evidence_strictness": "none",
                "risk_level": "none",
                "category": "routine",
                "sources": [],
                "reasons": ["Coordinator thought this was a simple explanation."],
                "confidence": 0.52,
            },
            reviewer_route={
                "review_status": "escalate",
                "difficulty": "hard",
                "workflow": "evidence_strict",
                "evidence_mode": "required",
                "evidence_strictness": "strict",
                "risk_level": "medium",
                "category": "academic",
                "sources": ["rag", "papers", "web", "user_experience"],
                "findings": ["The user asks for sources for a scientific explanation, so evidence must run."],
                "reasons": ["Scientific source-backed answer requires reviewed evidence."],
                "confidence": 0.91,
            },
            final_answer="证据不足，已由 reviewer 打回。",
        )

        with patch(
            "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
            autospec=True,
            return_value=[],
        ) as retrieve:
            payload = orchestrator.run("高熵合金强度影响因素有哪些？请给来源。")

        self.assertEqual(payload["routing"]["source"], "llm+reviewer")
        self.assertEqual(payload["reviewer"]["status"], "escalate")
        self.assertEqual(payload["workflow"]["variant"], "evidence_strict")
        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["strictness"], "strict")
        self.assertEqual(payload["evidence"]["category"], "academic")
        self.assertEqual(payload["gates"]["decisions"][-1]["status"], "interrupt")
        self.assertEqual(retrieve.call_count, 2)

    def test_payload_exposes_external_search_diagnostics_and_consolidation(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, _client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="DeepSeek: 已基于证据回答。",
        )

        def fake_external_retrieve(searcher, query, *, top_k, sources):
            searcher.diagnostics.append(
                {
                    "source": "semantic_scholar",
                    "status": "error",
                    "message": "HTTP 429 Too Many Requests",
                }
            )
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
            payload = orchestrator.run("请比较三种向量数据库在企业知识库场景下的优缺点，并给出可靠选择建议。")

        self.assertGreaterEqual(len(payload["external_evidence"]["diagnostics"]), 1)
        self.assertIn("429", payload["external_evidence"]["diagnostics"][0]["message"])
        self.assertGreaterEqual(len(payload["evidence"]["consolidation_proposals"]), 1)
        self.assertIn("consolidation-proposals", payload["evidence"]["consolidation_proposals"][0]["path"])

    def test_answer_hides_evidence_ids_and_payload_exposes_reference_index(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, _client = _structured_orchestrator(
            config,
            root,
            coordinator_route={
                "difficulty": "medium",
                "workflow": "evidence_soft",
                "evidence_mode": "required",
                "evidence_strictness": "soft",
                "risk_level": "medium",
                "category": "scientific_claim",
                "sources": ["web", "papers"],
                "reasons": ["scientific claim needs evidence"],
                "confidence": 0.86,
            },
            final_answer="高熵合金强度主要受固溶强化影响。[ev_fake_should_not_render] 证据：ev_fake_should_not_render",
        )

        def fake_external_retrieve(searcher, query, *, top_k, sources):
            evidence = searcher.ledger.record(
                source_type="paper",
                uri="https://example.com/hea-strength",
                locator="mock paper",
                content="High entropy alloy strength is affected by solid solution strengthening.",
                summary="High entropy alloy strength evidence.",
                confidence=0.84,
                used_for=["external:paper"],
            )
            return [
                KnowledgeChunk(
                    source=evidence.uri,
                    span=evidence.locator,
                    text="Solid solution strengthening affects high entropy alloy strength.",
                    content_hash=stable_hash("hea-strength-reference"),
                    score=0.84,
                    evidence_id=evidence.id,
                )
            ]

        with patch(
            "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
            autospec=True,
            side_effect=fake_external_retrieve,
        ):
            payload = orchestrator.run("高熵合金的强度影响因素有哪些？")

        self.assertNotIn("ev_", payload["answer"])
        self.assertNotIn("证据：", payload["answer"])
        references = payload["evidence"]["references"]
        self.assertEqual(references[0]["index"], 1)
        self.assertEqual(references[0]["id"], payload["evidence"]["items"][0]["id"])
        self.assertEqual(references[0]["uri"], "https://example.com/hea-strength")
        self.assertIn("High entropy alloy strength evidence", references[0]["summary"])

    def test_status_exposes_running_agent_and_workflow_while_chat_is_in_flight(self):
        release_event = threading.Event()
        client = BlockingFinalAnswerDeepSeekClient(release_event=release_event)
        server = create_server(
            host="127.0.0.1",
            port=0,
            config_path="agent.yaml",
            workspace_root=".",
            llm_client_factory=lambda _config: client,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            with patch(
                "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
                autospec=True,
                return_value=[],
            ):
                accepted = _post_json(
                    f"{base_url}/api/chat",
                    {"message": "RAG 的最佳实践是什么？", "async": True},
                )
                self.assertEqual(accepted["status"], "accepted")
                running = None
                for _ in range(50):
                    status = _get_json(f"{base_url}/api/status")
                    if status["status"] == "running":
                        running = status
                        release_event.set()
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(running)
                self.assertTrue(running["workflow"]["active"])
                self.assertEqual(running["workflow"]["status"], "running")
                self.assertTrue(any(agent["active"] for agent in running["agents"]))
                self.assertTrue(any(node["status"] == "active" for node in running["workflow"]["nodes"]))
                self.assertIn("working_hint", running)

                completed = None
                for _ in range(50):
                    status = _get_json(f"{base_url}/api/status")
                    if status["status"] == "completed":
                        completed = status
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(completed)
                self.assertEqual(completed["answer"], "FINAL_AFTER_BLOCK")
        finally:
            release_event.set()
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_short_term_memory_is_injected_into_next_deepseek_prompt(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        config = AgentConfig.default(workspace_root=Path(workspace.name))
        orchestrator, client = _fake_orchestrator(
            config=config,
            workspace_root=workspace.name,
            content="MEMORY_MARK",
        )

        first = orchestrator.run("我喜欢喝乌龙茶。")
        second = orchestrator.run("我刚才说我喜欢喝什么？")

        self.assertEqual(first["memory"]["short_term"]["turns"], 1)
        self.assertEqual(second["memory"]["short_term"]["turns"], 2)
        prompt_text = "\n".join(message.content for message in client.calls[-1]["messages"])
        self.assertIn("短期记忆", prompt_text)
        self.assertIn("我喜欢喝乌龙茶", prompt_text)
        self.assertIn("MEMORY_MARK", prompt_text)

    def test_short_term_memory_is_isolated_by_thread_id(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        config = AgentConfig.default(workspace_root=Path(workspace.name))
        orchestrator, client = _fake_orchestrator(
            config=config,
            workspace_root=workspace.name,
            content="THREAD_MARK",
        )

        orchestrator.run("我喜欢喝乌龙茶。", thread_id="tea-thread")
        orchestrator.run("我喜欢喝咖啡。", thread_id="coffee-thread")
        payload = orchestrator.run("我刚才说我喜欢喝什么？", thread_id="tea-thread")

        self.assertEqual(payload["memory"]["short_term"]["thread_id"], "tea-thread")
        self.assertEqual(payload["memory"]["short_term"]["turns"], 2)
        prompt_text = "\n".join(message.content for message in client.calls[-1]["messages"])
        self.assertIn("我喜欢喝乌龙茶", prompt_text)
        self.assertNotIn("我喜欢喝咖啡", prompt_text)

    def test_evidence_followup_uses_short_term_context_for_retrieval(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        (root / "knowledge").mkdir()
        config = AgentConfig.default(workspace_root=root)
        config.knowledge["directory"] = str(root / "knowledge")
        orchestrator, _client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="UNRELATED_PAPERS_NOISE",
        )
        captured_queries = []

        def fake_external_retrieve(searcher, query, *, top_k, sources):
            captured_queries.append(query)
            evidence = searcher.ledger.record(
                source_type="web",
                uri="https://example.com/vector-db-evidence",
                locator="mock search result",
                content="Vector database selection evidence for enterprise knowledge bases.",
                summary="Vector database selection evidence.",
                confidence=0.74,
                used_for=["external:web"],
            )
            return [
                KnowledgeChunk(
                    source=evidence.uri,
                    span=evidence.locator,
                    text="Vector database selection evidence for enterprise knowledge bases.",
                    content_hash=stable_hash("vector-db-evidence"),
                    score=0.74,
                    evidence_id=evidence.id,
                )
            ]

        with patch(
            "reasoning_agent_template.workflow.ExternalEvidenceSearch.retrieve",
            autospec=True,
            side_effect=fake_external_retrieve,
        ):
            orchestrator.run("我们刚才讨论的是企业知识库的向量数据库选型。", thread_id="evidence-thread")
            payload = orchestrator.run("请给出相应依据。", thread_id="evidence-thread")

        self.assertEqual(payload["evidence"]["mode"], "required")
        self.assertEqual(payload["evidence"]["category"], "explicit_evidence_request")
        self.assertIn("web", payload["external_evidence"]["attempted_sources"])
        self.assertIn("papers", payload["external_evidence"]["attempted_sources"])
        self.assertEqual(payload["external_evidence"]["count"], 1)
        self.assertTrue(any("向量数据库" in query for query in captured_queries))
        self.assertFalse(any("UNRELATED_PAPERS_NOISE" in query for query in captured_queries))
        self.assertIn("web", payload["evidence"]["sources"])

    def test_explicit_long_term_memory_persists_across_orchestrators(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        config = AgentConfig.default(workspace_root=root)
        first_orchestrator, _first_client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="已记住。",
        )

        first_payload = first_orchestrator.run("请记住：我的代号是蓝鲸。")

        self.assertGreaterEqual(first_payload["memory"]["long_term"]["writes"], 1)
        self.assertTrue((root / "memory" / "user.jsonl").exists())

        second_client = FakeDeepSeekClient(content="你的代号是蓝鲸。")
        second_orchestrator = MultiAgentOrchestrator(
            config=config,
            workspace_root=root,
            llm_client_factory=lambda _config: second_client,
        )
        second_orchestrator.run("我的代号是什么？")

        prompt_text = "\n".join(message.content for message in second_client.calls[-1]["messages"])
        self.assertIn("长期记忆", prompt_text)
        self.assertIn("蓝鲸", prompt_text)

    def test_document_or_paper_memory_request_is_rejected_as_knowledge_base_content(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        config = AgentConfig.default(workspace_root=root)
        orchestrator, _client = _fake_orchestrator(
            config=config,
            workspace_root=root,
            content="这应该进入知识库，不写长期记忆。",
        )

        payload = orchestrator.run("请记住：这篇论文指出向量数据库评测基准应放入知识库。")

        self.assertEqual(payload["memory"]["long_term"]["writes"], 1)
        decision = payload["memory"]["long_term"]["write_decisions"][0]
        self.assertEqual(decision["status"], "deny")
        self.assertIn("knowledge base", " ".join(decision["reasons"]))
        self.assertFalse((root / "memory" / "user.jsonl").exists())

    def test_missing_deepseek_key_raises_instead_of_template_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            orchestrator = MultiAgentOrchestrator(config=config, workspace_root=tmp)

            with self.assertRaises(MissingApiKeyError):
                orchestrator.run("你是谁")

    def test_web_status_and_chat_api(self):
        client = FakeDeepSeekClient(content="WEB_DEEPSEEK_MARK")
        server = create_server(
            host="127.0.0.1",
            port=0,
            config_path="agent.yaml",
            workspace_root=".",
            llm_client_factory=lambda _config: client,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            status = _get_json(f"{base_url}/api/status")
            self.assertEqual(status["runtime"]["agent"], "reasoning-agent-template")
            self.assertEqual(status["status"], "ready")
            self.assertIn("agents", status)
            self.assertIn("workflow", status)
            self.assertEqual(status["workflow"]["status"], "idle")

            workflow = _get_json(f"{base_url}/api/workflow")
            self.assertIn("nodes", workflow)
            self.assertIn("edges", workflow)
            self.assertEqual(workflow["status"], "idle")

            chat = _post_json(
                f"{base_url}/api/chat",
                {"message": "What constraints does this template enforce?"},
            )
            self.assertEqual(chat["answer"], "WEB_DEEPSEEK_MARK")
            self.assertEqual(chat["state_machine"]["current"], "respond")
            self.assertEqual(chat["workflow"]["status"], "completed")
            self.assertEqual(chat["evidence"]["mode"], "optional")
            self.assertEqual(chat["runtime"]["llm"]["status"], "called")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_web_chat_api_returns_structured_llm_errors(self):
        server = create_server(
            host="127.0.0.1",
            port=0,
            config_path="agent.yaml",
            workspace_root=".",
            llm_client_factory=lambda _config: FailingDeepSeekClient(),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            error_payload = _post_json(
                f"{base_url}/api/chat",
                {"message": "你是谁？"},
                expect_status=502,
            )
            self.assertEqual(error_payload["type"], "LLMRequestError")
            self.assertEqual(error_payload["phase"], "llm")
            self.assertIn("DeepSeek HTTP 500", error_payload["error"])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_orchestrator_retries_transient_deepseek_failure_once(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        root = Path(workspace.name)
        config = AgentConfig.default(workspace_root=root)
        client = FlakyDeepSeekClient()
        orchestrator = MultiAgentOrchestrator(
            config=config,
            workspace_root=root,
            llm_client_factory=lambda _config: client,
        )

        payload = orchestrator.run("你是谁？")

        self.assertEqual(payload["answer"], "RETRY_OK")
        self.assertEqual(client.calls, 4)
        self.assertTrue(any(event["kind"] == "routing_retry" for event in payload["events"]))

    def test_status_marks_llm_configured_from_local_secret_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs" / "secrets.local.json").write_text(
                json.dumps({"deepseek_api_key": "local-secret-key"}),
                encoding="utf-8",
            )
            config = load_agent_config("agent.yaml")
            config.workspace_root = root
            orchestrator = MultiAgentOrchestrator(config=config, workspace_root=root)

            payload = orchestrator.status()

        self.assertTrue(payload["runtime"]["llm"]["configured"])

    def test_web_ui_contains_debug_surfaces(self):
        server = create_server(host="127.0.0.1", port=0, config_path="agent.yaml", workspace_root=".")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            html = request.urlopen(f"{base_url}/", timeout=10).read().decode("utf-8")
            self.assertIn("Agent 调试控制台", html)
            self.assertIn("状态机", html)
            self.assertIn("证据系统", html)
            self.assertIn("RAG", html)
            self.assertIn("外部证据", html)
            self.assertIn("多 Agent 运行时", html)
            self.assertIn("工作流", html)
            script = request.urlopen(f"{base_url}/assets/app.js", timeout=10).read().decode("utf-8")
            self.assertIn("workflow-chain", script)
            self.assertIn("参考文献", script)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_web_ui_contains_debug_surfaces(self):
        server = create_server(host="127.0.0.1", port=0, config_path="agent.yaml", workspace_root=".")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            html = request.urlopen(f"{base_url}/", timeout=10).read().decode("utf-8")
            self.assertIn("Agent 调试控制台", html)
            self.assertIn("状态机", html)
            self.assertIn("证据系统", html)
            self.assertIn("RAG", html)
            self.assertIn("外部证据", html)
            self.assertIn("多 Agent 运行时", html)
            self.assertIn("工作流", html)
            self.assertIn("路由/审查", html)
            self.assertIn("progressBanner", html)
            self.assertIn("workflowGraph", html)
            self.assertNotIn("workflowChain", html)
            self.assertNotIn("stateTimeline", html)
            script = request.urlopen(f"{base_url}/assets/app.js", timeout=10).read().decode("utf-8")
            self.assertIn("workflow-graph", script)
            self.assertIn("workflowGraphMarkup", script)
            self.assertIn("graph-node-agent-label", script)
            self.assertIn("graph-node-effect-label", script)
            self.assertNotIn("workflowChainMarkup", script)
            self.assertNotIn("renderTimeline", script)
            self.assertIn("参考文献", script)
            self.assertIn("async: true", script)
            css = request.urlopen(f"{base_url}/assets/app.css", timeout=10).read().decode("utf-8")
            self.assertIn(".workflow-graph", css)
            self.assertIn("grid-column: 1 / -1", css)
            self.assertNotIn(".workflow-chain", css)
            self.assertNotIn(".timeline", css)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_web_ui_contains_debug_surfaces(self):
        server = create_server(host="127.0.0.1", port=0, config_path="agent.yaml", workspace_root=".")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            html = request.urlopen(f"{base_url}/", timeout=10).read().decode("utf-8")
            self.assertIn("Agent 调试控制台", html)
            self.assertIn("当前节点", html)
            self.assertIn("证据系统", html)
            self.assertIn("RAG", html)
            self.assertIn("外部证据", html)
            self.assertIn("多 Agent 运行时", html)
            self.assertIn("工作流点线图", html)
            self.assertIn("路由/审查", html)
            self.assertIn("data-chat-mode", html)
            self.assertIn("搭建助手", html)
            self.assertIn("/配置", html)
            self.assertIn("progressBanner", html)
            self.assertIn("workflowGraph", html)
            self.assertIn("/assets/vendor/cytoscape.min.js", html)
            self.assertNotIn("/assets/vendor/dagre.min.js", html)
            self.assertNotIn("/assets/vendor/cytoscape-dagre.min.js", html)
            self.assertNotIn("workflowChain", html)
            self.assertNotIn("stateTimeline", html)

            vendor = request.urlopen(f"{base_url}/assets/vendor/cytoscape.min.js", timeout=10).read().decode("utf-8")
            self.assertIn("cytoscape", vendor.lower())

            script = request.urlopen(f"{base_url}/assets/app.js", timeout=10).read().decode("utf-8")
            self.assertIn("workflow-graph", script)
            self.assertIn("cytoscape({", script)
            self.assertIn('name: "preset"', script)
            self.assertIn("workflowHybridLayoutOptions", script)
            self.assertIn("workflowNodePosition", script)
            self.assertIn("workflowLoopClusterNodes", script)
            self.assertIn("workflowLoopNodeOrder", script)
            self.assertIn("selectedWorkflowElement", script)
            self.assertIn("handleWorkflowGraphSelection", script)
            self.assertIn("focusWorkflowSelectionPanel", script)
            self.assertIn("selection-focus", script)
            self.assertIn("renderWorkflowSelectionPanel", script)
            self.assertIn("workflowEdgeTransitionDetails", script)
            self.assertIn("renderArtifactBlock", script)
            self.assertIn("renderWorkflowEditor", script)
            self.assertIn("renderAgentsEditor", script)
            self.assertIn("builderVisibleAgents", script)
            self.assertIn("builderVisibleWorkflowNodes", script)
            self.assertIn("搭建视图", script)
            self.assertIn("setChatMode", script)
            self.assertIn("isConfiguratorMessage", script)
            self.assertIn("sendConfiguratorMessage", script)
            self.assertIn("/api/configurator/compose", script)
            self.assertNotIn("configuratorDraftAgents", script)
            self.assertNotIn("configuratorDraftWorkflow", script)
            self.assertNotIn("configurator-draft", script)
            self.assertNotIn("agentConfiguratorPrompt", script)
            self.assertNotIn("workflowConfiguratorPrompt", script)
            self.assertIn("/api/agents/draft", script)
            self.assertIn("/api/agents/proposal", script)
            self.assertIn("/api/agents/apply", script)
            self.assertIn("data-workflow-action", script)
            self.assertIn("data-agent-action", script)
            self.assertIn("/api/workflow/draft", script)
            self.assertIn("/api/workflow/proposal", script)
            self.assertIn("/api/workflow/apply", script)
            self.assertIn("/api/templates", script)
            self.assertIn("/api/templates/apply", script)
            self.assertIn("/api/templates/save", script)
            self.assertIn("renderTemplateLibrary", script)
            self.assertIn("data-template-action", script)
            self.assertIn("add-domain-review", script)
            self.assertIn("actual_input", script)
            self.assertIn("actual_output", script)
            self.assertIn("工作过程与结果", script)
            self.assertIn("转交流程审查", script)
            self.assertNotIn("workflowDetailGraph", script)
            self.assertIn("displayLabel", script)
            self.assertIn("data(displayLabel)", script)
            self.assertNotIn("cytoscapeDagre", script)
            self.assertNotIn('name: "circle"', script)
            self.assertNotIn("workflowCircleLayoutOptions", script)
            self.assertNotIn("workflowRingRadius", script)
            self.assertNotIn("workflow-graph-center", script)
            self.assertNotIn('name: "dagre"', script)
            self.assertNotIn('rankDir: "LR"', script)
            self.assertNotIn("wheelSensitivity", script)
            self.assertNotIn("workflowGraphMarkup", script)
            self.assertNotIn("workflow-graph-svg", script)
            self.assertNotIn("foreignObject", script)
            self.assertNotIn("workflowChainMarkup", script)
            self.assertNotIn("renderTimeline", script)
            self.assertIn("参考文献", script)
            self.assertIn("async: true", script)

            css = request.urlopen(f"{base_url}/assets/app.css", timeout=10).read().decode("utf-8")
            self.assertIn(".workflow-graph", css)
            self.assertIn(".workflow-graph-canvas", css)
            self.assertIn(".workflow-graph-legend", css)
            self.assertIn(".workflow-selection", css)
            self.assertIn(".workflow-selection.selection-focus", css)
            self.assertIn(".workflow-selection-grid", css)
            self.assertIn(".workflow-editor", css)
            self.assertIn(".workflow-edit-grid", css)
            self.assertIn(".agent-editor", css)
            self.assertIn(".agent-edit-grid", css)
            self.assertIn(".template-library", css)
            self.assertIn(".conversation-modes", css)
            self.assertIn(".conversation-mode-hint", css)
            self.assertNotIn(".workflow-configurator", css)
            self.assertNotIn(".agent-configurator", css)
            self.assertIn("node.selected", script)
            self.assertIn("edge.selected", script)
            self.assertIn("grid-column: 1 / -1", css)
            self.assertNotIn("overflow-x: auto", css)
            self.assertNotIn(".workflow-graph-center", css)
            self.assertNotIn(".workflow-graph-svg", css)
            self.assertNotIn(".workflow-chain", css)
            self.assertNotIn(".timeline", css)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_web_static_assets_are_served_as_utf8(self):
        server = create_server(host="127.0.0.1", port=0, config_path="agent.yaml", workspace_root=".")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"

            html = request.urlopen(f"{base_url}/", timeout=10)
            script = request.urlopen(f"{base_url}/assets/app.js", timeout=10)

            self.assertIn("charset=utf-8", html.headers["Content-Type"])
            self.assertIn("charset=utf-8", script.headers["Content-Type"])
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_rag_query_api_supports_method_switching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "hea.md").write_text(
                "High entropy alloy strength depends on microstructure and processing history.",
                encoding="utf-8",
            )
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: FakeDeepSeekClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                payload = _post_json(
                    f"{base_url}/api/rag/query",
                    {
                        "query": "高熵合金强度受哪些微观组织因素影响",
                        "methods": ["semantic", "graph"],
                        "top_k": 3,
                    },
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        self.assertEqual(payload["query"], "高熵合金强度受哪些微观组织因素影响")
        self.assertEqual(payload["methods"], ["semantic", "graph"])
        self.assertGreaterEqual(payload["count"], 1)
        self.assertTrue(payload["results"][0]["source"].endswith("hea.md"))
        self.assertIn("semantic", payload["results"][0]["score_breakdown"])
        self.assertIn("graph", payload["results"][0]["retrieval_method"])


def _get_json(url):
    return json.loads(request.urlopen(url, timeout=10).read().decode("utf-8"))


def _post_json(url, payload, *, expect_status=200):
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = request.urlopen(req, timeout=10)
        if expect_status != 200:
            raise AssertionError(f"expected HTTP {expect_status}, got 200")
        return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if not hasattr(exc, "code"):
            raise
        if exc.code != expect_status:
            raise
        return json.loads(exc.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
