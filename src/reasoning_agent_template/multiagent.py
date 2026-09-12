from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Protocol

from reasoning_agent_template.agents_spec import AgentsSpec, AgentsSpecStore
from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.evidence import EvidenceLedger
from reasoning_agent_template.gates import GatePolicy
from reasoning_agent_template.llm import ChatMessage, DeepSeekChatClient, LLMRequestError
from reasoning_agent_template.memory import (
    LongTermMemoryStore,
    ShortTermConversationMemory,
    deny_knowledge_memory_write,
    explicit_memory_candidate,
)
from reasoning_agent_template.models import AgentState, stable_hash, utc_now
from reasoning_agent_template.risk import classify_evidence_requirement, is_explicit_evidence_request
from reasoning_agent_template.skills import SkillRegistry
from reasoning_agent_template.sessions import SessionStore
from reasoning_agent_template.workflow import TemplateCoordinator
from reasoning_agent_template.workflow_spec import WorkflowNodeSpec, WorkflowSpec, WorkflowSpecStore


AGENT_ROLES = [
    ("coordinator", "负责运行调度，并拥有状态机。"),
    ("planner", "生成边界清晰、可审计的推理计划。"),
    ("retriever", "检索本地知识库、论文/外部来源，并记录证据。"),
    ("reasoner", "基于检索证据生成答案草案。"),
    ("critic", "审计证据绑定和门禁结果。"),
    ("memory", "提出长期记忆沉淀候选，不直接写入保护资产。"),
    ("evolver", "生成自进化提案，不直接修改核心技能。"),
    ("reviewer", "审查 Coordinator 的难度、风险、证据需求和工作流选择，并可升级过松判断。"),
]

WORKFLOW_DEFINITIONS = [
    ("intake", "coordinator", "接收目标并建立运行上下文", "用户问题", "规范化目标"),
    ("plan", "planner", "生成可审计的推理计划", "规范化目标", "计划步骤"),
    ("retrieve", "retriever", "检索本地知识库、外部论文和用户经验证据", "计划步骤", "证据候选"),
    ("reason", "reasoner", "基于证据生成草案答案", "证据候选", "答案草案"),
    ("evidence_audit", "critic", "检查关键结论是否绑定证据", "答案草案", "证据审计结果"),
    ("gate", "critic", "执行风险、证据和审批门禁", "证据审计结果", "门禁决策"),
    ("act_or_answer", "coordinator", "根据门禁结果行动或回答", "门禁决策", "可发布结果"),
    ("verify", "critic", "验证状态、证据和输出一致性", "可发布结果", "验证记录"),
    ("consolidate", "memory", "只生成记忆/演进候选，不直接写保护资产", "验证记录", "沉淀候选"),
    ("respond", "coordinator", "输出最终答案和调试遥测", "沉淀候选", "最终响应"),
]

WORKFLOW_CHECKPOINTS = ["retrieve", "evidence_audit", "gate", "verify", "consolidate"]
WORKFLOW_STAGE_ORDER = {stage: index for index, (stage, *_rest) in enumerate(WORKFLOW_DEFINITIONS)}
STAGE_AGENTS = {stage: agent for stage, agent, *_rest in WORKFLOW_DEFINITIONS}
WORKFLOW_CONTROL_EDGES = [
    {"from": "intake", "to": "plan", "type": "flow", "label": "normalized goal"},
    {"from": "plan", "to": "retrieve", "type": "branch", "label": "evidence required"},
    {"from": "plan", "to": "reason", "type": "branch", "label": "routine path"},
    {"from": "retrieve", "to": "reason", "type": "flow", "label": "evidence context"},
    {"from": "reason", "to": "evidence_audit", "type": "flow", "label": "claim audit"},
    {"from": "evidence_audit", "to": "gate", "type": "flow", "label": "audit result"},
    {"from": "evidence_audit", "to": "retrieve", "type": "retry", "label": "evidence gap"},
    {"from": "gate", "to": "act_or_answer", "type": "flow", "label": "gate decision"},
    {"from": "gate", "to": "retrieve", "type": "retry", "label": "gate blocked"},
    {"from": "act_or_answer", "to": "verify", "type": "flow", "label": "answer/action"},
    {"from": "verify", "to": "consolidate", "type": "flow", "label": "verified output"},
    {"from": "verify", "to": "reason", "type": "revise", "label": "verification failed"},
    {"from": "consolidate", "to": "respond", "type": "flow", "label": "publishable result"},
    {"from": "consolidate", "to": "evidence_audit", "type": "loop", "label": "proposal audit"},
]


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[ChatMessage], *, temperature: float, max_tokens: int) -> Any:
        ...


class MultiAgentOrchestrator:
    """Wrap TemplateCoordinator with explicit multi-agent telemetry."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        workspace_root: str | Path,
        llm_client_factory: Callable[[AgentConfig], ChatClient] | None = None,
        session_store: SessionStore | None = None,
    ):
        self.config = config
        self.workspace_root = Path(workspace_root)
        self.llm_client_factory = llm_client_factory
        self.session_store = session_store
        self.workflow_store = WorkflowSpecStore(self.workspace_root, self.config.runtime)
        self.agents_store = AgentsSpecStore(self.workspace_root, self.config.runtime)
        self.short_term_memories: dict[str, ShortTermConversationMemory] = {}
        self._status_lock = RLock()
        self._active_run: dict[str, Any] | None = None
        self._last_run: dict[str, Any] | None = None

    def run(self, message: str, *, thread_id: str = "default") -> dict[str, Any]:
        thread_id = _normalize_thread_id(thread_id)
        short_term_memory_store = self._short_term_memory_for(thread_id)
        started_at = utc_now()
        run_started = perf_counter()
        run_id = f"run_{stable_hash(started_at + message + thread_id)[:12]}"
        events: list[dict[str, Any]] = []
        self._set_active_phase(
            run_id=run_id,
            thread_id=thread_id,
            started_at=started_at,
            message=message,
            agent="coordinator",
            phase="routing",
            stage="intake",
            hint="Coordinator 正在自主判断任务难度、风险等级和证据工作流。",
            events=events,
        )
        try:
            short_term_before = short_term_memory_store.snapshot(limit=int(self.config.memory.get("prompt_turns", 6)))
            long_term_before = self._load_long_term_memories(limit=int(self.config.memory.get("prompt_long_term_items", 12)))
            coordinator_route, route_events = self._call_coordinator_router(
                message,
                short_term_memory=short_term_before,
                long_term_memory=long_term_before,
            )
            events.extend(route_events)
            self._set_active_phase(
                run_id=run_id,
                thread_id=thread_id,
                started_at=started_at,
                message=message,
                agent="reviewer",
                phase="route_review",
                stage="intake",
                hint="Reviewer 正在复核 Coordinator 的难度、风险和证据策略。",
                events=events,
            )
            routing_decision, reviewer_payload, reviewer_events = self._call_reviewer(
                message,
                coordinator_route,
                short_term_memory=short_term_before,
                long_term_memory=long_term_before,
            )
            events.extend(reviewer_events)
            workflow_goal = str(routing_decision.get("retrieval_query") or "").strip()
            if not workflow_goal:
                workflow_goal = _contextual_goal_for_evidence_followup(message, short_term_before)
            self._set_active_phase(
                run_id=run_id,
                thread_id=thread_id,
                started_at=started_at,
                message=message,
                agent="coordinator",
                phase="workflow",
                stage="intake",
                hint="状态机已启动，正在进入工作流节点链。",
                events=events,
            )
            result = TemplateCoordinator(
                config=self.config,
                workspace_root=self.workspace_root,
                event_callback=lambda progress_state, event: self._set_active_state(
                    run_id=run_id,
                    thread_id=thread_id,
                    started_at=started_at,
                    message=message,
                    state=progress_state,
                    agent=str(event.get("agent", "coordinator")),
                    phase=str(event.get("kind", "workflow")),
                    stage=str(event.get("stage", progress_state.current_stage)),
                    hint=_event_hint(event),
                    events=[*events, *progress_state.stage_events],
                ),
            ).run(
                workflow_goal,
                routing_decision=routing_decision,
            )
            events.extend(self._events(result.state))
            self._set_active_state(
                run_id=run_id,
                thread_id=thread_id,
                started_at=started_at,
                message=message,
                state=result.state,
                agent="reasoner",
                phase="final_answer",
                stage="respond",
                hint="Reasoner 正在调用 DeepSeek 生成最终自然语言回答。",
                events=events,
            )
            answer, llm_status, llm_events = self._call_deepseek(
                message,
                result.state,
                short_term_memory=short_term_before,
                long_term_memory=long_term_before,
            )
            answer = _sanitize_answer_for_user(answer)
            events.extend(llm_events)

            memory_writes = self._write_explicit_long_term_memory(message)
            short_term_memory_store.append(user=message, assistant=answer, run_id=run_id)
            if self.session_store is not None:
                session_messages: list[dict[str, Any]] = []
                for item in short_term_memory_store.snapshot():
                    session_messages.extend(
                        [
                            {"role": "user", "content": item["user"], "run_id": item["run_id"]},
                            {"role": "assistant", "content": item["assistant"], "run_id": item["run_id"]},
                        ]
                    )
                self.session_store.record_snapshot(
                    session_id=thread_id,
                    messages=session_messages,
                    events=events[-120:],
                    status="completed",
                    metadata={"run_id": run_id, "thread_id": thread_id},
                )
            long_term_after = self._load_long_term_memories(
                limit=int(self.config.memory.get("prompt_long_term_items", 12))
            )
        except Exception:
            with self._status_lock:
                self._active_run = None
            raise
        events.append(
            {
                "time": utc_now(),
                "agent": "memory",
                "kind": "short_term_memory",
                "message": f"线程 {thread_id} 短期记忆已记录 {short_term_memory_store.count()} 轮",
            }
        )
        for write in memory_writes:
            events.append(
                {
                    "time": utc_now(),
                    "agent": "memory",
                    "kind": "long_term_memory",
                    "message": f"{write.partition}/{write.key} 写入状态={write.decision.status}",
                }
            )
        skills = SkillRegistry(Path(self.config.skills.get("directory", "skills"))).load()
        completed_at = utc_now()
        payload = {
            "run_id": run_id,
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": round((perf_counter() - run_started) * 1000, 2),
            "question": message,
            "thread_id": thread_id,
            "answer": answer,
            "runtime": {
                "agent": self.config.identity.get("name", "reasoning-agent-template"),
                "workspace": str(self.workspace_root.resolve()),
                "llm": llm_status,
            },
            "routing": {
                "source": result.state.routing_source,
                "difficulty": result.state.difficulty,
                "workflow": result.state.workflow_variant,
                "confidence": result.state.routing_confidence,
                "decision": dict(result.state.routing_decision),
                "coordinator": dict(coordinator_route),
            },
            "reviewer": {
                "status": result.state.reviewer_status,
                "decision": dict(result.state.reviewer_decision),
                "findings": list(result.state.reviewer_decision.get("findings", []))
                if isinstance(result.state.reviewer_decision.get("findings", []), list)
                else [],
            },
            "agents": self._agents(result.state),
            "state_machine": self._state_machine(result.state),
            "workflow": self._workflow(result.state),
            "evidence": {
                "count": len(result.evidence),
                "mode": result.state.evidence_mode,
                "strictness": result.state.evidence_strictness,
                "status": result.state.evidence_status,
                "required": result.state.evidence_mode == "required",
                "risk_level": result.state.risk_level,
                "category": result.state.evidence_category,
                "reasons": list(result.state.evidence_reasons),
                "sources": list(result.state.evidence_sources),
                "qualified_evidence_ids": list(result.state.qualified_evidence_ids),
                "unqualified_evidence_ids": list(result.state.unqualified_evidence_ids),
                "consolidation_proposals": list(result.state.evidence_consolidation_proposals),
                "references": _evidence_references(result.evidence),
                "items": [item.to_dict() for item in result.evidence],
            },
            "rag": {
                "query": message,
                "count": len(result.state.retrieval_results),
                "methods": list(result.state.rag_methods),
                "diagnostics": list(result.state.rag_diagnostics),
                "results": [self._chunk_artifact(chunk) for chunk in result.state.retrieval_results],
            },
            "external_evidence": {
                "query": message,
                "count": len(result.state.external_results),
                "attempted_sources": list(result.state.external_search_attempted),
                "diagnostics": list(result.state.external_search_diagnostics),
                "results": [
                    {
                        "source": chunk.source,
                        "span": chunk.span,
                        "score": chunk.score,
                        "content_hash": chunk.content_hash,
                        "evidence_id": chunk.evidence_id,
                        "text": chunk.text,
                    }
                    for chunk in result.state.external_results
                ],
            },
            "gates": {
                "count": len(result.gate_decisions),
                "decisions": [decision.to_dict() for decision in result.gate_decisions],
            },
            "memory": {
                "policy": "short-term rolling window; explicit long-term writes require gate",
                "partitions": self.config.memory.get("partitions", []),
                "read_only": self.config.memory.get("read_only_partitions", []),
                "boundaries": MEMORY_BOUNDARIES,
                "pending_consolidation": list(result.state.pending_consolidation),
                "short_term": {
                    "thread_id": thread_id,
                    "turns": short_term_memory_store.count(),
                    "items": short_term_memory_store.snapshot(limit=int(self.config.memory.get("prompt_turns", 6))),
                },
                "long_term": {
                    "records": len(long_term_after),
                    "items": long_term_after,
                    "writes": len(memory_writes),
                    "write_decisions": [
                        {
                            "partition": write.partition,
                            "key": write.key,
                            "status": write.decision.status,
                            "reasons": write.decision.reasons,
                            "required_evidence": write.decision.required_evidence,
                        }
                        for write in memory_writes
                    ],
                },
            },
            "skills": {
                "enabled": self.config.skills.get("enabled", []),
                "loaded": sorted(skills),
                "count": len(skills),
            },
            "events": events,
        }
        with self._status_lock:
            self._last_run = copy.deepcopy(payload)
            self._active_run = None
        return payload

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            if self._active_run is not None:
                return copy.deepcopy(self._active_run)
            if self._last_run is not None:
                return copy.deepcopy(self._last_run)
        return self._idle_status()

    def record_failure(
        self,
        *,
        message: str,
        thread_id: str,
        error: Exception,
        phase: str,
    ) -> None:
        with self._status_lock:
            payload = copy.deepcopy(self._active_run) if self._active_run is not None else self._idle_status()
            payload.update(
                {
                    "status": "failed",
                    "question": message,
                    "thread_id": thread_id,
                    "error": str(error),
                    "type": type(error).__name__,
                    "phase": phase,
                    "working_hint": {
                        "agent": "coordinator",
                        "phase": phase,
                        "stage": payload.get("state_machine", {}).get("current", "unknown"),
                        "message": f"运行失败：{error}",
                        "completed": True,
                    },
                }
            )
            self._last_run = payload
            self._active_run = None

    def _idle_status(self) -> dict[str, Any]:
        skills = SkillRegistry(Path(self.config.skills.get("directory", "skills"))).load()
        return {
            "status": "ready",
            "runtime": {
                "agent": self.config.identity.get("name", "reasoning-agent-template"),
                "workspace": str(self.workspace_root.resolve()),
                "model": self.config.models.get("worker", {}),
                "llm": {
                    "required": True,
                    "provider": self.config.models.get("worker", {}).get("provider", "deepseek"),
                    "model": self.config.models.get("worker", {}).get("model", "deepseek-v4-flash"),
                    "secret_sources": ["configs/secrets.local.json", "DEEPSEEK_API_KEY"],
                    "configured": DeepSeekChatClient.is_configured(self.config),
                },
            },
            "agents": self._idle_agents(),
            "routing": {
                "source": "idle",
                "difficulty": "idle",
                "workflow": "idle",
                "confidence": 0.0,
                "decision": {},
                "coordinator": {},
            },
            "reviewer": {"status": "idle", "decision": {}, "findings": []},
            "state_machine": {
                "active": False,
                "current": "idle",
                "trace": [],
                "configured": [
                    "intake",
                    "plan",
                    "retrieve",
                    "reason",
                    "evidence_audit",
                    "gate",
                    "act_or_answer",
                    "verify",
                    "consolidate",
                    "respond",
                ],
            },
            "workflow": self.workflow_status(),
            "evidence": {
                "count": 0,
                "mode": "idle",
                "strictness": "none",
                "status": "idle",
                "required": False,
                "risk_level": "none",
                "category": "idle",
                "reasons": [],
                "sources": [],
                "qualified_evidence_ids": [],
                "unqualified_evidence_ids": [],
                "consolidation_proposals": [],
                "references": [],
                "items": [],
            },
            "rag": {"count": 0, "methods": [], "diagnostics": [], "results": []},
            "external_evidence": {"count": 0, "attempted_sources": [], "diagnostics": [], "results": []},
            "gates": {"count": 0, "decisions": []},
            "memory": {
                "partitions": self.config.memory.get("partitions", []),
                "read_only": self.config.memory.get("read_only_partitions", []),
                "boundaries": MEMORY_BOUNDARIES,
                "short_term": {
                    "thread_id": "default",
                    "turns": self._short_term_memory_for("default").count(),
                    "items": self._short_term_memory_for("default").snapshot(),
                    "threads": sorted(self.short_term_memories),
                },
                "long_term": {
                    "records": len(self._load_long_term_memories()),
                    "items": self._load_long_term_memories(),
                    "writes": 0,
                    "write_decisions": [],
                },
            },
            "skills": {"loaded": sorted(skills), "count": len(skills)},
        }

    def _set_active_phase(
        self,
        *,
        run_id: str,
        thread_id: str,
        started_at: str,
        message: str,
        agent: str,
        phase: str,
        stage: str,
        hint: str,
        events: list[dict[str, Any]],
    ) -> None:
        payload = self._running_payload(
            run_id=run_id,
            thread_id=thread_id,
            started_at=started_at,
            message=message,
            agent=agent,
            phase=phase,
            stage=stage,
            hint=hint,
            events=events,
        )
        with self._status_lock:
            self._active_run = payload

    def _set_active_state(
        self,
        *,
        run_id: str,
        thread_id: str,
        started_at: str,
        message: str,
        state: AgentState,
        agent: str,
        phase: str,
        stage: str,
        hint: str,
        events: list[dict[str, Any]],
    ) -> None:
        payload = self._running_payload(
            run_id=run_id,
            thread_id=thread_id,
            started_at=started_at,
            message=message,
            agent=agent,
            phase=phase,
            stage=stage,
            hint=hint,
            events=events,
            state=state,
        )
        with self._status_lock:
            self._active_run = payload

    def _running_payload(
        self,
        *,
        run_id: str,
        thread_id: str,
        started_at: str,
        message: str,
        agent: str,
        phase: str,
        stage: str,
        hint: str,
        events: list[dict[str, Any]],
        state: AgentState | None = None,
    ) -> dict[str, Any]:
        payload = self._idle_status()
        payload.update(
            {
                "run_id": run_id,
                "status": "running",
                "started_at": started_at,
                "question": message,
                "thread_id": thread_id,
                "working_hint": {
                    "agent": agent,
                    "phase": phase,
                    "stage": stage,
                    "message": hint,
                    "completed": False,
                },
                "events": list(events)[-120:],
            }
        )
        payload["runtime"]["llm"]["status"] = "running"
        payload["agents"] = self._running_agents(state=state, active_agent=agent, active_stage=stage, hint=hint)
        if state is None:
            payload["state_machine"]["active"] = True
            payload["state_machine"]["current"] = stage
            payload["workflow"] = self._activate_workflow(self.workflow_status(), active_stage=stage)
            return payload

        payload["routing"] = {
            "source": state.routing_source,
            "difficulty": state.difficulty,
            "workflow": state.workflow_variant,
            "confidence": state.routing_confidence,
            "decision": dict(state.routing_decision),
            "coordinator": {},
        }
        payload["reviewer"] = {
            "status": state.reviewer_status,
            "decision": dict(state.reviewer_decision),
            "findings": list(state.reviewer_decision.get("findings", []))
            if isinstance(state.reviewer_decision.get("findings", []), list)
            else [],
        }
        payload["state_machine"] = self._activate_state_machine(self._state_machine(state), active_stage=stage)
        payload["workflow"] = self._activate_workflow(self._workflow(state), active_stage=stage)
        payload["evidence"] = self._evidence_monitor(state)
        payload["rag"] = self._rag_monitor(state, query=message)
        payload["external_evidence"] = self._external_monitor(state, query=message)
        payload["gates"] = {
            "count": len(state.gate_decisions),
            "decisions": [decision.to_dict() for decision in state.gate_decisions],
        }
        payload["memory"]["pending_consolidation"] = list(state.pending_consolidation)
        return payload

    def _running_agents(
        self,
        *,
        state: AgentState | None,
        active_agent: str,
        active_stage: str,
        hint: str,
    ) -> list[dict[str, Any]]:
        if state is not None:
            agents = self._agents(state)
        else:
            role_descriptions = {name: description for name, description in AGENT_ROLES}
            for node in self._workflow_spec().nodes:
                role_descriptions.setdefault(node.agent, f"Workflow agent for {node.label or node.id}.")
            agents = [
                {"name": name, "description": description, "status": "idle", "active": False, "current_stage": ""}
                for name, description in role_descriptions.items()
            ]
        for item in agents:
            if item["name"] == active_agent:
                item["status"] = "active"
                item["active"] = True
                item["current_stage"] = active_stage
                item["last_event"] = hint
            else:
                item["active"] = False
        return agents

    def _activate_state_machine(self, state_machine: dict[str, Any], *, active_stage: str) -> dict[str, Any]:
        state_machine["active"] = True
        state_machine["current"] = active_stage
        for stage in state_machine.get("stages", []):
            if stage.get("name") == active_stage:
                stage["status"] = "active"
        return state_machine

    def _activate_workflow(self, workflow: dict[str, Any], *, active_stage: str) -> dict[str, Any]:
        workflow["active"] = True
        workflow["status"] = "running"
        workflow["current"] = active_stage
        for node in workflow.get("nodes", []):
            if node.get("id") == active_stage:
                node["status"] = "active"
                node["effective_status"] = "active"
                node["skip_reason"] = ""
        return workflow

    def _evidence_monitor(self, state: AgentState) -> dict[str, Any]:
        return {
            "count": len(state.evidence),
            "mode": state.evidence_mode,
            "strictness": state.evidence_strictness,
            "status": state.evidence_status,
            "required": state.evidence_mode == "required",
            "risk_level": state.risk_level,
            "category": state.evidence_category,
            "reasons": list(state.evidence_reasons),
            "sources": list(state.evidence_sources),
            "qualified_evidence_ids": list(state.qualified_evidence_ids),
            "unqualified_evidence_ids": list(state.unqualified_evidence_ids),
            "consolidation_proposals": list(state.evidence_consolidation_proposals),
            "references": _evidence_references(state.evidence),
            "items": [item.to_dict() for item in state.evidence],
        }

    def _rag_monitor(self, state: AgentState, *, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "count": len(state.retrieval_results),
            "methods": list(state.rag_methods),
            "diagnostics": list(state.rag_diagnostics),
            "results": [self._chunk_artifact(chunk) for chunk in state.retrieval_results],
        }

    def _external_monitor(self, state: AgentState, *, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "count": len(state.external_results),
            "attempted_sources": list(state.external_search_attempted),
            "diagnostics": list(state.external_search_diagnostics),
            "results": [
                {
                    "source": chunk.source,
                    "span": chunk.span,
                    "score": chunk.score,
                    "content_hash": chunk.content_hash,
                    "evidence_id": chunk.evidence_id,
                    "text": chunk.text,
                }
                for chunk in state.external_results
            ],
        }

    def workflow_status(self) -> dict[str, Any]:
        spec = self._workflow_spec()
        return {
            "status": "idle",
            "active": False,
            "current": "idle",
            "variant": "idle",
            "spec": {
                "name": spec.name,
                "revision": spec.revision,
                "version": spec.version,
                "start_node": spec.start_node,
                "terminal_nodes": list(spec.terminal_nodes),
            },
            "nodes": [self._workflow_node_from_spec(node) for node in spec.nodes],
            "edges": self._workflow_edges(spec),
            "checkpoints": spec.checkpoints(),
        }

    def _workflow_spec(self) -> WorkflowSpec:
        return self.workflow_store.load()

    def _agents_spec(self) -> AgentsSpec:
        return self.agents_store.load()

    def _workflow_edges(self, spec: WorkflowSpec) -> list[dict[str, Any]]:
        return [
            {
                "id": edge.id,
                "from": edge.from_node,
                "to": edge.to_node,
                "type": edge.type,
                "label": edge.condition,
                "condition": edge.condition,
                "handoff_contract": dict(edge.handoff_contract),
                "gate_policy": dict(edge.gate_policy),
                "planner_contract": dict(edge.planner_contract),
                "reviewer_required": edge.reviewer_required,
            }
            for edge in spec.edges
        ]

    def _workflow_node_from_spec(
        self,
        node: WorkflowNodeSpec,
        *,
        state: AgentState | None = None,
        completed: set[str] | None = None,
        durations: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return self._workflow_node(
            node.id,
            node.agent,
            node.description,
            node.input_contract,
            node.output_contract,
            node_spec=node,
            state=state,
            completed=completed,
            durations=durations,
        )

    def _workflow_node(
        self,
        stage: str,
        agent: str,
        description: str,
        input_name: str,
        output_name: str,
        *,
        node_spec: WorkflowNodeSpec | None = None,
        state: AgentState | None = None,
        completed: set[str] | None = None,
        durations: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        completed = completed or set()
        durations = durations or {}
        status = "completed" if stage in completed else "pending"
        observed = "waiting"
        effective_status = status
        work_done = False
        skip_reason = ""
        if state is not None:
            observed = self._workflow_observed(stage, state)
            effect = self._workflow_effect(stage, state, completed=completed)
            effective_status = effect["effective_status"]
            work_done = effect["work_done"]
            skip_reason = effect["skip_reason"]
        return {
            "id": stage,
            "agent": agent,
            "description": description,
            "input": input_name,
            "output": output_name,
            "status": status,
            "effective_status": effective_status,
            "work_done": work_done,
            "skip_reason": skip_reason,
            "checkpoint": node_spec.checkpoint if node_spec is not None else stage in WORKFLOW_CHECKPOINTS,
            "observed": observed,
            "duration_ms": durations.get(stage),
            "artifacts": self._workflow_artifacts(stage, state),
            "label": node_spec.label if node_spec is not None else stage,
            "work": node_spec.work if node_spec is not None else description,
            "handler_kind": node_spec.handler_kind if node_spec is not None else "builtin",
            "handler": node_spec.handler if node_spec is not None else stage,
            "gate_policy": dict(node_spec.gate_policy) if node_spec is not None else {},
            "ui": dict(node_spec.ui) if node_spec is not None else {},
        }

    def _agents(self, state: AgentState) -> list[dict[str, Any]]:
        metrics = {
            "coordinator": len(state.stage_trace),
            "planner": len(state.plan),
            "retriever": len(state.retrieval_results) + len(state.external_results),
            "reasoner": 1 if state.answer else 0,
            "reviewer": 1 if state.reviewer_status != "not_run" else 0,
            "critic": len(state.verification_notes),
            "memory": len(state.pending_consolidation),
            "evolver": 0,
        }
        agents_spec = self._agents_spec()
        role_descriptions = {agent.id: agent.description for agent in agents_spec.agents}
        role_specs = agents_spec.agent_map()
        for node in self._workflow_spec().nodes:
            role_descriptions.setdefault(node.agent, f"Workflow agent for {node.label or node.id}.")
        return [
            {
                "name": name,
                "label": role_specs.get(name).label if name in role_specs else name,
                "description": description,
                "status": "completed" if name != "evolver" else "idle",
                "active": False,
                "current_stage": state.current_stage if name == "coordinator" else name,
                "metric": metrics.get(name, 0),
                "last_event": self._last_event_for(name, state),
                "responsibilities": list(role_specs.get(name).responsibilities) if name in role_specs else [],
                "model_role": role_specs.get(name).model_role if name in role_specs else "worker",
                "tools": list(role_specs.get(name).tools) if name in role_specs else [],
                "memory_access": list(role_specs.get(name).memory_access) if name in role_specs else [],
                "workflow_nodes": list(role_specs.get(name).workflow_nodes) if name in role_specs else [],
                "protected": name in agents_spec.protected_agents,
            }
            for name, description in role_descriptions.items()
        ]

    def _idle_agents(self) -> list[dict[str, Any]]:
        agents_spec = self._agents_spec()
        return [
            {
                "name": agent.id,
                "label": agent.label,
                "status": "idle",
                "description": agent.description,
                "active": False,
                "current_stage": agent.id,
                "metric": 0,
                "last_event": "",
                "responsibilities": list(agent.responsibilities),
                "model_role": agent.model_role,
                "tools": list(agent.tools),
                "memory_access": list(agent.memory_access),
                "workflow_nodes": list(agent.workflow_nodes),
                "protected": agent.id in agents_spec.protected_agents,
            }
            for agent in agents_spec.agents
        ]

    def _state_machine(self, state: AgentState) -> dict[str, Any]:
        trace = list(state.stage_trace)
        durations = self._stage_durations(state)
        return {
            "active": False,
            "current": state.current_stage,
            "trace": trace,
            "stages": [
                {
                    "name": stage,
                    "status": "completed" if stage in trace else "pending",
                    "index": index,
                    "duration_ms": durations.get(stage),
                }
                for index, stage in enumerate(trace)
            ],
        }

    def _workflow(self, state: AgentState) -> dict[str, Any]:
        spec = self._workflow_spec()
        trace = list(state.stage_trace)
        completed = set(trace)
        durations = self._stage_durations(state)
        return {
            "status": "completed" if trace and trace[-1] in spec.terminal_nodes else "running",
            "active": False,
            "current": state.current_stage,
            "variant": state.workflow_variant,
            "spec": {
                "name": spec.name,
                "revision": spec.revision,
                "version": spec.version,
                "start_node": spec.start_node,
                "terminal_nodes": list(spec.terminal_nodes),
            },
            "nodes": [
                self._workflow_node_from_spec(
                    node,
                    state=state,
                    completed=completed,
                    durations=durations,
                )
                for node in spec.nodes
            ],
            "edges": self._workflow_edges(spec),
            "checkpoints": spec.checkpoints(),
        }

    def _workflow_effect(self, stage: str, state: AgentState, *, completed: set[str]) -> dict[str, Any]:
        if stage == "retrieve" and state.evidence_mode != "required":
            return {
                "effective_status": "skipped",
                "work_done": False,
                "skip_reason": "evidence_not_required",
            }
        if stage not in completed:
            return {"effective_status": "pending", "work_done": False, "skip_reason": ""}
        if stage == "evidence_audit" and state.evidence_mode != "required":
            return {
                "effective_status": "skipped",
                "work_done": False,
                "skip_reason": "evidence_not_required",
            }
        if stage == "act_or_answer" and state.gate_decisions and state.gate_decisions[-1].status == "allow":
            return {
                "effective_status": "noop",
                "work_done": False,
                "skip_reason": "gate_allowed_existing_answer",
            }
        if stage == "verify" and state.gate_decisions and state.gate_decisions[-1].status != "allow":
            return {
                "effective_status": "noop",
                "work_done": False,
                "skip_reason": "gate_not_allowed",
            }
        if stage == "consolidate" and not state.evidence_consolidation_proposals:
            return {
                "effective_status": "noop",
                "work_done": False,
                "skip_reason": "no_qualified_evidence_for_consolidation",
            }
        work_done_by_stage = {
            "intake": bool(state.routing_source or state.evidence_mode),
            "plan": bool(state.plan),
            "retrieve": bool(
                state.retrieval_results
                or state.external_results
                or state.external_search_attempted
                or "rag" in state.evidence_sources
            ),
            "reason": bool(state.answer),
            "evidence_audit": bool(state.verification_notes or state.evidence_status != "pending"),
            "gate": bool(state.gate_decisions),
            "act_or_answer": bool(state.answer),
            "verify": bool(state.verification_notes),
            "consolidate": bool(state.evidence_consolidation_proposals),
            "respond": bool(state.answer),
        }
        work_done = bool(work_done_by_stage.get(stage, stage in completed))
        return {
            "effective_status": "completed" if work_done else "noop",
            "work_done": work_done,
            "skip_reason": "" if work_done else "no_observable_output",
        }

    def _workflow_observed(self, stage: str, state: AgentState) -> str:
        if stage == "plan":
            return f"{len(state.plan)} 个计划步骤"
        if stage == "retrieve":
            return f"{len(state.retrieval_results)} 条本地 RAG，{len(state.external_results)} 条外部证据"
        if stage == "evidence_audit":
            return f"{len(state.evidence)} 条证据"
        if stage == "gate":
            return f"{len(state.gate_decisions)} 个门禁决策"
        if stage == "verify":
            return "; ".join(state.verification_notes) or "等待验证"
        if stage == "consolidate":
            return "; ".join(state.pending_consolidation) or "无沉淀候选"
        if stage == "respond":
            return "已生成最终响应" if state.answer else "未响应"
        return "已执行" if stage in state.stage_trace else "等待执行"

    def _workflow_artifacts(self, stage: str, state: AgentState | None) -> dict[str, Any]:
        empty = {"actual_input": {}, "actual_output": {}, "process": [], "handoff": {}}
        if state is None:
            return empty
        artifacts_by_stage = {
            "intake": self._intake_artifacts,
            "plan": self._plan_artifacts,
            "retrieve": self._retrieve_artifacts,
            "reason": self._reason_artifacts,
            "evidence_audit": self._evidence_audit_artifacts,
            "gate": self._gate_artifacts,
            "act_or_answer": self._act_or_answer_artifacts,
            "verify": self._verify_artifacts,
            "consolidate": self._consolidate_artifacts,
            "respond": self._respond_artifacts,
        }
        builder = artifacts_by_stage.get(stage)
        artifacts = builder(state) if builder else self._generic_stage_artifacts(stage, state)
        artifacts.setdefault("actual_input", {})
        artifacts.setdefault("actual_output", {})
        artifacts.setdefault("process", [])
        artifacts.setdefault("handoff", self._stage_handoff(stage))
        return artifacts

    def _generic_stage_artifacts(self, stage: str, state: AgentState) -> dict[str, Any]:
        node = self._workflow_spec().node_map().get(stage)
        return {
            "actual_input": {
                "input_contract": node.input_contract if node else "",
                "current_answer": state.answer,
                "gate_decisions": [decision.to_dict() for decision in state.gate_decisions],
            },
            "actual_output": {
                "output_contract": node.output_contract if node else "",
                "handler_kind": node.handler_kind if node else "builtin",
                "handler": node.handler if node else stage,
                "action_results": list(state.action_results),
                "verification_notes": list(state.verification_notes),
            },
            "process": self._stage_event_messages(state, stage),
            "handoff": self._stage_handoff(stage),
        }

    def _stage_handoff(self, stage: str) -> dict[str, Any]:
        spec = self._workflow_spec()
        order = spec.node_ids()
        stage_agents = spec.stage_agents()
        index = order.index(stage) if stage in order else -1
        previous_stage = order[index - 1] if index > 0 else None
        next_stage = order[index + 1] if 0 <= index < len(order) - 1 else None
        return {
            "from": previous_stage,
            "to": stage,
            "next": next_stage,
            "from_agent": stage_agents.get(previous_stage or "", ""),
            "agent": stage_agents.get(stage, "coordinator"),
            "next_agent": stage_agents.get(next_stage or "", ""),
        }

    def _intake_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {"user_goal": state.user_goal},
            "actual_output": {
                "routing": {
                    "source": state.routing_source,
                    "difficulty": state.difficulty,
                    "workflow": state.workflow_variant,
                    "confidence": state.routing_confidence,
                    "risk_level": state.risk_level,
                    "evidence_mode": state.evidence_mode,
                    "evidence_strictness": state.evidence_strictness,
                    "evidence_category": state.evidence_category,
                    "evidence_sources": list(state.evidence_sources),
                    "evidence_reasons": list(state.evidence_reasons),
                },
                "coordinator_decision": dict(state.routing_decision),
                "reviewer_decision": dict(state.reviewer_decision),
            },
            "process": self._stage_event_messages(state, "intake"),
            "handoff": self._stage_handoff("intake"),
        }

    def _plan_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "normalized_goal": state.user_goal,
                "routing": dict(state.routing_decision),
                "reviewer": dict(state.reviewer_decision),
                "evidence": {
                    "mode": state.evidence_mode,
                    "strictness": state.evidence_strictness,
                    "sources": list(state.evidence_sources),
                    "reasons": list(state.evidence_reasons),
                },
            },
            "actual_output": {"plan_steps": list(state.plan)},
            "process": self._stage_event_messages(state, "plan"),
            "handoff": self._stage_handoff("plan"),
        }

    def _retrieve_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "query": state.user_goal,
                "plan_steps": list(state.plan),
                "evidence_sources": list(state.evidence_sources),
                "external_sources_attempted": list(state.external_search_attempted),
            },
            "actual_output": {
                "rag_results": [self._chunk_artifact(chunk) for chunk in state.retrieval_results],
                "external_results": [self._chunk_artifact(chunk) for chunk in state.external_results],
                "diagnostics": list(state.external_search_diagnostics),
                "evidence_ids": [item.id for item in state.evidence],
            },
            "process": self._stage_event_messages(state, "retrieve"),
            "handoff": self._stage_handoff("retrieve"),
        }

    def _reason_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "plan_steps": list(state.plan),
                "evidence": [self._evidence_artifact(item) for item in state.evidence],
                "qualified_evidence_ids": list(state.qualified_evidence_ids),
            },
            "actual_output": {"answer_draft": state.answer, "response_kind": state.response_kind},
            "process": self._stage_event_messages(state, "reason"),
            "handoff": self._stage_handoff("reason"),
        }

    def _evidence_audit_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "answer_draft": state.answer,
                "evidence": [self._evidence_artifact(item) for item in state.evidence],
            },
            "actual_output": {
                "evidence_status": state.evidence_status,
                "verification_notes": list(state.verification_notes),
                "qualified_evidence_ids": list(state.qualified_evidence_ids),
                "unqualified_evidence_ids": list(state.unqualified_evidence_ids),
            },
            "process": self._stage_event_messages(state, "evidence_audit"),
            "handoff": self._stage_handoff("evidence_audit"),
        }

    def _gate_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "risk_level": state.risk_level,
                "evidence_status": state.evidence_status,
                "qualified_evidence_ids": list(state.qualified_evidence_ids),
                "unqualified_evidence_ids": list(state.unqualified_evidence_ids),
            },
            "actual_output": {"gate_decisions": [decision.to_dict() for decision in state.gate_decisions]},
            "process": self._stage_event_messages(state, "gate"),
            "handoff": self._stage_handoff("gate"),
        }

    def _act_or_answer_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "gate_decisions": [decision.to_dict() for decision in state.gate_decisions],
                "answer_before_gate_action": state.answer,
            },
            "actual_output": {
                "answer_after_gate_action": state.answer,
                "action_results": list(state.action_results),
            },
            "process": self._stage_event_messages(state, "act_or_answer"),
            "handoff": self._stage_handoff("act_or_answer"),
        }

    def _verify_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "answer": state.answer,
                "gate_decisions": [decision.to_dict() for decision in state.gate_decisions],
                "evidence_status": state.evidence_status,
            },
            "actual_output": {"verification_notes": list(state.verification_notes)},
            "process": self._stage_event_messages(state, "verify"),
            "handoff": self._stage_handoff("verify"),
        }

    def _consolidate_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "qualified_evidence_ids": list(state.qualified_evidence_ids),
                "gate_decisions": [decision.to_dict() for decision in state.gate_decisions],
            },
            "actual_output": {
                "pending_consolidation": list(state.pending_consolidation),
                "evidence_consolidation_proposals": list(state.evidence_consolidation_proposals),
            },
            "process": self._stage_event_messages(state, "consolidate"),
            "handoff": self._stage_handoff("consolidate"),
        }

    def _respond_artifacts(self, state: AgentState) -> dict[str, Any]:
        return {
            "actual_input": {
                "answer_draft": state.answer,
                "pending_consolidation": list(state.pending_consolidation),
            },
            "actual_output": {"workflow_answer": state.answer},
            "process": self._stage_event_messages(state, "respond"),
            "handoff": self._stage_handoff("respond"),
        }

    def _stage_event_messages(self, state: AgentState, stage: str) -> list[dict[str, Any]]:
        return [
            {
                "time": event.get("time"),
                "agent": event.get("agent"),
                "kind": event.get("kind"),
                "message": event.get("message"),
                "duration_ms": event.get("duration_ms"),
            }
            for event in state.stage_events
            if event.get("stage") == stage
        ]

    def _chunk_artifact(self, chunk: Any) -> dict[str, Any]:
        return {
            "source": chunk.source,
            "span": chunk.span,
            "score": chunk.score,
            "content_hash": chunk.content_hash,
            "evidence_id": chunk.evidence_id,
            "text": _excerpt(chunk.text, limit=1400),
            "retrieval_method": getattr(chunk, "retrieval_method", "keyword"),
            "score_breakdown": dict(getattr(chunk, "score_breakdown", {}) or {}),
            "metadata": dict(getattr(chunk, "metadata", {}) or {}),
        }

    def _evidence_artifact(self, item: Any) -> dict[str, Any]:
        data = item.to_dict()
        if "summary" in data:
            data["summary"] = _excerpt(str(data["summary"]), limit=1400)
        return data

    def _events(self, state: AgentState) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = list(state.stage_events)
        for chunk in state.retrieval_results:
            events.append(
                {
                    "time": utc_now(),
                    "agent": "retriever",
                    "kind": "rag",
                    "message": f"{Path(chunk.source).name} {chunk.span} score={chunk.score:.2f}",
                }
            )
        for chunk in state.external_results:
            events.append(
                {
                    "time": utc_now(),
                    "agent": "retriever",
                    "kind": "external_evidence",
                    "message": f"{chunk.source} {chunk.span} score={chunk.score:.2f}",
                    "evidence_id": chunk.evidence_id,
                }
            )
        for decision in state.gate_decisions:
            status_label = {"allow": "允许", "interrupt": "中断", "deny": "拒绝"}.get(
                decision.status,
                decision.status,
            )
            risk_label = {
                "none": "无事实结论",
                "low": "低",
                "medium": "中",
                "high": "高",
                "critical": "严重",
            }.get(decision.risk_level, decision.risk_level)
            events.append(
                {
                    "time": utc_now(),
                    "agent": "critic",
                    "kind": "gate",
                    "message": f"{status_label} 风险={risk_label}",
                }
            )
        return events

    def _stage_durations(self, state: AgentState) -> dict[str, float]:
        return {
            str(event["stage"]): float(event["duration_ms"])
            for event in state.stage_events
            if event.get("kind") == "stage_completed" and "duration_ms" in event
        }

    def _last_event_for(self, name: str, state: AgentState) -> str:
        if name == "planner":
            return state.plan[-1] if state.plan else "暂无计划"
        if name == "retriever":
            return f"{len(state.retrieval_results)} 条本地 RAG，{len(state.external_results)} 条外部证据"
        if name == "reasoner":
            return "已生成答案草稿" if state.answer else "暂无答案"
        if name == "critic":
            return state.verification_notes[-1] if state.verification_notes else "门禁已检查"
        if name == "reviewer":
            return state.reviewer_status
        if name == "memory":
            return state.pending_consolidation[-1] if state.pending_consolidation else "暂无记忆提案"
        if name == "evolver":
            return "本次运行没有自进化提案"
        return state.current_stage

    def _call_coordinator_router(
        self,
        message: str,
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        client = self._deepseek_client(role="planner")
        messages = self._routing_messages(
            message,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
        )
        started = perf_counter()
        result, retry_events = self._chat_with_retry(
            client,
            messages,
            temperature=0.0,
            max_tokens=700,
            agent="coordinator",
            retry_kind="routing_retry",
        )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        parsed = _json_object_from_text(result.content)
        events = [
            *retry_events,
            {
                "time": utc_now(),
                "agent": "coordinator",
                "kind": "llm_routing",
                "duration_ms": duration_ms,
                "message": "DeepSeek coordinator completed autonomous routing.",
            }
        ]
        if parsed is None:
            route = _fallback_route_decision(
                message,
                source="rules_fallback_after_invalid_llm_json",
                reason="Coordinator did not return valid ROUTING_DECISION_JSON.",
            )
            events.append(
                {
                    "time": utc_now(),
                    "agent": "coordinator",
                    "kind": "routing_fallback",
                    "message": "Coordinator JSON parse failed; deterministic fallback route was used and marked.",
                }
            )
            return route, events
        return _normalize_route_decision(parsed, source="llm", fallback_message=message), events

    def _call_reviewer(
        self,
        message: str,
        coordinator_route: dict[str, Any],
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        client = self._deepseek_client(role="critic")
        messages = self._reviewer_messages(
            message,
            coordinator_route,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
        )
        started = perf_counter()
        result, retry_events = self._chat_with_retry(
            client,
            messages,
            temperature=0.0,
            max_tokens=700,
            agent="reviewer",
            retry_kind="reviewer_retry",
        )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        parsed = _json_object_from_text(result.content)
        events = [
            *retry_events,
            {
                "time": utc_now(),
                "agent": "reviewer",
                "kind": "route_review",
                "duration_ms": duration_ms,
                "message": "DeepSeek reviewer completed route review.",
            }
        ]
        if parsed is None:
            reviewer = {
                "review_status": "not_available",
                "findings": ["Reviewer did not return valid REVIEW_DECISION_JSON."],
            }
            final_route = dict(coordinator_route)
            final_route["reviewer"] = reviewer
            events.append(
                {
                    "time": utc_now(),
                    "agent": "reviewer",
                    "kind": "reviewer_fallback",
                    "message": "Reviewer JSON parse failed; coordinator route was kept and marked.",
                }
            )
            return final_route, reviewer, events
        reviewer = _normalize_reviewer_decision(parsed)
        final_route = _merge_reviewer_route(coordinator_route, reviewer, fallback_message=message)
        events.append(
            {
                "time": utc_now(),
                "agent": "reviewer",
                "kind": "review_decision",
                "message": f"review_status={reviewer.get('review_status', 'approve')}",
            }
        )
        return final_route, reviewer, events

    def _routing_messages(
        self,
        message: str,
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> list[ChatMessage]:
        system = (
            "你是前端 Coordinator，必须自主判断用户输入的任务难度、风险、证据需求和工作流。"
            "不要依赖固定关键词清单；根据语义、领域不确定性、事实可变性、用户后果、是否需要外部验证来判断。"
            "简单寒暄、身份介绍、纯创作、低风险解释可以 optional。"
            "技术、科学、事实性解释、影响因素、机制、比较、可靠判断通常至少是 medium，并应触发 required+soft 证据工作流。"
            "学术综述、论文依据、监管领域、生产/安全/写入/自进化、高风险决策或证据不足会造成误导的任务应为 hard 或 strict。"
            "不确定时宁可提高证据要求。只输出 JSON，不要输出解释文本。"
        )
        user = (
            "ROUTING_DECISION_JSON schema:\n"
            "{\n"
            '  "difficulty": "simple|medium|hard",\n'
            '  "workflow": "routine|evidence_soft|evidence_strict|protected_action",\n'
            '  "evidence_mode": "optional|required",\n'
            '  "evidence_strictness": "none|soft|strict",\n'
            '  "risk_level": "none|low|medium|high|critical",\n'
            '  "category": "routine|scientific_claim|technical_claim|academic|decision_analysis|regulated_domain|high_risk_action|current_factual|explicit_evidence_request|hard_reasoning",\n'
            '  "sources": ["rag","web","papers","user_experience"],\n'
            '  "reasons": ["short reason"],\n'
            '  "confidence": 0.0,\n'
            '  "retrieval_query": "optional rewritten retrieval query"\n'
            "}\n\n"
            f"用户输入:\n{message}\n\n"
            f"短期记忆 JSON:\n{json.dumps(short_term_memory, ensure_ascii=False)}\n\n"
            f"长期记忆 JSON:\n{json.dumps(long_term_memory, ensure_ascii=False)}\n"
        )
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _reviewer_messages(
        self,
        message: str,
        coordinator_route: dict[str, Any],
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> list[ChatMessage]:
        system = (
            "你是 Reviewer，负责审查 Coordinator 的路由是否过松、是否漏掉证据系统、是否选错工作流。"
            "你的职责是防止低估任务难度。不要按固定关键词清单判断；审查语义风险、科学/技术事实依赖、学术性、当前性、用户后果和证据缺口。"
            "如果 Coordinator 把需要证据的问题判成 optional，必须升级。"
            "如果任务是普通闲聊或身份问题，可以批准 optional。只输出 JSON。"
        )
        user = (
            "REVIEW_DECISION_JSON schema:\n"
            "{\n"
            '  "review_status": "approve|escalate|revise",\n'
            '  "difficulty": "simple|medium|hard",\n'
            '  "workflow": "routine|evidence_soft|evidence_strict|protected_action",\n'
            '  "evidence_mode": "optional|required",\n'
            '  "evidence_strictness": "none|soft|strict",\n'
            '  "risk_level": "none|low|medium|high|critical",\n'
            '  "category": "routine|scientific_claim|technical_claim|academic|decision_analysis|regulated_domain|high_risk_action|current_factual|explicit_evidence_request|hard_reasoning",\n'
            '  "sources": ["rag","web","papers","user_experience"],\n'
            '  "findings": ["review finding"],\n'
            '  "reasons": ["why the final route is acceptable"],\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            f"用户输入:\n{message}\n\n"
            f"Coordinator 路由 JSON:\n{json.dumps(coordinator_route, ensure_ascii=False)}\n\n"
            f"短期记忆 JSON:\n{json.dumps(short_term_memory, ensure_ascii=False)}\n\n"
            f"长期记忆 JSON:\n{json.dumps(long_term_memory, ensure_ascii=False)}\n"
        )
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _chat_with_retry(
        self,
        client: ChatClient,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        agent: str,
        retry_kind: str,
    ) -> tuple[Any, list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            attempt_started = perf_counter()
            try:
                return client.chat(messages, temperature=temperature, max_tokens=max_tokens), events
            except LLMRequestError as exc:
                duration_ms = round((perf_counter() - attempt_started) * 1000, 2)
                if attempt >= max_attempts:
                    raise
                events.append(
                    {
                        "time": utc_now(),
                        "agent": agent,
                        "kind": retry_kind,
                        "duration_ms": duration_ms,
                        "message": f"DeepSeek call failed, retrying once: {exc}",
                    }
                )
        raise RuntimeError("unreachable retry state")

    def _call_deepseek(
        self,
        message: str,
        state: AgentState,
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        llm_status = {
            "enabled": True,
            "provider": self.config.models.get("worker", {}).get("provider", "deepseek"),
            "model": self.config.models.get("worker", {}).get("model", "deepseek-v4-flash"),
            "status": "not_called",
        }
        client = self._deepseek_client()
        messages = self._deepseek_messages(
            message,
            state,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
        )
        events: list[dict[str, Any]] = []
        max_attempts = 2
        started = perf_counter()
        for attempt in range(1, max_attempts + 1):
            attempt_started = perf_counter()
            try:
                result = client.chat(messages, temperature=0.2, max_tokens=768)
                break
            except LLMRequestError as exc:
                duration_ms = round((perf_counter() - attempt_started) * 1000, 2)
                if attempt >= max_attempts:
                    raise
                events.append(
                    {
                        "time": utc_now(),
                        "agent": "reasoner",
                        "kind": "llm_retry",
                        "duration_ms": duration_ms,
                        "message": f"DeepSeek 调用失败，准备重试：{exc}",
                    }
                )
        duration_ms = round((perf_counter() - started) * 1000, 2)
        llm_status["status"] = "called"
        llm_status["model"] = result.model
        events.append(
            {
                "time": utc_now(),
                "agent": "reasoner",
                "kind": "llm_completed",
                "duration_ms": duration_ms,
                "message": "DeepSeek 已生成回答。",
            }
        )
        return (
            result.content.strip(),
            llm_status,
            events,
        )

    def _deepseek_client(self, *, role: str = "worker") -> ChatClient:
        if self.llm_client_factory is not None:
            return self.llm_client_factory(self.config)
        return DeepSeekChatClient.from_config(self.config, role=role)

    def _deepseek_messages(
        self,
        message: str,
        state: AgentState,
        *,
        short_term_memory: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> list[ChatMessage]:
        gate = state.gate_decisions[-1] if state.gate_decisions else None
        state_summary = {
            "evidence_mode": state.evidence_mode,
            "evidence_strictness": state.evidence_strictness,
            "evidence_status": state.evidence_status,
            "risk_level": state.risk_level,
            "evidence_category": state.evidence_category,
            "evidence_reasons": state.evidence_reasons,
            "evidence_sources": state.evidence_sources,
            "external_search_attempted": state.external_search_attempted,
            "gate_status": gate.status if gate else "unknown",
            "gate_reasons": gate.reasons if gate else [],
            "evidence_ids": [item.id for item in state.evidence],
            "qualified_evidence_ids": state.qualified_evidence_ids,
            "unqualified_evidence_ids": state.unqualified_evidence_ids,
            "external_search_diagnostics": state.external_search_diagnostics,
            "evidence_consolidation_proposals": state.evidence_consolidation_proposals,
            "stage_trace": state.stage_trace,
        }
        evidence_items = [
            {
                "id": item.id,
                "source_type": item.source_type,
                "uri": item.uri,
                "locator": item.locator,
                "summary": item.summary,
                "confidence": item.confidence,
            }
            for item in state.evidence
        ]
        rag_results = [
            {
                "source": chunk.source,
                "span": chunk.span,
                "score": chunk.score,
                "evidence_id": chunk.evidence_id,
                "retrieval_method": getattr(chunk, "retrieval_method", "keyword"),
                "score_breakdown": dict(getattr(chunk, "score_breakdown", {}) or {}),
                "text": chunk.text[:1200],
            }
            for chunk in state.retrieval_results
        ]
        external_evidence_results = [
            {
                "source": chunk.source,
                "span": chunk.span,
                "score": chunk.score,
                "evidence_id": chunk.evidence_id,
                "text": chunk.text[:1200],
            }
            for chunk in state.external_results
        ]
        policy_line = (
            f"evidence_mode={state.evidence_mode}; risk_level={state.risk_level}; "
            f"evidence_strictness={state.evidence_strictness}; evidence_status={state.evidence_status}; "
            f"evidence_category={state.evidence_category}; evidence_sources={state.evidence_sources}; "
            f"external_search_attempted={state.external_search_attempted}"
        )
        system = (
            "你是通过 DeepSeek API 调用的多 Agent worker。必须直接回答用户，不要输出预设模板话术。"
            "普通对话可以自然回答，不需要引用证据。"
            "记忆边界必须严格遵守：短期记忆只代表当前 thread 的最近对话；长期记忆只代表已批准的用户/项目/流程事实；"
            "知识库只代表文档、论文、API 文档、规范和数据资料，不能把知识库资料当成长期记忆。"
            "你必须读取提供的短期记忆和长期记忆；当用户追问刚才说过什么、偏好、代号、项目设定时，优先依据记忆回答。"
            "证据策略分三层：optional 表示普通对话不需要证据；required+soft 表示中等问题必须先检索，"
            "证据不足或 evidence_status=exhausted/partial 时可以给受限回答，但必须明确证据不足，不能伪造引用或确定性结论；"
            "required+strict 表示困难、高危、学术综述、可靠决策或受监管问题，证据不足时必须打回。"
            "当 evidence_items、rag_results 或 external_evidence_results 存在时，优先依据它们回答；"
            "不要在最终回答正文中输出 evidence id、证据编号或来源详情，参考文献将由调试界面的证据栏展示。"
            "当 evidence_strictness=soft 且 gate_status=allow 但证据不足时，可以输出受限回答，说明已尝试检索但证据不足。"
            "当 evidence_strictness=strict 时，只能依据提供的证据回答；证据不足或 gate_status 不是 allow 时不能给底层结论。"
            "如果证据不足，要明确说明限制并建议继续搜索论文、网络资料、RAG 文档或用户经验证据。"
            "如果 gate_status 不是 allow，不能执行或给出底层高风险方案，只能说明门禁未通过以及需要哪些证据或审批。"
            "不要编造证据 id，不要声称读取了未提供的资料。默认使用中文，除非用户明显使用其他语言。"
        )
        user = (
            "证据策略：\n"
            f"{policy_line}\n\n"
            "用户输入：\n"
            f"{message}\n\n"
            "运行状态 JSON：\n"
            f"{state_summary}\n\n"
            "记忆边界 JSON：\n"
            f"{MEMORY_BOUNDARIES}\n\n"
            "短期记忆 JSON：\n"
            f"{short_term_memory}\n\n"
            "长期记忆 JSON：\n"
            f"{long_term_memory}\n\n"
            "证据项 JSON：\n"
            f"{evidence_items}\n\n"
            "RAG 结果 JSON：\n"
            f"{rag_results}\n\n"
            "external_evidence_results JSON：\n"
            f"{external_evidence_results}\n\n"
            "请给出 DeepSeek 实时生成的最终回答。"
        )
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _short_term_memory_for(self, thread_id: str) -> ShortTermConversationMemory:
        if thread_id not in self.short_term_memories:
            memory = ShortTermConversationMemory(
                max_turns=int(self.config.memory.get("short_term_turns", 8))
            )
            if self.session_store is not None:
                try:
                    snapshot = self.session_store.load(thread_id)
                except FileNotFoundError:
                    snapshot = None
                if snapshot is not None:
                    turns: list[dict[str, Any]] = []
                    current: dict[str, Any] = {}
                    for message in snapshot.messages:
                        role = str(message.get("role", ""))
                        if role == "user":
                            if current:
                                turns.append(current)
                            current = {"user": message.get("content", ""), "run_id": message.get("run_id", "restored")}
                        elif role == "assistant" and current:
                            current["assistant"] = message.get("content", "")
                            turns.append(current)
                            current = {}
                    if current:
                        turns.append(current)
                    memory.restore(turns)
            self.short_term_memories[thread_id] = memory
        return self.short_term_memories[thread_id]

    def _memory_root(self) -> Path:
        root = Path(self.config.memory.get("directory", "memory"))
        if not root.is_absolute():
            root = self.workspace_root / root
        return root

    def _memory_store(self) -> LongTermMemoryStore:
        return LongTermMemoryStore(
            self._memory_root(),
            gate_policy=GatePolicy(
                workspace_root=self.workspace_root,
                min_evidence_by_risk=dict(self.config.gates.get("min_evidence_by_risk", {})),
                approval_required_actions=set(self.config.gates.get("approval_required_actions", [])),
            ),
        )

    def _load_long_term_memories(self, *, limit: int = 12) -> list[dict[str, Any]]:
        store = self._memory_store()
        records: list[dict[str, Any]] = []
        for partition in self.config.memory.get("partitions", ["user", "project"]):
            for record in store.list_partition(str(partition)):
                records.append(record)
        return records[-limit:]

    def _write_explicit_long_term_memory(self, message: str) -> list[Any]:
        candidate = explicit_memory_candidate(message)
        if candidate is None:
            return []

        ledger = EvidenceLedger(self.workspace_root / "evidence" / "ledger.jsonl")
        evidence = ledger.record(
            source_type="user_memory_request",
            uri="user://current-message",
            locator="prompt",
            content=message,
            summary=f"用户明确要求长期记住：{candidate['value']}",
            confidence=1.0,
            used_for=["memory:write"],
        )
        if candidate.get("boundary") == "knowledge_base":
            return [
                deny_knowledge_memory_write(
                    key=candidate["key"],
                    value=candidate["value"],
                    evidence=[evidence],
                )
            ]
        result = self._memory_store().write(
            partition=candidate["partition"],
            key=candidate["key"],
            value=candidate["value"],
            evidence=[evidence],
            approved_by="user_explicit_request",
        )
        return [result]


ROUTE_KEYS = {
    "difficulty",
    "workflow",
    "evidence_mode",
    "evidence_strictness",
    "risk_level",
    "category",
    "sources",
    "reasons",
    "confidence",
    "retrieval_query",
}
SOURCE_VALUES = {"rag", "web", "papers", "user_experience"}
DIFFICULTY_ORDER = {"simple": 0, "medium": 1, "hard": 2}
STRICTNESS_ORDER = {"none": 0, "soft": 1, "strict": 2}
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _json_object_from_text(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_route_decision(message: str, *, source: str, reason: str) -> dict[str, Any]:
    requirement = classify_evidence_requirement(message)
    strictness = _strictness_for_route(requirement.category, requirement.risk_level)
    difficulty = "simple" if requirement.mode != "required" else ("hard" if strictness == "strict" else "medium")
    return {
        "difficulty": difficulty,
        "workflow": _workflow_for_route(requirement.mode, requirement.risk_level, requirement.category, strictness),
        "evidence_mode": requirement.mode,
        "evidence_strictness": strictness,
        "risk_level": requirement.risk_level,
        "category": requirement.category,
        "sources": list(requirement.sources),
        "reasons": [*requirement.reasons, reason],
        "confidence": 0.25,
        "source": source,
    }


def _normalize_route_decision(
    route: dict[str, Any],
    *,
    source: str,
    fallback_message: str,
) -> dict[str, Any]:
    fallback = _fallback_route_decision(
        fallback_message,
        source=source,
        reason="Fallback fields filled from deterministic safety policy.",
    )
    mode = _one_of(route.get("evidence_mode"), {"optional", "required"}, fallback["evidence_mode"])
    risk_level = _one_of(route.get("risk_level"), set(RISK_ORDER), fallback["risk_level"])
    category = str(route.get("category") or fallback["category"]).strip() or fallback["category"]
    strictness = _one_of(
        route.get("evidence_strictness"),
        set(STRICTNESS_ORDER),
        _strictness_for_route(category, risk_level) if mode == "required" else "none",
    )
    difficulty = _one_of(route.get("difficulty"), set(DIFFICULTY_ORDER), fallback["difficulty"])
    sources = [item for item in _list_of_strings(route.get("sources")) if item in SOURCE_VALUES]
    if mode == "required" and not sources:
        sources = ["rag", "web", "papers", "user_experience"]
    if mode == "optional":
        sources = []
    reasons = _list_of_strings(route.get("reasons")) or list(fallback["reasons"])
    workflow = _one_of(
        route.get("workflow"),
        {"routine", "evidence_soft", "evidence_strict", "protected_action"},
        _workflow_for_route(mode, risk_level, category, strictness),
    )
    normalized = {
        "difficulty": difficulty,
        "workflow": workflow,
        "evidence_mode": mode,
        "evidence_strictness": strictness,
        "risk_level": risk_level,
        "category": category,
        "sources": sources,
        "reasons": reasons,
        "confidence": _float_between(route.get("confidence"), default=float(fallback["confidence"])),
        "source": source,
    }
    retrieval_query = str(route.get("retrieval_query") or "").strip()
    if retrieval_query:
        normalized["retrieval_query"] = retrieval_query
    return normalized


def _normalize_reviewer_decision(decision: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "review_status": _one_of(decision.get("review_status"), {"approve", "escalate", "revise"}, "approve"),
        "findings": _list_of_strings(decision.get("findings")),
    }
    for key in ROUTE_KEYS:
        if key not in decision:
            continue
        if key in {"sources", "reasons"}:
            values = _list_of_strings(decision.get(key))
            if key == "sources":
                values = [item for item in values if item in SOURCE_VALUES]
            if values:
                normalized[key] = values
        elif key == "confidence":
            normalized[key] = _float_between(decision.get(key), default=0.0)
        else:
            value = str(decision.get(key) or "").strip()
            if value:
                normalized[key] = value
    return normalized


def _merge_reviewer_route(
    coordinator_route: dict[str, Any],
    reviewer: dict[str, Any],
    *,
    fallback_message: str,
) -> dict[str, Any]:
    status = str(reviewer.get("review_status", "approve"))
    reviewer_stricter = _reviewer_is_stricter(coordinator_route, reviewer)
    should_apply = reviewer_stricter
    if should_apply:
        merged = dict(coordinator_route)
        if RISK_ORDER.get(str(reviewer.get("risk_level", "none")), 0) > RISK_ORDER.get(
            str(merged.get("risk_level", "none")), 0
        ):
            merged["risk_level"] = reviewer["risk_level"]
        if DIFFICULTY_ORDER.get(str(reviewer.get("difficulty", "simple")), 0) > DIFFICULTY_ORDER.get(
            str(merged.get("difficulty", "simple")), 0
        ):
            merged["difficulty"] = reviewer["difficulty"]
        if reviewer.get("evidence_mode") == "required":
            merged["evidence_mode"] = "required"
        if STRICTNESS_ORDER.get(str(reviewer.get("evidence_strictness", "none")), 0) > STRICTNESS_ORDER.get(
            str(merged.get("evidence_strictness", "none")), 0
        ):
            merged["evidence_strictness"] = reviewer["evidence_strictness"]
        if reviewer.get("category") and merged.get("category") in {None, "routine"}:
            merged["category"] = reviewer["category"]
        merged["sources"] = _dedupe_strings(
            [*_list_of_strings(merged.get("sources")), *_list_of_strings(reviewer.get("sources"))]
        )
        merged["reasons"] = _dedupe_strings(
            [*_list_of_strings(merged.get("reasons")), *_list_of_strings(reviewer.get("reasons")), *_list_of_strings(reviewer.get("findings"))]
        )
        merged["workflow"] = _workflow_for_route(
            str(merged.get("evidence_mode", "optional")),
            str(merged.get("risk_level", "none")),
            str(merged.get("category", "routine")),
            str(merged.get("evidence_strictness", "none")),
        )
        source = "llm+reviewer" if str(coordinator_route.get("source")) == "llm" else f"{coordinator_route.get('source', 'route')}+reviewer"
        final = _normalize_route_decision(merged, source=source, fallback_message=fallback_message)
    else:
        final = dict(coordinator_route)
        if status in {"escalate", "revise"}:
            final["reasons"] = _dedupe_strings(
                [*_list_of_strings(coordinator_route.get("reasons")), *_list_of_strings(reviewer.get("findings"))]
            )
            final["reviewer_route_ignored"] = "weaker_than_coordinator"
    final["reviewer"] = reviewer
    return final


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _reviewer_is_stricter(coordinator_route: dict[str, Any], reviewer: dict[str, Any]) -> bool:
    if reviewer.get("evidence_mode") == "required" and coordinator_route.get("evidence_mode") != "required":
        return True
    if STRICTNESS_ORDER.get(str(reviewer.get("evidence_strictness", "none")), 0) > STRICTNESS_ORDER.get(
        str(coordinator_route.get("evidence_strictness", "none")),
        0,
    ):
        return True
    if RISK_ORDER.get(str(reviewer.get("risk_level", "none")), 0) > RISK_ORDER.get(
        str(coordinator_route.get("risk_level", "none")),
        0,
    ):
        return True
    return DIFFICULTY_ORDER.get(str(reviewer.get("difficulty", "simple")), 0) > DIFFICULTY_ORDER.get(
        str(coordinator_route.get("difficulty", "simple")),
        0,
    )


def _evidence_references(evidence: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "id": item.id,
            "source_type": item.source_type,
            "uri": item.uri,
            "locator": item.locator,
            "summary": item.summary,
            "confidence": item.confidence,
            "content_hash": item.content_hash,
        }
        for index, item in enumerate(evidence, start=1)
    ]


def _excerpt(value: str, *, limit: int = 1400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _sanitize_answer_for_user(answer: str) -> str:
    original = answer.strip()
    text = re.sub(
        r"\s*(?:证据|引用|参考文献|参考|Evidence|Source|Reference)\s*[:：]\s*"
        r"(?:\[[^\]]*\bev_[A-Za-z0-9_-]+\b[^\]]*\]|\bev_[A-Za-z0-9_-]+\b)\s*[。.;；,，]?",
        "",
        original,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[[^\]]*\bev_[A-Za-z0-9_-]+\b[^\]]*\]", "", text)
    text = re.sub(r"\(\s*\bev_[A-Za-z0-9_-]+\b\s*\)", "", text)
    text = re.sub(r"\bev_[A-Za-z0-9_-]+\b", "", text)
    text = re.sub(r"[ \t]+([，。！？；：,.!?;:])", r"\1", text)
    text = re.sub(r"([。！？；,.!?;]){2,}", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or original


def _event_hint(event: dict[str, Any]) -> str:
    stage = str(event.get("stage") or "workflow")
    kind = str(event.get("kind") or "stage_started")
    agent = str(event.get("agent") or "agent")
    if kind == "stage_completed":
        return f"{agent} 已完成 {stage} 节点。"
    if kind == "external_search_attempted":
        return "Retriever 正在检索外部证据来源。"
    return f"{agent} 正在执行 {stage} 节点。"


def _strictness_for_route(category: str, risk_level: str) -> str:
    if risk_level in {"high", "critical"}:
        return "strict"
    if category in {
        "academic",
        "regulated_advice",
        "regulated_domain",
        "high_risk_action",
        "hard_reasoning",
        "decision_analysis",
    }:
        return "strict"
    if risk_level == "medium":
        return "soft"
    return "none"


def _workflow_for_route(mode: str, risk_level: str, category: str, strictness: str) -> str:
    if category == "high_risk_action":
        return "protected_action"
    if mode != "required":
        return "routine"
    if strictness == "strict" or risk_level in {"high", "critical"}:
        return "evidence_strict"
    return "evidence_soft"


def _one_of(value: object, allowed: set[str], default: str) -> str:
    normalized = str(value or "").lower().strip()
    return normalized if normalized in allowed else default


def _list_of_strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _float_between(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


MEMORY_BOUNDARIES = {
    "short_term": "current thread conversation only; rolling window; never persisted as knowledge",
    "long_term": "explicit user/project/procedural facts only; JSONL partitions under memory/; write_memory gate required",
    "knowledge_base": "documents, papers, API docs, specifications, datasets, and source material under knowledge/; retrieved through RAG",
}


def _normalize_thread_id(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value).strip())
    return normalized[:80] or "default"


def _contextual_goal_for_evidence_followup(message: str, short_term_memory: list[dict[str, Any]]) -> str:
    if not short_term_memory or not is_explicit_evidence_request(message):
        return message
    context_lines: list[str] = []
    for item in short_term_memory[-3:]:
        user_text = str(item.get("user", "")).strip()
        if user_text:
            context_lines.append(f"previous_user: {user_text}")
    if not context_lines:
        return message
    return f"{message}\n\nContext for evidence retrieval:\n" + "\n".join(context_lines)
