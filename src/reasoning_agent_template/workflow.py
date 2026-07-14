from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.evidence import EvidenceConsolidationEngine, EvidenceLedger
from reasoning_agent_template.external_evidence import ExternalEvidenceSearch
from reasoning_agent_template.gates import GatePolicy
from reasoning_agent_template.knowledge import LocalKnowledgeBase
from reasoning_agent_template.models import (
    AgentState,
    EvidenceRequirement,
    GateDecision,
    KnowledgeChunk,
    WorkflowResult,
    stable_hash,
    utc_now,
)
from reasoning_agent_template.risk import classify_evidence_requirement
from reasoning_agent_template.workflow_spec import (
    BUILTIN_STAGE_HANDLERS,
    WorkflowNodeSpec,
    load_workflow_spec,
)


STAGES = [
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
]

STAGE_AGENTS = {
    "intake": "coordinator",
    "plan": "planner",
    "retrieve": "retriever",
    "reason": "reasoner",
    "evidence_audit": "critic",
    "gate": "critic",
    "act_or_answer": "coordinator",
    "verify": "critic",
    "consolidate": "memory",
    "respond": "coordinator",
}

STRICT_EVIDENCE_CATEGORIES = {
    "academic",
    "regulated_advice",
    "regulated_domain",
    "high_risk_action",
    "hard_reasoning",
    "decision_analysis",
}

QUALIFIED_EXTERNAL_SOURCE_TYPES = {"paper", "web", "user_experience"}
STRICT_CURRENT_FACTUAL_TERMS = {"来源", "依据", "引用", "source", "sources", "cite"}
PROTECTED_ACTION_TERMS = {
    "删库",
    "删除数据",
    "绕过审批",
    "绕过批准",
    "跳过审批",
    "生产环境删",
    "delete database",
    "drop database",
    "bypass approval",
    "skip approval",
    "without approval",
    "绕过权限",
}


class TemplateCoordinator:
    """Deterministic implementation of the template's required state path."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        workspace_root: Path,
        event_callback: Callable[[AgentState, dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.workspace_root = Path(workspace_root)
        self.event_callback = event_callback
        self.ledger = EvidenceLedger(self.workspace_root / "evidence" / "ledger.jsonl")
        self.gate_policy = GatePolicy(
            workspace_root=self.workspace_root,
            min_evidence_by_risk=dict(config.gates.get("min_evidence_by_risk", {})),
            approval_required_actions=set(config.gates.get("approval_required_actions", [])),
        )
        self.workflow_spec = load_workflow_spec(self.workspace_root, self.config.runtime)
        validation = self.workflow_spec.validate(known_builtin_handlers=BUILTIN_STAGE_HANDLERS)
        if validation.errors or validation.requires_code:
            details = [*validation.errors, *validation.requires_code]
            raise ValueError("workflow spec is not runnable: " + "; ".join(details))

    def run(self, user_goal: str, *, routing_decision: dict | None = None) -> WorkflowResult:
        state = AgentState(user_goal=user_goal)
        if routing_decision:
            state.routing_decision = dict(routing_decision)
        for node in self.workflow_spec.execution_nodes():
            stage = node.id
            started = perf_counter()
            state.current_stage = stage
            state.stage_trace.append(stage)
            state.stage_events.append(
                {
                    "time": utc_now(),
                    "agent": node.agent,
                    "stage": stage,
                    "kind": "stage_started",
                    "message": f"{stage} started",
                }
            )
            self._emit_progress(state, state.stage_events[-1])
            self._run_node(node, state)
            state.stage_events.append(
                {
                    "time": utc_now(),
                    "agent": node.agent,
                    "stage": stage,
                    "kind": "stage_completed",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "message": f"{stage} completed",
                }
            )
            self._emit_progress(state, state.stage_events[-1])
        return WorkflowResult(
            answer=state.answer,
            state=state,
            stage_trace=list(state.stage_trace),
            evidence=list(state.evidence),
            gate_decisions=list(state.gate_decisions),
        )

    def _emit_progress(self, state: AgentState, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(state, event)

    def _run_node(self, node: WorkflowNodeSpec, state: AgentState) -> None:
        if node.handler_kind == "plugin_tool":
            self._plugin_tool_node(node, state)
            return
        handler = getattr(self, f"_{node.handler}", None)
        if handler is None:
            raise ValueError(f"workflow node {node.id} uses unknown builtin handler: {node.handler}")
        handler(state)

    def _plugin_tool_node(self, node: WorkflowNodeSpec, state: AgentState) -> None:
        state.action_results.append(f"plugin_tool node {node.id} requested handler {node.handler}")
        state.verification_notes.append(f"plugin_tool node {node.id} has no local executor")

    def _intake(self, state: AgentState) -> None:
        if state.routing_decision:
            requirement = _requirement_from_routing(state.routing_decision, state.user_goal)
            state.routing_source = str(state.routing_decision.get("source", "llm"))
            state.routing_confidence = float(state.routing_decision.get("confidence", 0.0) or 0.0)
            state.difficulty = _normalized_difficulty(str(state.routing_decision.get("difficulty", "simple")))
            state.workflow_variant = _normalized_workflow_variant(
                str(state.routing_decision.get("workflow", "")),
                requirement=requirement,
                strictness=str(state.routing_decision.get("evidence_strictness", "") or ""),
            )
            state.reviewer_decision = dict(state.routing_decision.get("reviewer", {}) or {})
            state.reviewer_status = str(state.reviewer_decision.get("review_status", "not_run"))
        else:
            requirement = classify_evidence_requirement(state.user_goal)
            state.routing_source = "rules"
            state.routing_confidence = 0.0
            state.difficulty = "simple" if requirement.mode != "required" else "medium"
            state.workflow_variant = _normalized_workflow_variant("", requirement=requirement, strictness="")
        state.risk_level = requirement.risk_level
        state.evidence_mode = requirement.mode
        state.evidence_strictness = _normalized_strictness(
            str(state.routing_decision.get("evidence_strictness", "") if state.routing_decision else "")
        ) or _evidence_strictness(
            category=requirement.category,
            risk_level=requirement.risk_level,
            user_goal=state.user_goal,
        )
        state.workflow_variant = _normalized_workflow_variant(
            state.workflow_variant,
            requirement=requirement,
            strictness=state.evidence_strictness,
        )
        state.evidence_status = "pending" if requirement.mode == "required" else "not_required"
        state.evidence_category = requirement.category
        state.evidence_reasons = list(requirement.reasons)
        state.evidence_sources = list(requirement.sources)
        if requirement.mode == "required":
            state.response_kind = "evidence_required_answer"
        else:
            state.response_kind = "routine"
        state.plan = []

    def _plan(self, state: AgentState) -> None:
        if state.evidence_mode == "required":
            if state.evidence_strictness == "soft":
                state.plan = [
                    "识别为中等风险事实/技术问题。",
                    "必须先检索本地 RAG、外部网络、论文或用户经验。",
                    "证据不足时只允许受限回答，不伪造引用。",
                    "记录证据缺口和检索尝试。",
                    "不写入长期记忆或技能。",
                ]
                return
            state.plan = [
                "识别高风险或高难强推理目标。",
                "检索带可引用位置的本地知识。",
                "把关键结论绑定到证据。",
                "回答前通过严格证据门禁。",
                "只记录记忆或技能更新提案。",
            ]
            return
        state.plan = [
            "识别为普通对话或低风险说明。",
            "跳过强制证据系统。",
            "直接生成可理解的中文回复。",
            "保留状态机和门禁遥测。",
            "不写入长期记忆或技能。",
        ]

    def _passthrough(self, state: AgentState) -> None:
        state.action_results.append(f"{state.current_stage} passthrough completed")

    @staticmethod
    def _split_claims(draft: str) -> list[str]:
        claims: list[str] = []
        start = 0
        hard_boundaries = set("\u3002\uff01\uff1f\uff1b;!?\r\n")

        def should_split_period(index: int) -> bool:
            if (
                index + 1 < len(draft)
                and not draft[index + 1].isspace()
            ):
                return False

            segment = draft[start:index + 1].strip()
            lowered = segment.lower()

            if lowered.endswith(("e.g.", "i.e.")):
                return False

            initialism = re.search(
                r"(?:[A-Za-z]\.){2,}$",
                segment,
            )

            if initialism:
                remainder = draft[index + 1:].lstrip()

                if not remainder:
                    return True

                if remainder[0].isupper():
                    prefix = (
                        segment[:initialism.start()]
                        .strip()
                        .lower()
                    )

                    return prefix not in {
                        "",
                        "the",
                        "a",
                        "an",
                    }

                return False

            return True

        for index, char in enumerate(draft):
            split_here = char in hard_boundaries

            if char == ".":
                split_here = should_split_period(index)

            if split_here:
                claim = draft[start:index].strip()

                if claim:
                    claims.append(claim)

                start = index + 1

        final_claim = draft[start:].strip()

        if final_claim:
            claims.append(final_claim)

        return claims

    def _claim_check(self, state: AgentState) -> None:
        """检查回答草稿中的结论与候选证据缺口。"""

        draft = str(state.answer or "").strip()
        claims = self._split_claims(draft)

        candidate_chunks = [
            *state.retrieval_results,
            *state.external_results,
        ]

        def text_terms(value: str) -> set[str]:
            lowered = value.lower()

            latin_terms = re.findall(
                r"[a-z0-9_]{2,}",
                lowered,
            )

            chinese_runs = re.findall(
                r"[\u4e00-\u9fff]+",
                lowered,
            )

            chinese_terms = [
                run[index:index + 2]
                for run in chinese_runs
                for index in range(len(run) - 1)
            ]

            return set(
                latin_terms + chinese_terms
            )

        def claim_matches_chunk(
            claim: str,
            chunk_text: str,
        ) -> bool:
            normalized_claim = re.sub(
                r"[\W_]+",
                "",
                claim.lower(),
            )
            normalized_chunk = re.sub(
                r"[\W_]+",
                "",
                chunk_text.lower(),
            )

            if (
                normalized_claim
                and normalized_claim in normalized_chunk
            ):
                return True

            claim_terms = text_terms(claim)
            chunk_terms = text_terms(chunk_text)

            if not claim_terms:
                return False

            overlap = (
                claim_terms
                & chunk_terms
            )

            coverage = (
                len(overlap)
                / len(claim_terms)
            )

            return coverage >= 0.5

        claim_evidence_bindings: dict[
            str,
            list[str],
        ] = {}

        for claim in claims:
            matched_ids = [
                chunk.evidence_id
                for chunk in candidate_chunks
                if (
                    chunk.evidence_id
                    and claim_matches_chunk(
                        claim,
                        chunk.text,
                    )
                )
            ]

            claim_evidence_bindings[claim] = list(
                dict.fromkeys(matched_ids)
            )

        claims_with_candidate_evidence = [
            claim
            for claim in claims
            if claim_evidence_bindings[claim]
        ]

        unsupported_claims = [
            claim
            for claim in claims
            if (
                state.evidence_mode == "required"
                and not claim_evidence_bindings[claim]
            )
        ]

        candidate_evidence_ids = list(
            dict.fromkeys(
                [
                    chunk.evidence_id
                    for chunk in candidate_chunks
                    if chunk.evidence_id
                ]
                + [
                    item.id
                    for item in state.evidence
                ]
            )
        )

        required_follow_up: list[str] = []

        if unsupported_claims:
            required_follow_up.append(
                "重新检索与关键结论直接相关的证据，"
                "并在 evidence_audit 中核验。"
            )
        elif candidate_chunks:
            required_follow_up.append(
                "在 evidence_audit 中核验候选证据的"
                "相关性、来源质量和门禁资格。"
            )

        state.claim_review = {
            "answer_draft": draft,
            "claims": claims,
            "candidate_evidence_ids": (
                candidate_evidence_ids
            ),
            "claim_evidence_bindings": (
                claim_evidence_bindings
            ),
            "claims_with_candidate_evidence": (
                claims_with_candidate_evidence
            ),
            "unsupported_claims": (
                unsupported_claims
            ),
            "required_follow_up": (
                required_follow_up
            ),
        }

        state.verification_notes.append(
            "claim_check 已检查 "
            f"{len(claims)} 条结论，"
            f"发现 {len(unsupported_claims)} 条"
            "尚无候选证据支持的结论"
        )


    def _review_note(self, state: AgentState) -> None:
        state.verification_notes.append(f"{state.current_stage} review completed")

    def _retrieve(self, state: AgentState) -> None:
        if state.evidence_mode != "required":
            state.retrieval_results = []
            state.external_results = []
            state.evidence = []
            return
        knowledge_dir = Path(self.config.knowledge.get("directory", "knowledge"))
        if not knowledge_dir.is_absolute():
            knowledge_dir = self.workspace_root / knowledge_dir
        kb = LocalKnowledgeBase(knowledge_dir, ledger=self.ledger)
        top_k = int(self.config.knowledge.get("top_k", 5))
        primary_methods = _knowledge_methods(self.config.knowledge)
        fallback_methods = _knowledge_fallback_methods(self.config.knowledge)
        min_score = float(self.config.knowledge.get("min_score", 0.0))
        state.rag_methods = list(primary_methods)
        state.retrieval_results = kb.retrieve(
            state.user_goal,
            top_k=top_k,
            methods=primary_methods,
            min_score=min_score,
            wiki_top_k=int(self.config.knowledge.get("wiki_top_k", 2)),
        )
        state.rag_diagnostics = list(kb.diagnostics)
        fallback_threshold = float(self.config.knowledge.get("fallback_min_score", 0.35))
        best_local_score = max((chunk.score for chunk in state.retrieval_results), default=0.0)
        if fallback_methods and best_local_score < fallback_threshold:
            fallback_results = kb.retrieve(
                state.user_goal,
                top_k=max(1, int(self.config.knowledge.get("fallback_top_k", 2))),
                methods=fallback_methods,
                min_score=0.0,
                wiki_top_k=int(self.config.knowledge.get("wiki_top_k", 2)),
            )
            state.rag_methods = [*state.rag_methods, *[method for method in fallback_methods if method not in state.rag_methods]]
            state.rag_diagnostics.extend(kb.diagnostics)
            state.retrieval_results = _dedupe_knowledge_chunks([*state.retrieval_results, *fallback_results])[:top_k]
        state.external_results = []
        external_sources = [source for source in state.evidence_sources if source in {"papers", "web", "user_experience"}]
        state.external_search_attempted = list(external_sources)
        if external_sources:
            state.stage_events.append(
                {
                    "time": utc_now(),
                    "agent": "retriever",
                    "stage": "retrieve",
                    "kind": "external_search_attempted",
                    "message": f"attempted external sources: {', '.join(external_sources)}",
                }
            )
            searcher = ExternalEvidenceSearch(
                ledger=self.ledger,
                timeout_seconds=int(self.config.knowledge.get("external_timeout_seconds", 8)),
            )
            state.external_results = searcher.retrieve(
                state.user_goal,
                top_k=int(self.config.knowledge.get("external_top_k", top_k)),
                sources=external_sources,
            )
            state.external_search_diagnostics = list(searcher.diagnostics)
        by_id = {item.id: item for item in self.ledger.list()}
        ordered_ids = [
            result.evidence_id
            for result in [*state.retrieval_results, *state.external_results]
            if result.evidence_id in by_id
        ]
        state.evidence = [by_id[evidence_id] for evidence_id in dict.fromkeys(ordered_ids)]

    def _config_identity_chunk(self) -> KnowledgeChunk:
        content = json.dumps(
            {
                "identity": self.config.identity,
                "models": self.config.models,
                "skills": self.config.skills,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        evidence = self.ledger.record(
            source_type="config",
            uri="agent.yaml",
            locator="identity, models, skills",
            content=content,
            summary=(
                f"来自 agent.yaml 的 {self.config.identity.get('name', 'reasoning-agent-template')} "
                "身份和已启用能力。"
            ),
            confidence=1.0,
            used_for=["identity:answer"],
        )
        return KnowledgeChunk(
            source="agent.yaml",
            span="identity, models, skills",
            text=content,
            content_hash=stable_hash(content),
            score=1.0,
            evidence_id=evidence.id,
        )

    def _reason(self, state: AgentState) -> None:
        if _is_identity_question(state.user_goal):
            name = self.config.identity.get("name", "reasoning-agent-template")
            purpose = _localized_purpose(
                self.config.identity.get("purpose", "Build evidence-first heavy-reasoning agents."),
                chinese=_has_cjk(state.user_goal),
            )
            if _has_cjk(state.user_goal):
                state.answer = (
                    f"我是 {name}，一个面向{purpose}的多 Agent 调试模板。"
                    "普通对话不会强制走证据系统；高风险或高难强推理任务才会启动证据、RAG 和高风险门禁。"
                )
            else:
                state.answer = (
                    f"I am {name}, a multi-agent debug template for {purpose}. "
                    "Routine chat does not require the evidence system; high-risk or hard reasoning tasks do."
                )
            return
        if _is_template_question(state.user_goal) and state.evidence_mode != "required":
            if _has_cjk(state.user_goal):
                state.answer = (
                    "这个模板通过工作流状态机串联 intake、plan、retrieve、reason、evidence_audit、gate、"
                    "act_or_answer、verify、consolidate 和 respond。证据系统不是每轮对话都强制启用；"
                    "只有高风险、高难强推理、关键修改、长期记忆、自进化、外部执行等任务才进入强制证据门禁。"
                )
            else:
                state.answer = (
                    "The template runs intake, plan, retrieve, reason, evidence_audit, gate, act_or_answer, "
                    "verify, consolidate, and respond. Evidence is mandatory only for high-risk or hard reasoning tasks."
                )
            return
        if not state.retrieval_results and state.evidence_mode != "required":
            state.answer = (
                "可以聊。当前这类普通对话不会强制启动证据系统；如果你提出高风险、高难强推理、"
                "关键修改、长期记忆或自进化任务，我会再切换到证据和门禁模式。"
                if _has_cjk(state.user_goal)
                else "We can chat. Routine conversation does not force evidence mode; high-risk or hard reasoning tasks will."
            )
            return
        evidence_results = [*state.retrieval_results, *state.external_results]
        if not evidence_results:
            if state.evidence_strictness == "soft":
                state.answer = (
                    "已触发证据检索，但未检索到足够证据；下面只能作为受限回答，不能当作可靠结论，"
                    "也不会引用不存在的证据。"
                    if _has_cjk(state.user_goal)
                    else "Evidence retrieval ran but did not find enough support; any answer must be limited and uncited."
                )
                return
            state.answer = (
                "这是高风险或高难强推理任务，但当前没有检索到足够证据，所以不能直接给结论。"
                "请补充知识库资料、明确证据来源，或降低任务风险后再继续。"
                if _has_cjk(state.user_goal)
                else "This is a high-risk or hard reasoning task, but I do not have enough evidence to answer it."
            )
            return
        best = evidence_results[0]
        if _has_cjk(state.user_goal):
            state.answer = (
                "这是高风险或高难强推理任务，已启用证据系统。当前回答只基于已检索证据生成；"
                "参考文献和证据详情请查看调试界面的证据栏。"
            )
        else:
            state.answer = (
                "Evidence mode is required for this high-risk or hard reasoning task. "
                "References and evidence details are available in the evidence panel."
            )

    def _evidence_audit(self, state: AgentState) -> None:
        if state.evidence_mode != "required":
            state.evidence_status = "not_required"
            state.verification_notes.append("普通对话未启用强制证据系统")
            return
        if state.retrieval_results and not state.evidence:
            state.verification_notes.append("检索结果缺少 ledger 证据")
        if "papers" in state.evidence_sources and not any(item.source_type == "paper" for item in state.evidence):
            state.verification_notes.append("学术/研究任务缺少论文或外部学术证据")
        if not state.evidence:
            state.verification_notes.append("必需证据模式缺少支持证据")
        if state.evidence_strictness == "soft":
            state.verification_notes.append("中等风险软证据策略：已触发检索，证据不足时仅允许受限回答")

    def _gate(self, state: AgentState) -> None:
        gate_evidence = self._qualified_gate_evidence(state)
        required_count = self.gate_policy.min_evidence_by_risk.get(state.risk_level, 1)
        state.qualified_evidence_ids = [item.id for item in gate_evidence]
        state.unqualified_evidence_ids = [
            item.id for item in state.evidence if item.id not in set(state.qualified_evidence_ids)
        ]
        state.evidence_status = _evidence_status(
            state=state,
            qualified_evidence=gate_evidence,
            required_count=required_count,
        )
        if state.evidence_category == "high_risk_action" and _is_protected_action_request(state.user_goal):
            state.qualified_evidence_ids = []
            state.unqualified_evidence_ids = [item.id for item in state.evidence]
            state.evidence_status = "protected_denied"
            reasons = [
                "保护性动作请求：生产、删除、绕过审批或破坏性执行不能由证据数量放行",
                "需要明确人工审批、授权凭证和安全替代方案；当前对话直接拒绝",
            ]
            decision = GateDecision(
                gate_id=f"gate_{stable_hash('|'.join([state.response_kind, state.risk_level, 'protected-deny', *reasons]))[:12]}",
                risk_level=state.risk_level,
                status="deny",
                reasons=reasons,
                required_evidence=state.qualified_evidence_ids,
            )
            state.verification_notes.append("保护性动作门禁拒绝")
            state.gate_decisions.append(decision)
            return
        if (
            state.evidence_mode == "required"
            and state.evidence_strictness == "soft"
            and len(gate_evidence) < required_count
            and self._evidence_search_was_attempted(state)
        ):
            reasons = [
                "中等风险软证据策略：已尝试检索但未检索到足够证据，允许受限回答",
                "输出必须明确证据不足，不能伪造引用或给出确定性结论",
            ]
            decision = GateDecision(
                gate_id=f"gate_{stable_hash('|'.join([state.response_kind, state.risk_level, state.evidence_status, *reasons]))[:12]}",
                risk_level=state.risk_level,
                status="allow",
                reasons=reasons,
                required_evidence=[item.id for item in state.evidence],
            )
            state.verification_notes.append("软证据门禁已允许受限回答")
            state.gate_decisions.append(decision)
            return
        decision = self.gate_policy.evaluate(
            action=state.response_kind,
            risk_level=state.risk_level,
            evidence=gate_evidence,
            target_path=None,
        )
        state.gate_decisions.append(decision)

    def _qualified_gate_evidence(self, state: AgentState) -> list:
        gate_evidence = state.evidence
        if state.evidence_category == "academic":
            return [
                item
                for item in state.evidence
                if item.source_type in QUALIFIED_EXTERNAL_SOURCE_TYPES
            ]
        if state.evidence_category in {
            "explicit_evidence_request",
            "current_factual",
            "decision_analysis",
            "technical_claim",
            "scientific_claim",
            "hard_reasoning",
            "regulated_advice",
            "regulated_domain",
        }:
            external_evidence = [
                item
                for item in state.evidence
                if item.source_type in QUALIFIED_EXTERNAL_SOURCE_TYPES
            ]
            if external_evidence:
                return external_evidence
            if not self._has_strong_local_rag(state):
                return []
        return gate_evidence

    def _evidence_search_was_attempted(self, state: AgentState) -> bool:
        return bool(state.external_search_attempted or "rag" in state.evidence_sources)

    def _has_strong_local_rag(self, state: AgentState) -> bool:
        threshold = float(self.config.gates.get("local_evidence_min_score", 0.45))
        semantic_threshold = float(self.config.gates.get("local_evidence_min_semantic_score", 0.25))
        for chunk in state.retrieval_results:
            if chunk.score < threshold:
                continue
            breakdown = getattr(chunk, "score_breakdown", {}) or {}
            method = getattr(chunk, "retrieval_method", "keyword")
            if "semantic" in method and breakdown.get("semantic", 0.0) < semantic_threshold:
                continue
            return True
        return False

    def _act_or_answer(self, state: AgentState) -> None:
        if state.gate_decisions[-1].status != "allow":
            state.answer = (
                "这是高风险或高难强推理任务，证据门禁未通过。请补充可靠证据、降低风险范围，"
                "或把问题拆成可验证的小任务后再继续。"
                if _has_cjk(state.user_goal)
                else "The answer gate did not pass; gather more evidence before responding."
            )

    def _verify(self, state: AgentState) -> None:
        if state.gate_decisions and state.gate_decisions[-1].status == "allow":
            state.verification_notes.append("门禁已通过")

    def _consolidate(self, state: AgentState) -> None:
        gate = state.gate_decisions[-1] if state.gate_decisions else None
        qualified = [
            item
            for item in state.evidence
            if item.id in set(state.qualified_evidence_ids)
            and item.source_type in QUALIFIED_EXTERNAL_SOURCE_TYPES
            and item.confidence >= 0.7
        ]
        if gate and gate.status == "allow" and qualified:
            engine = EvidenceConsolidationEngine(self.workspace_root / "evidence" / "consolidation-proposals")
            proposal = engine.propose(
                query=state.user_goal,
                category=state.evidence_category,
                risk_level=state.risk_level,
                evidence=qualified,
            )
            state.evidence_consolidation_proposals.append(proposal)
            state.pending_consolidation.append(f"已生成证据沉淀提案：{proposal['path']}")
            return
        state.pending_consolidation.append("不直接写入长期记忆或知识库；无合格外部证据时不生成证据沉淀提案。")

    def _respond(self, state: AgentState) -> None:
        return


def _requirement_from_routing(route: dict, user_goal: str) -> EvidenceRequirement:
    fallback = classify_evidence_requirement(user_goal)
    mode = str(route.get("evidence_mode") or fallback.mode).lower()
    if mode not in {"optional", "required"}:
        mode = fallback.mode
    risk_level = str(route.get("risk_level") or fallback.risk_level).lower()
    if risk_level not in {"none", "low", "medium", "high", "critical"}:
        risk_level = fallback.risk_level
    category = str(route.get("category") or fallback.category).strip() or fallback.category
    reasons = _string_list(route.get("reasons")) or list(fallback.reasons)
    sources = _source_list(route.get("sources")) or list(fallback.sources)
    if mode == "required" and not sources:
        sources = ["rag", "web", "papers", "user_experience"]
    return EvidenceRequirement(
        mode=mode,
        risk_level=risk_level,
        category=category,
        reasons=reasons,
        sources=sources,
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _source_list(value: object) -> list[str]:
    allowed = {"rag", "web", "papers", "user_experience"}
    return [item for item in _string_list(value) if item in allowed]


def _normalized_strictness(value: str) -> str:
    normalized = value.lower().strip()
    return normalized if normalized in {"none", "soft", "strict"} else ""


def _normalized_difficulty(value: str) -> str:
    normalized = value.lower().strip()
    return normalized if normalized in {"simple", "medium", "hard"} else "simple"


def _normalized_workflow_variant(value: str, *, requirement: EvidenceRequirement, strictness: str) -> str:
    normalized = value.lower().strip()
    if normalized in {"routine", "evidence_soft", "evidence_strict", "protected_action"}:
        return normalized
    if requirement.category == "high_risk_action":
        return "protected_action"
    if requirement.mode != "required":
        return "routine"
    if strictness == "strict" or requirement.risk_level in {"high", "critical"}:
        return "evidence_strict"
    return "evidence_soft"


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _localized_purpose(value: str, *, chinese: bool) -> str:
    if not chinese:
        return value
    known = "Build evidence-first heavy-reasoning agents with state gates, knowledge, memory, and reviewed self-evolution."
    if value == known:
        return "证据优先、带状态门禁、知识库、记忆和审核式自进化的重推理 Agent 开发"
    return value


def _evidence_strictness(*, category: str, risk_level: str, user_goal: str = "") -> str:
    if risk_level in {"high", "critical"} or category in STRICT_EVIDENCE_CATEGORIES:
        return "strict"
    if category == "current_factual" and _needs_strict_current_factual(user_goal):
        return "strict"
    if risk_level == "medium":
        return "soft"
    return "none"


def _evidence_status(*, state: AgentState, qualified_evidence: list, required_count: int) -> str:
    if state.evidence_mode != "required":
        return "not_required"
    if len(qualified_evidence) >= required_count:
        return "sufficient"
    if qualified_evidence:
        return "partial"
    if state.evidence:
        return "unqualified"
    if state.external_search_attempted or "rag" in state.evidence_sources:
        return "exhausted"
    return "missing"


def _needs_strict_current_factual(value: str) -> bool:
    text = value.lower()
    has_year_or_current = bool(re.search(r"\b20\d{2}\b", text)) or any(
        term in text for term in ["最新", "当前", "现在", "current", "latest", "recent"]
    )
    asks_sources = any(term in text for term in STRICT_CURRENT_FACTUAL_TERMS)
    return has_year_or_current and asks_sources


def _is_protected_action_request(value: str) -> bool:
    text = value.lower()
    return any(term in text for term in PROTECTED_ACTION_TERMS)


def _is_template_question(value: str) -> bool:
    text = value.lower().strip()
    terms = [
        "工作流",
        "状态机",
        "证据系统",
        "rag",
        "门禁",
        "记忆",
        "技能",
        "workflow",
        "state machine",
        "evidence",
        "gates",
        "memory",
        "skills",
        "constraints",
    ]
    return any(term in text for term in terms)


def _requires_evidence_system(value: str) -> bool:
    return classify_evidence_requirement(value).mode == "required"


def _is_identity_question(value: str) -> bool:
    text = value.lower().strip()

    chinese_identity_terms = [
        "你是谁",
        "你是啥",
        "你是什么",
        "你能做什么",
        "你可以做什么",
        "介绍一下你",
    ]

    if any(
        term in text
        for term in chinese_identity_terms
    ):
        return True

    # “你好”只有在整句话基本就是问候时才触发，
    # 避免把“请改写‘你好’”误判成身份问题。
    if re.fullmatch(
        r"\s*(你好|您好)[！!。.?？]*\s*",
        text,
    ):
        return True

    english_identity_patterns = [
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat can you do\b",
    ]

    if any(
        re.search(pattern, text)
        for pattern in english_identity_patterns
    ):
        return True

    # hello / hi 同样只在独立问候时触发。
    return bool(
        re.fullmatch(
            r"\s*(hello|hi)[!.?]*\s*",
            text,
        )
    )



def _knowledge_methods(config: dict[str, Any]) -> list[str]:
    methods = config.get("retrieval_methods")
    if isinstance(methods, str):
        return [item.strip() for item in methods.split(",") if item.strip()]
    if isinstance(methods, list):
        return [str(item).strip() for item in methods if str(item).strip()]
    index_type = str(config.get("index_type", "keyword")).strip() or "keyword"
    if index_type in {"hybrid", "local-hybrid"}:
        return ["bm25", "semantic", "graph"]
    if index_type in {"local-keyword", "keyword"}:
        return ["keyword"]
    return [index_type]


def _knowledge_fallback_methods(config: dict[str, Any]) -> list[str]:
    methods = config.get("fallback_methods")
    if isinstance(methods, str):
        return [item.strip() for item in methods.split(",") if item.strip()]
    if isinstance(methods, list):
        return [str(item).strip() for item in methods if str(item).strip()]
    if bool(config.get("wiki_enabled", False)):
        return ["wiki"]
    return []


def _dedupe_knowledge_chunks(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    best: dict[tuple[str, str], KnowledgeChunk] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.span)
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk
    return sorted(best.values(), key=lambda item: (-item.score, item.source, item.span))
