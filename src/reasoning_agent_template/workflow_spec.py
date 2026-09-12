from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reasoning_agent_template.models import GateDecision, stable_hash, utc_now


WORKFLOW_SPEC_VERSION = "1.0"
DEFAULT_WORKFLOW_SPEC_PATH = Path("configs/workflows/default.workflow.json")
DEFAULT_DRAFT_PATH = Path("configs/workflows/default.workflow.draft.json")
PROPOSAL_DIR = Path("configs/workflows/proposals")
SUPPORTED_HANDLER_KINDS = {"builtin", "plugin_tool"}
BUILTIN_STAGE_HANDLERS = {
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
    "passthrough",
    "review_note",
}


@dataclass(frozen=True)
class WorkflowNodeSpec:
    id: str
    label: str
    agent: str
    description: str
    work: str
    input_contract: str
    output_contract: str
    handler_kind: str = "builtin"
    handler: str = ""
    checkpoint: bool = False
    gate_policy: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowNodeSpec":
        node_id = str(data.get("id", "")).strip()
        return cls(
            id=node_id,
            label=str(data.get("label") or node_id),
            agent=str(data.get("agent") or "coordinator"),
            description=str(data.get("description") or ""),
            work=str(data.get("work") or data.get("description") or ""),
            input_contract=str(data.get("input_contract") or data.get("input") or ""),
            output_contract=str(data.get("output_contract") or data.get("output") or ""),
            handler_kind=str(data.get("handler_kind") or "builtin"),
            handler=str(data.get("handler") or node_id),
            checkpoint=bool(data.get("checkpoint", False)),
            gate_policy=dict(data.get("gate_policy") or {}),
            ui=dict(data.get("ui") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowEdgeSpec:
    id: str
    from_node: str
    to_node: str
    type: str = "flow"
    condition: str = ""
    handoff_contract: dict[str, Any] = field(default_factory=dict)
    gate_policy: dict[str, Any] = field(default_factory=dict)
    planner_contract: dict[str, Any] = field(default_factory=dict)
    reviewer_required: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowEdgeSpec":
        from_node = str(data.get("from") or data.get("from_node") or "").strip()
        to_node = str(data.get("to") or data.get("to_node") or "").strip()
        edge_id = str(data.get("id") or f"{from_node}->{to_node}").strip()
        return cls(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            type=str(data.get("type") or "flow"),
            condition=str(data.get("condition") or data.get("label") or ""),
            handoff_contract=dict(data.get("handoff_contract") or {}),
            gate_policy=dict(data.get("gate_policy") or {}),
            planner_contract=dict(data.get("planner_contract") or {}),
            reviewer_required=bool(data.get("reviewer_required", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["from"] = data.pop("from_node")
        data["to"] = data.pop("to_node")
        return data


@dataclass(frozen=True)
class WorkflowValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_code: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowSpec:
    version: str
    name: str
    revision: str
    start_node: str
    terminal_nodes: list[str]
    nodes: list[WorkflowNodeSpec]
    edges: list[WorkflowEdgeSpec]
    protected_nodes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSpec":
        nodes = [WorkflowNodeSpec.from_dict(item) for item in data.get("nodes", [])]
        edges = [WorkflowEdgeSpec.from_dict(item) for item in data.get("edges", [])]
        return cls(
            version=str(data.get("version") or WORKFLOW_SPEC_VERSION),
            name=str(data.get("name") or "default"),
            revision=str(data.get("revision") or "dev"),
            start_node=str(data.get("start_node") or (nodes[0].id if nodes else "")),
            terminal_nodes=[str(item) for item in data.get("terminal_nodes", [])],
            nodes=nodes,
            edges=edges,
            protected_nodes=[str(item) for item in data.get("protected_nodes", [])],
        )

    @classmethod
    def default(cls) -> "WorkflowSpec":
        return cls.from_dict(DEFAULT_WORKFLOW_SPEC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "revision": self.revision,
            "start_node": self.start_node,
            "terminal_nodes": list(self.terminal_nodes),
            "protected_nodes": list(self.protected_nodes),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def node_map(self) -> dict[str, WorkflowNodeSpec]:
        return {node.id: node for node in self.nodes}

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def checkpoints(self) -> list[str]:
        return [node.id for node in self.nodes if node.checkpoint]

    def execution_nodes(self) -> list[WorkflowNodeSpec]:
        return list(self.nodes)

    def outgoing_edges(self, node_id: str) -> list[WorkflowEdgeSpec]:
        return [edge for edge in self.edges if edge.from_node == node_id]

    def stage_agents(self) -> dict[str, str]:
        return {node.id: node.agent for node in self.nodes}

    def validate(
        self,
        *,
        known_builtin_handlers: set[str] | None = None,
        base: "WorkflowSpec | None" = None,
    ) -> WorkflowValidation:
        known = known_builtin_handlers or BUILTIN_STAGE_HANDLERS
        errors: list[str] = []
        warnings: list[str] = []
        requires_code: list[str] = []

        if self.version != WORKFLOW_SPEC_VERSION:
            errors.append(f"unsupported workflow spec version: {self.version}")
        if not self.nodes:
            errors.append("workflow must contain at least one node")
        duplicate_nodes = _duplicates(node.id for node in self.nodes)
        duplicate_edges = _duplicates(edge.id for edge in self.edges)
        errors.extend(f"duplicate node id: {node_id}" for node_id in duplicate_nodes)
        errors.extend(f"duplicate edge id: {edge_id}" for edge_id in duplicate_edges)

        ids = {node.id for node in self.nodes}
        for node in self.nodes:
            if not _VALID_ID.match(node.id):
                errors.append(f"invalid node id: {node.id}")
            if node.handler_kind not in SUPPORTED_HANDLER_KINDS:
                errors.append(f"unsupported handler_kind for {node.id}: {node.handler_kind}")
            if node.handler_kind == "builtin" and node.handler not in known:
                requires_code.append(f"node {node.id} requires builtin handler: {node.handler}")
            if node.handler_kind == "plugin_tool" and not node.handler:
                errors.append(f"plugin_tool node {node.id} must declare handler")

        if self.start_node not in ids:
            errors.append(f"start_node is missing from nodes: {self.start_node}")
        for terminal in self.terminal_nodes:
            if terminal not in ids:
                errors.append(f"terminal node is missing from nodes: {terminal}")
        for protected in self.protected_nodes:
            if protected not in ids:
                errors.append(f"protected node is missing from nodes: {protected}")

        for edge in self.edges:
            if not edge.from_node or not edge.to_node:
                errors.append(f"edge {edge.id} must declare from and to")
            if edge.from_node not in ids:
                errors.append(f"edge {edge.id} references missing from node: {edge.from_node}")
            if edge.to_node not in ids:
                errors.append(f"edge {edge.id} references missing to node: {edge.to_node}")

        if not errors and self.start_node in ids:
            reachable = _reachable_nodes(self.start_node, self.edges)
            unreachable = sorted(ids - reachable)
            if unreachable:
                errors.append(f"unreachable nodes from start_node: {', '.join(unreachable)}")
            terminal_reachable = [terminal for terminal in self.terminal_nodes if terminal in reachable]
            if self.terminal_nodes and not terminal_reachable:
                errors.append("no terminal node is reachable from start_node")

        if base is not None:
            missing_protected = [node_id for node_id in base.protected_nodes if node_id not in ids]
            errors.extend(f"protected node cannot be deleted: {node_id}" for node_id in missing_protected)
            for node_id in base.protected_nodes:
                current = self.node_map().get(node_id)
                original = base.node_map().get(node_id)
                if current and original and current.handler != original.handler:
                    warnings.append(f"protected node handler changed: {node_id}")

        return WorkflowValidation(
            ok=not errors and not requires_code,
            errors=errors,
            warnings=warnings,
            requires_code=requires_code,
        )


class WorkflowSpecStore:
    def __init__(self, workspace_root: str | Path, config: dict[str, Any] | None = None):
        self.workspace_root = Path(workspace_root)
        self.config = dict(config or {})
        self.spec_path = _resolve_workspace_path(
            self.workspace_root,
            self.config.get("workflow_spec") or DEFAULT_WORKFLOW_SPEC_PATH,
        )
        self.draft_path = _resolve_workspace_path(
            self.workspace_root,
            self.config.get("workflow_draft") or DEFAULT_DRAFT_PATH,
        )
        self.proposal_dir = _resolve_workspace_path(
            self.workspace_root,
            self.config.get("workflow_proposal_dir") or PROPOSAL_DIR,
        )

    def load(self) -> WorkflowSpec:
        if self.spec_path.exists():
            return _read_spec(self.spec_path)
        return WorkflowSpec.default()

    def load_draft(self) -> WorkflowSpec | None:
        if not self.draft_path.exists():
            return None
        return _read_spec(self.draft_path)

    def save_draft(self, spec: WorkflowSpec) -> dict[str, Any]:
        self.draft_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.draft_path, spec.to_dict())
        validation = spec.validate(base=self.load())
        return {"path": str(self.draft_path), "spec": spec.to_dict(), "validation": validation.to_dict()}

    def clear_draft(self) -> None:
        if self.draft_path.exists():
            self.draft_path.unlink()

    def create_proposal(self, spec: WorkflowSpec | None = None) -> dict[str, Any]:
        base = self.load()
        candidate = spec or self.load_draft()
        if candidate is None:
            raise ValueError("workflow draft is required before creating a proposal")
        validation = candidate.validate(base=base)
        draft_hash = stable_hash(_canonical_json(candidate.to_dict()))
        proposal_id = f"wfp_{draft_hash[:12]}"
        target_relative = self.spec_path.relative_to(self.workspace_root).as_posix()
        proposal = {
            "proposal_id": proposal_id,
            "created_at": utc_now(),
            "status": "pending_approval",
            "draft_hash": draft_hash,
            "target_path": target_relative,
            "modified_files": [target_relative],
            "test_command": "$env:PYTHONPATH='src'; python -m unittest discover -s tests -v",
            "validation": validation.to_dict(),
            "diff_preview": workflow_diff(base, candidate),
            "spec": candidate.to_dict(),
            "gate_decision": GateDecision(
                gate_id=f"gate_{draft_hash[:12]}",
                risk_level="medium" if validation.requires_code else "low",
                status="interrupt",
                reasons=["workflow spec changes require explicit approval before code-modifier apply"],
                required_evidence=[],
            ).to_dict(),
        }
        self.proposal_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.proposal_dir / f"{proposal_id}.json", proposal)
        return proposal

    def load_proposal(self, proposal_id: str) -> dict[str, Any]:
        proposal_path = self._proposal_path(proposal_id)
        if not proposal_path.exists():
            raise FileNotFoundError(f"workflow proposal not found: {proposal_id}")
        return json.loads(proposal_path.read_text(encoding="utf-8"))

    def save_proposal(self, proposal: dict[str, Any]) -> None:
        proposal_id = str(proposal.get("proposal_id") or "")
        if not proposal_id:
            raise ValueError("proposal_id is required")
        self.proposal_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self._proposal_path(proposal_id), proposal)

    def _proposal_path(self, proposal_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", proposal_id)
        return self.proposal_dir / f"{safe_id}.json"


def load_workflow_spec(workspace_root: str | Path, runtime_config: dict[str, Any] | None = None) -> WorkflowSpec:
    return WorkflowSpecStore(workspace_root, runtime_config).load()


def workflow_diff(before: WorkflowSpec, after: WorkflowSpec) -> str:
    before_lines = _canonical_json(before.to_dict()).splitlines(keepends=True)
    after_lines = _canonical_json(after.to_dict()).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="current.workflow.json",
            tofile="draft.workflow.json",
        )
    )


def _read_spec(path: Path) -> WorkflowSpec:
    return WorkflowSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(_canonical_json(data) + "\n", encoding="utf-8")


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def _duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        text = str(value)
        if text in seen and text not in duplicates:
            duplicates.append(text)
        seen.add(text)
    return duplicates


def _reachable_nodes(start_node: str, edges: list[WorkflowEdgeSpec]) -> set[str]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.from_node, []).append(edge.to_node)
    reachable = {start_node}
    frontier = [start_node]
    while frontier:
        current = frontier.pop()
        for target in adjacency.get(current, []):
            if target not in reachable:
                reachable.add(target)
                frontier.append(target)
    return reachable


_VALID_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_:-]*$")


DEFAULT_WORKFLOW_SPEC: dict[str, Any] = {
    "version": WORKFLOW_SPEC_VERSION,
    "name": "default-heavy-reasoning-workflow",
    "revision": "baseline",
    "start_node": "intake",
    "terminal_nodes": ["respond"],
    "protected_nodes": [
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
    "nodes": [
        {
            "id": "intake",
            "label": "接收",
            "agent": "coordinator",
            "description": "接收目标并建立运行上下文",
            "work": "规范化用户目标，形成难度、风险和证据工作流上下文。",
            "input_contract": "用户问题",
            "output_contract": "规范化目标",
            "handler_kind": "builtin",
            "handler": "intake",
            "checkpoint": False,
        },
        {
            "id": "plan",
            "label": "计划",
            "agent": "planner",
            "description": "生成可审计的推理计划",
            "work": "根据路由和 reviewer 审查生成可执行计划。",
            "input_contract": "规范化目标",
            "output_contract": "计划步骤",
            "handler_kind": "builtin",
            "handler": "plan",
            "checkpoint": False,
        },
        {
            "id": "retrieve",
            "label": "检索",
            "agent": "retriever",
            "description": "检索本地知识库、外部论文和用户经验证据",
            "work": "按证据策略检索 RAG、论文、网页或用户经验证据。",
            "input_contract": "计划步骤",
            "output_contract": "证据候选",
            "handler_kind": "builtin",
            "handler": "retrieve",
            "checkpoint": True,
        },
        {
            "id": "reason",
            "label": "推理",
            "agent": "reasoner",
            "description": "基于证据生成草案答案",
            "work": "基于当前证据、记忆和计划生成答案草案。",
            "input_contract": "证据候选",
            "output_contract": "答案草案",
            "handler_kind": "builtin",
            "handler": "reason",
            "checkpoint": False,
        },
        {
            "id": "evidence_audit",
            "label": "证据审计",
            "agent": "critic",
            "description": "检查关键结论是否绑定证据",
            "work": "审计答案草案和证据状态，记录不足与合格证据。",
            "input_contract": "答案草案",
            "output_contract": "证据审计结果",
            "handler_kind": "builtin",
            "handler": "evidence_audit",
            "checkpoint": True,
        },
        {
            "id": "gate",
            "label": "门禁",
            "agent": "critic",
            "description": "执行风险、证据和审批门禁",
            "work": "根据风险、证据和审批策略作出 allow/interruption/deny。",
            "input_contract": "证据审计结果",
            "output_contract": "门禁决策",
            "handler_kind": "builtin",
            "handler": "gate",
            "checkpoint": True,
            "gate_policy": {"approval_action": "answer", "risk": "dynamic"},
        },
        {
            "id": "act_or_answer",
            "label": "行动/回答",
            "agent": "coordinator",
            "description": "根据门禁结果行动或回答",
            "work": "在门禁允许时发布答案，阻断时返回受限状态。",
            "input_contract": "门禁决策",
            "output_contract": "可发布结果",
            "handler_kind": "builtin",
            "handler": "act_or_answer",
            "checkpoint": False,
        },
        {
            "id": "verify",
            "label": "验证",
            "agent": "critic",
            "description": "验证状态、证据和输出一致性",
            "work": "检查输出、门禁和证据状态是否一致。",
            "input_contract": "可发布结果",
            "output_contract": "验证记录",
            "handler_kind": "builtin",
            "handler": "verify",
            "checkpoint": True,
        },
        {
            "id": "consolidate",
            "label": "沉淀",
            "agent": "memory",
            "description": "只生成记忆/演进候选，不直接写保护资产",
            "work": "把合格证据生成可审查沉淀提案，不直接改长期记忆或技能。",
            "input_contract": "验证记录",
            "output_contract": "沉淀候选",
            "handler_kind": "builtin",
            "handler": "consolidate",
            "checkpoint": True,
        },
        {
            "id": "respond",
            "label": "响应",
            "agent": "coordinator",
            "description": "输出最终答案和调试遥测",
            "work": "输出最终答案、参考文献索引和调试 payload。",
            "input_contract": "沉淀候选",
            "output_contract": "最终响应",
            "handler_kind": "builtin",
            "handler": "respond",
            "checkpoint": False,
        },
    ],
    "edges": [
        {
            "id": "intake_to_plan",
            "from": "intake",
            "to": "plan",
            "type": "flow",
            "condition": "normalized goal",
            "handoff_contract": {"payload": "routing + normalized_goal"},
            "reviewer_required": True,
        },
        {"id": "plan_to_retrieve", "from": "plan", "to": "retrieve", "type": "branch", "condition": "evidence required"},
        {"id": "plan_to_reason", "from": "plan", "to": "reason", "type": "branch", "condition": "routine path"},
        {"id": "retrieve_to_reason", "from": "retrieve", "to": "reason", "type": "flow", "condition": "evidence context"},
        {"id": "reason_to_evidence_audit", "from": "reason", "to": "evidence_audit", "type": "flow", "condition": "claim audit"},
        {"id": "evidence_audit_to_gate", "from": "evidence_audit", "to": "gate", "type": "flow", "condition": "audit result"},
        {"id": "evidence_audit_to_retrieve", "from": "evidence_audit", "to": "retrieve", "type": "retry", "condition": "evidence gap"},
        {"id": "gate_to_act_or_answer", "from": "gate", "to": "act_or_answer", "type": "flow", "condition": "gate decision"},
        {"id": "gate_to_retrieve", "from": "gate", "to": "retrieve", "type": "retry", "condition": "gate blocked"},
        {"id": "act_or_answer_to_verify", "from": "act_or_answer", "to": "verify", "type": "flow", "condition": "answer/action"},
        {"id": "verify_to_consolidate", "from": "verify", "to": "consolidate", "type": "flow", "condition": "verified output"},
        {"id": "verify_to_reason", "from": "verify", "to": "reason", "type": "revise", "condition": "verification failed"},
        {"id": "consolidate_to_respond", "from": "consolidate", "to": "respond", "type": "flow", "condition": "publishable result"},
        {"id": "consolidate_to_evidence_audit", "from": "consolidate", "to": "evidence_audit", "type": "loop", "condition": "proposal audit"},
    ],
}
