from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    source_type: str
    uri: str
    locator: str
    content_hash: str
    summary: str
    confidence: float
    collected_at: str
    used_for: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(
            id=str(data["id"]),
            source_type=str(data["source_type"]),
            uri=str(data["uri"]),
            locator=str(data["locator"]),
            content_hash=str(data["content_hash"]),
            summary=str(data["summary"]),
            confidence=float(data["confidence"]),
            collected_at=str(data["collected_at"]),
            used_for=list(data.get("used_for", [])),
        )


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    span: str
    text: str
    content_hash: str
    score: float
    evidence_id: str
    retrieval_method: str = "keyword"
    score_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRequirement:
    mode: str
    risk_level: str
    category: str
    reasons: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GateDecision:
    gate_id: str
    risk_level: str
    status: str
    reasons: list[str]
    required_evidence: list[str] = field(default_factory=list)
    approved_by: str | None = None
    state_snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    user_goal: str = ""
    current_stage: str = "intake"
    response_kind: str = "routine"
    difficulty: str = "simple"
    workflow_variant: str = "routine"
    routing_source: str = "rules"
    routing_confidence: float = 0.0
    routing_decision: dict[str, Any] = field(default_factory=dict)
    reviewer_decision: dict[str, Any] = field(default_factory=dict)
    reviewer_status: str = "not_run"
    risk_level: str = "none"
    evidence_mode: str = "optional"
    evidence_strictness: str = "none"
    evidence_status: str = "not_required"
    evidence_category: str = "routine"
    evidence_reasons: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    retrieval_results: list[KnowledgeChunk] = field(default_factory=list)
    rag_methods: list[str] = field(default_factory=list)
    rag_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    external_results: list[KnowledgeChunk] = field(default_factory=list)
    external_search_attempted: list[str] = field(default_factory=list)
    external_search_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    qualified_evidence_ids: list[str] = field(default_factory=list)
    unqualified_evidence_ids: list[str] = field(default_factory=list)
    gate_decisions: list[GateDecision] = field(default_factory=list)
    action_results: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    claim_review: dict[str, Any] = field(default_factory=dict)
    pending_consolidation: list[str] = field(default_factory=list)
    evidence_consolidation_proposals: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    stage_trace: list[str] = field(default_factory=list)
    stage_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowResult:
    answer: str
    state: AgentState
    stage_trace: list[str]
    evidence: list[EvidenceItem]
    gate_decisions: list[GateDecision]


@dataclass(frozen=True)
class MemoryWriteResult:
    partition: str
    key: str
    decision: GateDecision
    path: Path | None = None


@dataclass(frozen=True)
class EvolutionProposal:
    proposal_id: str
    target: str
    path: Path
    status: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class RuntimeHandle:
    backend: str
    invoke: Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
