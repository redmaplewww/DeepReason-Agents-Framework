from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from reasoning_agent_template.gates import GatePolicy
from reasoning_agent_template.models import EvidenceItem, GateDecision, MemoryWriteResult, stable_hash, utc_now


WRITABLE_PARTITIONS = {"semantic", "episodic", "procedural", "project", "user"}
KNOWLEDGE_BOUNDARY_TERMS = [
    "论文",
    "文档",
    "资料",
    "知识库",
    "api 文档",
    "api文档",
    "源码",
    "规范",
    "手册",
    "报告",
    "paper",
    "document",
    "documentation",
    "manual",
    "specification",
    "source material",
]


class ShortTermConversationMemory:
    """In-process rolling conversation memory for the active web/CLI session."""

    def __init__(self, *, max_turns: int = 8):
        self.max_turns = max(1, max_turns)
        self._turns: list[dict[str, Any]] = []

    def append(self, *, user: str, assistant: str, run_id: str) -> None:
        self._turns.append(
            {
                "run_id": run_id,
                "user": user,
                "assistant": assistant,
                "recorded_at": utc_now(),
            }
        )
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def restore(self, turns: list[dict[str, Any]]) -> None:
        for turn in turns:
            user = str(turn.get("user", "")).strip()
            assistant = str(turn.get("assistant", "")).strip()
            if user or assistant:
                self.append(
                    user=user,
                    assistant=assistant,
                    run_id=str(turn.get("run_id") or "restored"),
                )

    def snapshot(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        turns = self._turns[-limit:] if limit else self._turns
        return [dict(turn) for turn in turns]

    def count(self) -> int:
        return len(self._turns)


class LongTermMemoryStore:
    """Partitioned JSONL memory store guarded by GatePolicy."""

    def __init__(self, root: Path, *, gate_policy: GatePolicy):
        self.root = Path(root)
        self.gate_policy = gate_policy

    def write(
        self,
        *,
        partition: str,
        key: str,
        value: Any,
        evidence: list[EvidenceItem],
        approved_by: str | None = None,
    ) -> MemoryWriteResult:
        if partition not in WRITABLE_PARTITIONS:
            decision = self.gate_policy.evaluate(
                action="write_memory",
                risk_level="medium",
                evidence=evidence,
                target_path=self._path_for(partition),
                approved_by=approved_by,
            )
            decision = type(decision)(
                gate_id=decision.gate_id,
                risk_level=decision.risk_level,
                status="deny",
                reasons=[*decision.reasons, f"{partition} memory is read-only or unknown"],
                required_evidence=decision.required_evidence,
                approved_by=decision.approved_by,
                state_snapshot_id=decision.state_snapshot_id,
            )
            return MemoryWriteResult(partition=partition, key=key, decision=decision)

        path = self._path_for(partition)
        decision = self.gate_policy.evaluate(
            action="write_memory",
            risk_level="medium",
            evidence=evidence,
            target_path=path,
            approved_by=approved_by,
        )
        if decision.status != "allow":
            return MemoryWriteResult(partition=partition, key=key, decision=decision)

        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "partition": partition,
            "key": key,
            "value": value,
            "evidence_ids": [item.id for item in evidence],
            "written_at": utc_now(),
            "approved_by": approved_by,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return MemoryWriteResult(partition=partition, key=key, decision=decision, path=path)

    def read(self, partition: str, key: str) -> Any:
        path = self._path_for(partition)
        if not path.exists():
            return None
        found = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("key") == key:
                        found = record.get("value")
        return found

    def list_partition(self, partition: str) -> list[dict[str, Any]]:
        path = self._path_for(partition)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _path_for(self, partition: str) -> Path:
        value = str(partition).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError("memory partition must be a simple alphanumeric name")
        root = self.root.resolve()
        path = (root / f"{value}.jsonl").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"memory partition escapes memory store: {partition}") from exc
        return path


def explicit_memory_candidate(message: str) -> dict[str, str] | None:
    text = " ".join(message.strip().split())
    lowered = text.lower()
    triggers = [
        "请记住",
        "帮我记住",
        "你要记住",
        "记住：",
        "记住:",
        "记住 ",
        "remember:",
        "remember that",
    ]
    if not any(trigger in lowered or trigger in text for trigger in triggers):
        return None

    content = text
    for trigger in triggers:
        content = content.replace(trigger, "")
    content = content.strip(" ：:，,。.")
    if not content:
        content = text

    if _looks_like_knowledge_base_content(content):
        return {
            "partition": "knowledge",
            "key": f"knowledge_candidate_{stable_hash(content)[:10]}",
            "value": content,
            "boundary": "knowledge_base",
        }

    key = f"user_fact_{stable_hash(content)[:10]}"
    for separator in ["是", "叫", "为", "="]:
        marker = f"我的"
        if marker in content and separator in content:
            before, _, after = content.partition(separator)
            field_name = before.replace(marker, "").strip(" ：:，,。.")
            if field_name and after.strip():
                key = field_name[:40]
                break

    return {"partition": "user", "key": key, "value": content, "boundary": "long_term_memory"}


def deny_knowledge_memory_write(*, key: str, value: str, evidence: list[EvidenceItem]) -> MemoryWriteResult:
    reasons = [
        "content belongs in knowledge base; add it under knowledge/ and ingest it instead of writing long-term memory"
    ]
    gate_id = f"gate_{stable_hash('|'.join(['write_memory', 'knowledge_base', value, *reasons]))[:12]}"
    decision = GateDecision(
        gate_id=gate_id,
        risk_level="medium",
        status="deny",
        reasons=reasons,
        required_evidence=[item.id for item in evidence],
        approved_by="boundary_policy",
    )
    return MemoryWriteResult(partition="knowledge", key=key, decision=decision)


def _looks_like_knowledge_base_content(value: str) -> bool:
    text = value.lower()
    return any(term in text for term in KNOWLEDGE_BOUNDARY_TERMS)
