from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    identity: dict[str, Any]
    models: dict[str, Any]
    knowledge: dict[str, Any]
    memory: dict[str, Any]
    gates: dict[str, Any]
    skills: dict[str, Any]
    evolution: dict[str, Any]
    runtime: dict[str, Any] = field(default_factory=dict)
    workspace_root: Path = Path(".")

    @classmethod
    def default(cls, *, workspace_root: Path) -> "AgentConfig":
        return cls(
            identity={
                "name": "reasoning-agent-template",
                "purpose": "Build evidence-first heavy-reasoning agents.",
                "audience": "agent developers",
                "failure_boundaries": ["never mutate without gates", "never answer key claims without evidence"],
            },
            models={
                "planner": {"provider": "local", "model": "deterministic"},
                "worker": {"provider": "local", "model": "deterministic"},
                "critic": {"provider": "local", "model": "deterministic"},
                "grader": {"provider": "local", "model": "deterministic"},
            },
            knowledge={
                "directory": "knowledge",
                "index_type": "hybrid",
                "retrieval_methods": ["bm25", "semantic", "graph"],
                "fallback_methods": [],
                "top_k": 5,
                "min_score": 0.1,
                "chunk_size": 1400,
                "fallback_min_score": 0.35,
                "fallback_top_k": 2,
                "wiki_top_k": 2,
                "wiki_enabled": False,
                "external_top_k": 5,
                "external_timeout_seconds": 8,
            },
            memory={
                "directory": "memory",
                "partitions": ["semantic", "episodic", "procedural", "project", "user"],
                "read_only_partitions": ["shared"],
                "write_requires_gate": True,
            },
            gates={
                "min_evidence_by_risk": {"low": 1, "medium": 1, "high": 2, "critical": 2},
                "approval_required_actions": ["write_file", "write_memory", "update_skill", "execute_command"],
                "workspace_root": str(workspace_root),
            },
            skills={
                "directory": "skills",
                "enabled": [
                    "project-intake",
                    "evidence-first",
                    "state-gates",
                    "knowledge-rag",
                    "memory-consolidation",
                    "minimal-change",
                    "self-evolution",
                    "configurator",
                    "testing-verification",
                ],
            },
            evolution={
                "enabled": True,
                "proposal_directory": "evolution/proposals",
                "direct_mutation_allowed": False,
                "approval_required": True,
            },
            runtime={
                "prefer_deepagents": False,
                "workflow_spec": "configs/workflows/default.workflow.json",
                "workflow_draft": "configs/workflows/default.workflow.draft.json",
                "workflow_proposal_dir": "configs/workflows/proposals",
                "agents_spec": "configs/agents/default.agents.json",
                "agents_draft": "configs/agents/default.agents.draft.json",
                "agents_proposal_dir": "configs/agents/proposals",
                "template_builtin_dir": "configs/templates/builtin",
                "template_user_dir": "configs/templates/user",
                "code_modifier": {"mode": "local_spec"},
            },
            workspace_root=Path(workspace_root),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, workspace_root: Path) -> "AgentConfig":
        defaults = cls.default(workspace_root=workspace_root)
        return cls(
            identity=data.get("identity", defaults.identity),
            models=data.get("models", defaults.models),
            knowledge=data.get("knowledge", defaults.knowledge),
            memory=data.get("memory", defaults.memory),
            gates=data.get("gates", defaults.gates),
            skills=data.get("skills", defaults.skills),
            evolution=data.get("evolution", defaults.evolution),
            runtime=data.get("runtime", defaults.runtime),
            workspace_root=workspace_root,
        )


def load_agent_config(path: Path) -> AgentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = _load_mapping(text)
    return AgentConfig.from_dict(data, workspace_root=path.parent.resolve())


def _load_mapping(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    value, _ = _parse_block(lines, 0, 0)
    if not isinstance(value, dict):
        raise ValueError("agent config must be a mapping")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    is_list = lines[index][1].startswith("- ")
    if is_list:
        values = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            item = lines[index][1][2:].strip()
            values.append(_parse_scalar(item))
            index += 1
        return values, index

    values: dict[str, Any] = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"unexpected indentation near {content}")
        key, _, raw_value = content.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            values[key] = _parse_scalar(raw_value)
        elif index < len(lines) and lines[index][0] > current_indent:
            child, index = _parse_block(lines, index, lines[index][0])
            values[key] = child
        else:
            values[key] = {}
    return values, index


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value == "[]":
        return []
    if value == "{}":
        return {}
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
