from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_agent_template.agents_spec import AgentsSpec, AgentsSpecStore
from reasoning_agent_template.models import utc_now
from reasoning_agent_template.workflow_spec import WorkflowSpec, WorkflowSpecStore


AGENT_TEMPLATE_VERSION = "1.0"
DEFAULT_TEMPLATE_BUILTIN_DIR = Path("configs/templates/builtin")
DEFAULT_TEMPLATE_USER_DIR = Path("configs/templates/user")
_TEMPLATE_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,79}$")


@dataclass(frozen=True)
class AgentTemplate:
    id: str
    label: str
    description: str
    version: str
    agents: AgentsSpec
    workflow: WorkflowSpec
    tags: list[str] = field(default_factory=list)
    source: str = "builtin"
    path: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source: str,
        path: Path | None = None,
        template_id: str | None = None,
    ) -> "AgentTemplate":
        resolved_id = _validate_template_id(str(data.get("id") or template_id or ""))
        return cls(
            id=resolved_id,
            label=str(data.get("label") or resolved_id),
            description=str(data.get("description") or ""),
            version=str(data.get("version") or AGENT_TEMPLATE_VERSION),
            agents=AgentsSpec.from_dict(dict(data.get("agents") or {})),
            workflow=WorkflowSpec.from_dict(dict(data.get("workflow") or {})),
            tags=[str(item) for item in data.get("tags", [])],
            source=source,
            path=str(path or ""),
            created_at=str(data.get("created_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "agents": self.agents.to_dict(),
            "workflow": self.workflow.to_dict(),
        }
        if self.created_at:
            data["created_at"] = self.created_at
        return data

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "source": self.source,
            "path": self.path,
            "created_at": self.created_at,
            "agents_count": len(self.agents.agents),
            "workflow_nodes_count": len(self.workflow.nodes),
        }


class AgentTemplateStore:
    def __init__(self, workspace_root: str | Path, config: dict[str, Any] | None = None):
        self.workspace_root = Path(workspace_root)
        self.config = dict(config or {})
        self.builtin_dir = _resolve_workspace_path(
            self.workspace_root,
            self.config.get("template_builtin_dir") or DEFAULT_TEMPLATE_BUILTIN_DIR,
        )
        self.user_dir = _resolve_workspace_path(
            self.workspace_root,
            self.config.get("template_user_dir") or DEFAULT_TEMPLATE_USER_DIR,
        )

    def list_templates(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        for source, directory in (("builtin", self.builtin_dir), ("user", self.user_dir)):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.template.json")):
                templates.append(self._read_template(path, source=source).metadata())
        return templates

    def load_template(self, template_id: str) -> AgentTemplate:
        safe_id = _validate_template_id(template_id)
        for source, directory in (("user", self.user_dir), ("builtin", self.builtin_dir)):
            path = directory / f"{safe_id}.template.json"
            if path.exists():
                return self._read_template(path, source=source)
        raise FileNotFoundError(f"agent template not found: {safe_id}")

    def save_template(
        self,
        *,
        template_id: str,
        label: str,
        description: str,
        agents: AgentsSpec,
        workflow: WorkflowSpec,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        safe_id = _validate_template_id(template_id)
        if (self.builtin_dir / f"{safe_id}.template.json").exists():
            raise ValueError(f"template id conflicts with builtin template: {safe_id}")
        _ensure_inside_workspace(self.workspace_root, self.user_dir)
        workflow_agents = {node.agent for node in workflow.nodes if node.agent}
        workflow_validation = workflow.validate()
        agents_validation = agents.validate(workflow_agent_ids=workflow_agents)
        if not workflow_validation.ok:
            raise ValueError(f"workflow spec is invalid: {'; '.join(workflow_validation.errors + workflow_validation.requires_code)}")
        if not agents_validation.ok:
            raise ValueError(f"agents spec is invalid: {'; '.join(agents_validation.errors)}")

        template = AgentTemplate(
            id=safe_id,
            label=label or safe_id,
            description=description,
            version=AGENT_TEMPLATE_VERSION,
            agents=agents,
            workflow=workflow,
            tags=list(tags or []),
            source="user",
            created_at=utc_now(),
        )
        self.user_dir.mkdir(parents=True, exist_ok=True)
        path = self.user_dir / f"{safe_id}.template.json"
        _write_json(path, template.to_dict())
        return self._read_template(path, source="user").metadata()

    def apply_template_to_drafts(
        self,
        template_id: str,
        *,
        agents_store: AgentsSpecStore,
        workflow_store: WorkflowSpecStore,
    ) -> dict[str, Any]:
        template = self.load_template(template_id)
        workflow_agents = {node.agent for node in template.workflow.nodes if node.agent}
        workflow_validation = template.workflow.validate(base=workflow_store.load())
        agents_validation = template.agents.validate(base=agents_store.load(), workflow_agent_ids=workflow_agents)
        if not workflow_validation.ok:
            reasons = workflow_validation.errors + workflow_validation.requires_code
            raise ValueError(f"template workflow cannot be applied: {'; '.join(reasons)}")
        if not agents_validation.ok:
            raise ValueError(f"template agents cannot be applied: {'; '.join(agents_validation.errors)}")
        workflow_payload = workflow_store.save_draft(template.workflow)
        agents_payload = agents_store.save_draft(template.agents, workflow_agent_ids=workflow_agents)
        return {
            "status": "drafted",
            "template": template.metadata(),
            "agents": agents_payload,
            "workflow": workflow_payload,
        }

    def _read_template(self, path: Path, *, source: str) -> AgentTemplate:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentTemplate.from_dict(data, source=source, path=path)


def _validate_template_id(value: str) -> str:
    template_id = str(value or "").strip()
    if not _TEMPLATE_ID.match(template_id):
        raise ValueError(f"invalid template id: {value}")
    return template_id


def _resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def _ensure_inside_workspace(workspace_root: Path, path: Path) -> None:
    root = workspace_root.resolve()
    target = path.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"template user directory must be inside workspace: {path}") from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
