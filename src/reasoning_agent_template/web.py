from __future__ import annotations

import json
import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from reasoning_agent_template.agent_templates import AgentTemplateStore
from reasoning_agent_template.agents_spec import AgentsSpec, AgentsSpecStore
from reasoning_agent_template.code_modifier import CodeModifierAdapter, LocalWorkflowSpecCodeModifier
from reasoning_agent_template.config import load_agent_config
from reasoning_agent_template.knowledge import LocalKnowledgeBase
from reasoning_agent_template.llm import ChatMessage, DeepSeekChatClient, LLMRequestError, MissingApiKeyError
from reasoning_agent_template.multiagent import ChatClient, MultiAgentOrchestrator
from reasoning_agent_template.skills import SkillRegistry
from reasoning_agent_template.workflow_spec import WorkflowSpec, WorkflowSpecStore


STATIC_ROOT = Path(__file__).parent / "web_static"


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: str | Path = "agent.yaml",
    workspace_root: str | Path = ".",
    llm_client_factory: Callable[[Any], ChatClient] | None = None,
    code_modifier_adapter: CodeModifierAdapter | None = None,
) -> ThreadingHTTPServer:
    workspace = Path(workspace_root)
    config = load_agent_config(Path(config_path))
    orchestrator = MultiAgentOrchestrator(
        config=config,
        workspace_root=workspace,
        llm_client_factory=llm_client_factory,
    )
    handler = _make_handler(
        orchestrator=orchestrator,
        workspace=workspace,
        config_path=Path(config_path),
        code_modifier_adapter=code_modifier_adapter,
        llm_client_factory=llm_client_factory,
    )
    return ThreadingHTTPServer((host, port), handler)


def _make_handler(
    *,
    orchestrator: MultiAgentOrchestrator,
    workspace: Path,
    config_path: Path,
    code_modifier_adapter: CodeModifierAdapter | None = None,
    llm_client_factory: Callable[[Any], ChatClient] | None = None,
) -> type[BaseHTTPRequestHandler]:
    workflow_store = WorkflowSpecStore(workspace, orchestrator.config.runtime)
    agents_store = AgentsSpecStore(workspace, orchestrator.config.runtime)
    template_store = AgentTemplateStore(workspace, orchestrator.config.runtime)
    code_modifier = code_modifier_adapter or LocalWorkflowSpecCodeModifier(workspace)

    class DebugHandler(BaseHTTPRequestHandler):
        server_version = "ReasoningAgentDebug/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_static("index.html")
                return
            if path == "/api/status":
                self._send_json(orchestrator.status())
                return
            if path == "/api/workflow":
                self._send_json(orchestrator.workflow_status())
                return
            if path == "/api/workflow/spec":
                self._send_json(_workflow_spec_payload(workflow_store))
                return
            if path == "/api/agents/spec":
                self._send_json(_agents_spec_payload(agents_store, workflow_store))
                return
            if path == "/api/templates":
                self._send_json(_template_list_payload(template_store))
                return
            if path == "/api/skills":
                skills = SkillRegistry(workspace / "skills").load()
                self._send_json(
                    {
                        "skills": [
                            {"name": skill.name, "description": skill.description, "path": str(skill.path)}
                            for skill in skills.values()
                        ]
                    }
                )
                return
            if path.startswith("/assets/"):
                self._send_static(path.removeprefix("/assets/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/workflow/draft":
                try:
                    body = self._read_json()
                    spec = WorkflowSpec.from_dict(dict(body.get("spec") or body))
                    self._send_json(workflow_store.save_draft(spec))
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/workflow/proposal":
                try:
                    body = self._read_json()
                    spec = WorkflowSpec.from_dict(dict(body["spec"])) if body.get("spec") else None
                    if spec is not None:
                        workflow_store.save_draft(spec)
                    self._send_json(workflow_store.create_proposal(spec))
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/workflow/apply":
                try:
                    body = self._read_json()
                    proposal_id = str(body.get("proposal_id", "")).strip()
                    if not proposal_id:
                        self._send_json({"error": "proposal_id is required"}, status=400)
                        return
                    if not bool(body.get("approved")):
                        self._send_json({"error": "approved=true is required"}, status=403)
                        return
                    proposal = workflow_store.load_proposal(proposal_id)
                    result = code_modifier.apply_workflow_proposal(
                        proposal,
                        approved_by=str(body.get("approved_by") or "local-user"),
                    )
                    payload = result.to_dict()
                    payload["workflow"] = _workflow_spec_payload(workflow_store)
                    self._send_json(payload, status=200 if result.status == "applied" else 409)
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path == "/api/agents/draft":
                try:
                    body = self._read_json()
                    spec = AgentsSpec.from_dict(dict(body.get("spec") or body))
                    self._send_json(
                        agents_store.save_draft(spec, workflow_agent_ids=_workflow_agent_ids(workflow_store))
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/agents/proposal":
                try:
                    body = self._read_json()
                    spec = AgentsSpec.from_dict(dict(body["spec"])) if body.get("spec") else None
                    if spec is not None:
                        agents_store.save_draft(spec, workflow_agent_ids=_workflow_agent_ids(workflow_store))
                    self._send_json(
                        agents_store.create_proposal(spec, workflow_agent_ids=_workflow_agent_ids(workflow_store))
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/agents/apply":
                try:
                    body = self._read_json()
                    proposal_id = str(body.get("proposal_id", "")).strip()
                    if not proposal_id:
                        self._send_json({"error": "proposal_id is required"}, status=400)
                        return
                    if not bool(body.get("approved")):
                        self._send_json({"error": "approved=true is required"}, status=403)
                        return
                    proposal = agents_store.load_proposal(proposal_id)
                    result = code_modifier.apply_agents_proposal(
                        proposal,
                        approved_by=str(body.get("approved_by") or "local-user"),
                    )
                    payload = result.to_dict()
                    payload["agents"] = _agents_spec_payload(agents_store, workflow_store)
                    self._send_json(payload, status=200 if result.status == "applied" else 409)
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path == "/api/templates/apply":
                try:
                    body = self._read_json()
                    template_id = str(body.get("template_id") or body.get("id") or "").strip()
                    if not template_id:
                        self._send_json({"error": "template_id is required"}, status=400)
                        return
                    self._send_json(
                        template_store.apply_template_to_drafts(
                            template_id,
                            agents_store=agents_store,
                            workflow_store=workflow_store,
                        )
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/templates/save":
                try:
                    body = self._read_json()
                    template_id = str(body.get("template_id") or body.get("id") or "").strip()
                    if not template_id:
                        self._send_json({"error": "template_id is required"}, status=400)
                        return
                    workflow = (
                        WorkflowSpec.from_dict(dict(body["workflow"]))
                        if body.get("workflow")
                        else workflow_store.load_draft() or workflow_store.load()
                    )
                    agents = (
                        AgentsSpec.from_dict(dict(body["agents"]))
                        if body.get("agents")
                        else agents_store.load_draft() or agents_store.load()
                    )
                    template = template_store.save_template(
                        template_id=template_id,
                        label=str(body.get("label") or template_id),
                        description=str(body.get("description") or ""),
                        tags=[str(item) for item in body.get("tags", [])] if isinstance(body.get("tags"), list) else [],
                        agents=agents,
                        workflow=workflow,
                    )
                    self._send_json({"status": "saved", "template": template})
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=400)
                return
            if path == "/api/configurator/agents":
                try:
                    body = self._read_json()
                    prompt = str(body.get("prompt", "")).strip()
                    if not prompt:
                        self._send_json({"error": "prompt is required"}, status=400)
                        return
                    result = _configurator_agents_payload(
                        prompt=prompt,
                        agents_store=agents_store,
                        workflow_store=workflow_store,
                        orchestrator=orchestrator,
                        llm_client_factory=llm_client_factory,
                    )
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path == "/api/configurator/workflow":
                try:
                    body = self._read_json()
                    prompt = str(body.get("prompt", "")).strip()
                    if not prompt:
                        self._send_json({"error": "prompt is required"}, status=400)
                        return
                    result = _configurator_workflow_payload(
                        prompt=prompt,
                        agents_store=agents_store,
                        workflow_store=workflow_store,
                        orchestrator=orchestrator,
                        llm_client_factory=llm_client_factory,
                    )
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path == "/api/configurator/compose":
                try:
                    body = self._read_json()
                    prompt = str(body.get("prompt", "")).strip()
                    target = str(body.get("target", "auto")).strip() or "auto"
                    if not prompt:
                        self._send_json({"error": "prompt is required"}, status=400)
                        return
                    result = _configurator_compose_payload(
                        prompt=prompt,
                        target=target,
                        agents_store=agents_store,
                        workflow_store=workflow_store,
                        orchestrator=orchestrator,
                        llm_client_factory=llm_client_factory,
                    )
                    self._send_json(result)
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path == "/api/rag/query":
                try:
                    body = self._read_json()
                    query = str(body.get("query", "")).strip()
                    if not query:
                        self._send_json({"error": "query is required"}, status=400)
                        return
                    self._send_json(
                        _rag_query_payload(
                            query=query,
                            config=orchestrator.config.knowledge,
                            workspace=workspace,
                            methods=body.get("methods"),
                            top_k=body.get("top_k"),
                            min_score=body.get("min_score"),
                            wiki_top_k=body.get("wiki_top_k"),
                        )
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc), "type": type(exc).__name__}, status=500)
                return
            if path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                body = self._read_json()
                message = str(body.get("message", "")).strip()
                thread_id = str(body.get("thread_id", "default")).strip() or "default"
                if not message:
                    self._send_json({"error": "message is required"}, status=400)
                    return
                if bool(body.get("async")):
                    worker = threading.Thread(
                        target=self._run_chat_background,
                        args=(message, thread_id),
                        daemon=True,
                    )
                    worker.start()
                    self._send_json({"status": "accepted", "thread_id": thread_id})
                    return
                self._send_json(orchestrator.run(message, thread_id=thread_id))
            except (LLMRequestError, MissingApiKeyError) as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "type": type(exc).__name__,
                        "phase": "llm",
                        "status": "failed",
                    },
                    status=502,
                )
            except Exception as exc:
                self._send_json(
                    {"error": str(exc), "type": type(exc).__name__, "phase": "server", "status": "failed"},
                    status=500,
                )

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _run_chat_background(self, message: str, thread_id: str) -> None:
            try:
                orchestrator.run(message, thread_id=thread_id)
            except (LLMRequestError, MissingApiKeyError) as exc:
                orchestrator.record_failure(message=message, thread_id=thread_id, error=exc, phase="llm")
            except Exception as exc:
                orchestrator.record_failure(message=message, thread_id=thread_id, error=exc, phase="server")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length).decode("utf-8")
            return json.loads(data) if data else {}

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, relative: str) -> None:
            target = (STATIC_ROOT / relative).resolve()
            try:
                target.relative_to(STATIC_ROOT.resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            data = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {
                "application/javascript",
                "application/json",
            }:
                content_type = f"{content_type}; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DebugHandler


def _template_list_payload(store: AgentTemplateStore) -> dict[str, Any]:
    return {
        "templates": store.list_templates(),
        "paths": {
            "builtin": str(store.builtin_dir),
            "user": str(store.user_dir),
        },
    }


def _workflow_spec_payload(store: WorkflowSpecStore) -> dict[str, Any]:
    spec = store.load()
    draft = store.load_draft()
    validation_target = draft or spec
    validation = validation_target.validate(base=spec if draft else None)
    return {
        "spec": spec.to_dict(),
        "draft": draft.to_dict() if draft else None,
        "validation": validation.to_dict(),
        "paths": {
            "spec": str(store.spec_path),
            "draft": str(store.draft_path),
            "proposal_dir": str(store.proposal_dir),
        },
    }


def _agents_spec_payload(store: AgentsSpecStore, workflow_store: WorkflowSpecStore) -> dict[str, Any]:
    spec = store.load()
    draft = store.load_draft()
    validation_target = draft or spec
    validation = validation_target.validate(
        base=spec if draft else None,
        workflow_agent_ids=_workflow_agent_ids(workflow_store),
    )
    return {
        "spec": spec.to_dict(),
        "draft": draft.to_dict() if draft else None,
        "validation": validation.to_dict(),
        "paths": {
            "spec": str(store.spec_path),
            "draft": str(store.draft_path),
            "proposal_dir": str(store.proposal_dir),
        },
    }


def _workflow_agent_ids(workflow_store: WorkflowSpecStore) -> set[str]:
    return {node.agent for node in workflow_store.load().nodes if node.agent}


def _rag_query_payload(
    *,
    query: str,
    config: dict[str, Any],
    workspace: Path,
    methods: Any = None,
    top_k: Any = None,
    min_score: Any = None,
    wiki_top_k: Any = None,
) -> dict[str, Any]:
    knowledge_dir = Path(str(config.get("directory", "knowledge")))
    if not knowledge_dir.is_absolute():
        knowledge_dir = workspace / knowledge_dir
    selected_methods = _as_string_list(methods) or _as_string_list(config.get("retrieval_methods")) or [
        str(config.get("index_type", "keyword"))
    ]
    kb = LocalKnowledgeBase(
        knowledge_dir,
        max_chunk_chars=int(config.get("chunk_size", 1400)),
    )
    results = kb.retrieve(
        query,
        top_k=int(top_k or config.get("top_k", 5)),
        methods=selected_methods,
        min_score=float(config.get("min_score", 0.0) if min_score is None else min_score),
        wiki_top_k=int(wiki_top_k or config.get("wiki_top_k", 2)),
    )
    return {
        "query": query,
        "methods": selected_methods,
        "count": len(results),
        "diagnostics": list(kb.diagnostics),
        "results": [_chunk_payload(chunk) for chunk in results],
    }


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    return {
        "source": chunk.source,
        "span": chunk.span,
        "score": chunk.score,
        "content_hash": chunk.content_hash,
        "evidence_id": chunk.evidence_id,
        "text": chunk.text,
        "retrieval_method": getattr(chunk, "retrieval_method", "keyword"),
        "score_breakdown": dict(getattr(chunk, "score_breakdown", {}) or {}),
        "metadata": dict(getattr(chunk, "metadata", {}) or {}),
    }


def _configurator_agents_payload(
    *,
    prompt: str,
    agents_store: AgentsSpecStore,
    workflow_store: WorkflowSpecStore,
    orchestrator: MultiAgentOrchestrator,
    llm_client_factory: Callable[[Any], ChatClient] | None,
    builder_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = builder_plan or _builder_plan(prompt)
    base = agents_store.load_draft() or agents_store.load()
    workflow = workflow_store.load()
    source = "deepseek"
    message = "configurator generated agents draft with DeepSeek"
    try:
        spec = _deepseek_configurator_agents(
            prompt=prompt,
            base=base,
            workflow=workflow,
            orchestrator=orchestrator,
            llm_client_factory=llm_client_factory,
            builder_plan=plan,
        )
    except (LLMRequestError, MissingApiKeyError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        source = "local_fallback"
        message = f"DeepSeek configurator unavailable, local configurator generated a draft: {exc}"
        spec = _local_configurator_agents(prompt=prompt, base=base, workflow=workflow, builder_plan=plan)
    saved = agents_store.save_draft(spec, workflow_agent_ids=_workflow_agent_ids(workflow_store))
    return {
        "source": source,
        "message": message,
        "prompt": prompt,
        "spec": saved["spec"],
        "validation": saved["validation"],
        "summary": _agents_draft_summary(saved["spec"]),
        "builder_plan": plan,
        "paths": {
            "draft": saved["path"],
            "spec": str(agents_store.spec_path),
            "proposal_dir": str(agents_store.proposal_dir),
        },
    }


def _configurator_compose_payload(
    *,
    prompt: str,
    target: str,
    agents_store: AgentsSpecStore,
    workflow_store: WorkflowSpecStore,
    orchestrator: MultiAgentOrchestrator,
    llm_client_factory: Callable[[Any], ChatClient] | None,
) -> dict[str, Any]:
    clean_prompt = _strip_configurator_trigger(prompt)
    selected = _configurator_targets(clean_prompt, target)
    builder_plan = _builder_plan(clean_prompt)
    payload: dict[str, Any] = {
        "status": "completed",
        "agent": "configurator",
        "prompt": clean_prompt,
        "targets": selected,
        "builder_plan": builder_plan,
        "agents": None,
        "workflow": None,
    }
    lines = ["配置助手已生成草稿，尚未应用。"]
    lines.append(
        f"规模判断: {builder_plan['scale']}，推荐设计层 {builder_plan['agent_count']} 个 Agent / "
        f"{builder_plan['node_count']} 个节点；依据: {'、'.join(builder_plan['reasons'])}。"
    )
    if "agents" in selected:
        agents_result = _configurator_agents_payload(
            prompt=clean_prompt,
            agents_store=agents_store,
            workflow_store=workflow_store,
            orchestrator=orchestrator,
            llm_client_factory=llm_client_factory,
            builder_plan=builder_plan,
        )
        payload["agents"] = agents_result
        summary = agents_result["summary"]
        lines.append(
            f"- Agent 草稿: 设计层 {summary['design_count']} 个 / 底座 {summary['protected_count']} 个 / 总计 {summary['total_count']} 个, "
            f"校验 {'通过' if agents_result['validation'].get('ok') else '未通过'}, 来源 {agents_result['source']}。"
        )
    if "workflow" in selected:
        workflow_result = _configurator_workflow_payload(
            prompt=clean_prompt,
            agents_store=agents_store,
            workflow_store=workflow_store,
            orchestrator=orchestrator,
            llm_client_factory=llm_client_factory,
            builder_plan=builder_plan,
        )
        payload["workflow"] = workflow_result
        summary = workflow_result["summary"]
        lines.append(
            f"- 工作流草稿: 设计层 {summary['design_count']} 个节点 / 底座 {summary['protected_count']} 个节点 / 总计 {summary['total_count']} 个节点, "
            f"校验 {'通过' if workflow_result['validation'].get('ok') else '未通过'}, 来源 {workflow_result['source']}。"
        )
    lines.append("说明：底座 Agent/节点用于保持运行时、证据和门禁可用；右侧编辑视图默认聚焦你要求的设计层。")
    lines.append("下一步：到多 Agent 或工作流面板做人工检查，然后保存草稿、生成提案、批准应用。")
    payload["answer"] = "\n".join(lines)
    return payload


def _strip_configurator_trigger(prompt: str) -> str:
    text = prompt.strip()
    triggers = ["/配置", "/搭建", "#配置", "#搭建", "配置助手:", "配置助手："]
    for trigger in triggers:
        if text.startswith(trigger):
            return text[len(trigger) :].strip() or text
    return text


def _configurator_targets(prompt: str, target: str) -> list[str]:
    normalized_target = target.lower()
    if normalized_target in {"agents", "agent", "multi-agent", "multi_agent"}:
        return ["agents"]
    if normalized_target in {"workflow", "workflows"}:
        return ["workflow"]
    text = prompt.lower()
    agents_only = any(
        term in text
        for term in [
            "只改 agent",
            "只改agent",
            "只生成 agent",
            "只生成agent",
            "仅 agent",
            "仅agent",
            "agent 模块",
            "agent模块",
            "多 agent 模块",
            "多agent模块",
        ]
    )
    workflow_only = any(
        term in text
        for term in [
            "只改工作流",
            "只生成工作流",
            "仅工作流",
            "工作流模块",
            "workflow only",
            "only workflow",
        ]
    )
    if agents_only and not workflow_only:
        return ["agents"]
    if workflow_only and not agents_only:
        return ["workflow"]
    wants_workflow = any(
        term in text
        for term in [
            "workflow",
            "handoff",
            "gate",
            "reviewer",
            "工作流",
            "流程",
            "节点",
            "状态机",
            "连线",
            "门禁",
            "环节",
            "阶段",
            "步骤",
            "插入",
            "加入",
            "推理后",
            "推理前",
            "后加入",
            "前加入",
            "转交",
            "审查",
            "复核",
        ]
    )
    wants_agents = any(term in text for term in ["agent", "多 agent", "多agent", "角色", "工具", "权限", "职责", "团队"])
    if wants_agents and not wants_workflow:
        return ["agents", "workflow"]
    if wants_workflow and not wants_agents:
        return ["agents", "workflow"]
    return ["agents", "workflow"]


def _requested_agent_count(prompt: str) -> int | None:
    return _requested_count(prompt, ["agent", "agents", "智能体", "角色", "代理"])


def _requested_node_count(prompt: str) -> int | None:
    return _requested_count(prompt, ["node", "nodes", "节点", "环节", "步骤"])


def _requested_count(prompt: str, nouns: list[str]) -> int | None:
    noun_pattern = "|".join(re.escape(noun) for noun in nouns)
    text = prompt.lower()
    patterns = [
        rf"([0-9一二两三四五六七八九十]+)\s*[-~到至]\s*([0-9一二两三四五六七八九十]+)\s*个?\s*(?:{noun_pattern})",
        rf"([0-9一二两三四五六七八九十]+)\s*个?\s*(?:{noun_pattern})",
        rf"(?:{noun_pattern})\s*[:：]?\s*([0-9一二两三四五六七八九十]+)\s*[-~到至]\s*([0-9一二两三四五六七八九十]+)",
        rf"(?:{noun_pattern})\s*[:：]?\s*([0-9一二两三四五六七八九十]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        values = [_parse_count_token(value) for value in match.groups() if value]
        values = [value for value in values if value is not None]
        if values:
            return max(1, min(max(values), 12))
    return None


def _parse_count_token(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    numerals = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in numerals:
        return numerals[value]
    if value.startswith("十") and len(value) == 2 and value[1] in numerals:
        return 10 + numerals[value[1]]
    if value.endswith("十") and len(value) == 2 and value[0] in numerals:
        return numerals[value[0]] * 10
    if "十" in value:
        left, right = value.split("十", 1)
        left_value = numerals.get(left, 1) if left else 1
        right_value = numerals.get(right, 0) if right else 0
        return left_value * 10 + right_value
    return None


def _builder_plan(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    requested_agents = _requested_agent_count(prompt)
    requested_nodes = _requested_node_count(prompt)
    scale, reasons = _infer_builder_scale(text)
    defaults = {
        "simple": (1, 2),
        "medium": (2, 3),
        "complex": (4, 5),
        "enterprise": (5, 7),
    }
    default_agents, default_nodes = defaults[scale]
    agent_count = requested_agents if requested_agents is not None else default_agents
    node_count = requested_nodes if requested_nodes is not None else default_nodes
    if requested_agents is not None and requested_nodes is None:
        node_count = max(default_nodes, min(requested_agents + 1, 8))
    if requested_nodes is not None and requested_agents is None:
        agent_count = max(default_agents, min(max(1, requested_nodes - 1), 6))
    agent_count = max(1, min(agent_count, 6))
    node_count = max(2, min(node_count, 8))
    source = "explicit+inferred" if requested_agents is not None or requested_nodes is not None else "inferred"
    return {
        "scale": scale,
        "agent_count": agent_count,
        "node_count": node_count,
        "explicit_agent_count": requested_agents is not None,
        "explicit_node_count": requested_nodes is not None,
        "source": source,
        "reasons": reasons,
    }


def _infer_builder_scale(text: str) -> tuple[str, list[str]]:
    simple_score = _signal_count(
        text,
        [
            ["轻量", "简单", "小型", "最小", "单轮", "只回答", "faq", "常见问题", "客服"],
            ["翻译", "总结", "改写", "分类", "解释"],
            ["不需要工具", "不用检索", "无需证据", "单 agent", "单agent"],
        ],
    )
    medium_score = _signal_count(
        text,
        [
            ["知识库", "rag", "检索", "文档", "资料"],
            ["reviewer", "审查", "复核", "质量"],
            ["工具", "api", "调用", "自动化"],
            ["领域", "专业", "分析", "报告"],
        ],
    )
    complex_score = _signal_count(
        text,
        [
            ["多 agent", "多agent", "multi-agent", "multi agent", "团队", "角色分工"],
            ["工作流", "workflow", "节点", "状态机", "编排", "连线", "handoff"],
            ["证据", "门禁", "gate", "权限", "审批", "审计"],
            ["长期记忆", "短期记忆", "记忆沉淀", "知识沉淀"],
            ["学术", "论文", "研究", "科研", "材料", "合金", "paper", "research"],
            ["自进化", "自我进化", "proposal", "提案", "代码修改"],
        ],
    )
    enterprise_score = _signal_count(
        text,
        [
            ["平台", "通用模板", "生产", "企业", "多租户", "部署"],
            ["治理", "合规", "审计", "权限体系", "安全边界"],
            ["插件", "可插拔", "mcp", "openapi", "适配器"],
            ["动态工作流", "代码修改 agent", "自动优化", "自进化"],
        ],
    )
    if enterprise_score >= 2:
        return "enterprise", _builder_scale_reasons("enterprise", enterprise_score, complex_score, medium_score, simple_score)
    if complex_score >= 2 or (complex_score >= 1 and medium_score >= 2):
        return "complex", _builder_scale_reasons("complex", enterprise_score, complex_score, medium_score, simple_score)
    if medium_score >= 2 or (medium_score >= 1 and simple_score == 0):
        return "medium", _builder_scale_reasons("medium", enterprise_score, complex_score, medium_score, simple_score)
    return "simple", _builder_scale_reasons("simple", enterprise_score, complex_score, medium_score, simple_score)


def _signal_count(text: str, groups: list[list[str]]) -> int:
    return sum(1 for group in groups if any(term in text for term in group))


def _builder_scale_reasons(
    scale: str,
    enterprise_score: int,
    complex_score: int,
    medium_score: int,
    simple_score: int,
) -> list[str]:
    reasons = [f"{scale} 规模"]
    if simple_score:
        reasons.append(f"轻量信号 {simple_score}")
    if medium_score:
        reasons.append(f"中等复杂信号 {medium_score}")
    if complex_score:
        reasons.append(f"复杂信号 {complex_score}")
    if enterprise_score:
        reasons.append(f"平台级信号 {enterprise_score}")
    if len(reasons) == 1:
        reasons.append("未发现需要重型编排的要求")
    return reasons


def _configurator_workflow_payload(
    *,
    prompt: str,
    agents_store: AgentsSpecStore,
    workflow_store: WorkflowSpecStore,
    orchestrator: MultiAgentOrchestrator,
    llm_client_factory: Callable[[Any], ChatClient] | None,
    builder_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = builder_plan or _builder_plan(prompt)
    base = workflow_store.load_draft() or workflow_store.load()
    agents = agents_store.load_draft() or agents_store.load()
    source = "deepseek"
    message = "configurator generated workflow draft with DeepSeek"
    try:
        spec = _deepseek_configurator_workflow(
            prompt=prompt,
            base=base,
            agents=agents,
            orchestrator=orchestrator,
            llm_client_factory=llm_client_factory,
            builder_plan=plan,
        )
    except (LLMRequestError, MissingApiKeyError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        source = "local_fallback"
        message = f"DeepSeek configurator unavailable, local configurator generated a workflow draft: {exc}"
        spec = _local_configurator_workflow(prompt=prompt, base=base, agents=agents, builder_plan=plan)
    saved = workflow_store.save_draft(spec)
    return {
        "source": source,
        "message": message,
        "prompt": prompt,
        "spec": saved["spec"],
        "validation": saved["validation"],
        "summary": _workflow_draft_summary(saved["spec"]),
        "builder_plan": plan,
        "paths": {
            "draft": saved["path"],
            "spec": str(workflow_store.spec_path),
            "proposal_dir": str(workflow_store.proposal_dir),
        },
    }


def _deepseek_configurator_agents(
    *,
    prompt: str,
    base: AgentsSpec,
    workflow: WorkflowSpec,
    orchestrator: MultiAgentOrchestrator,
    llm_client_factory: Callable[[Any], ChatClient] | None,
    builder_plan: dict[str, Any] | None = None,
) -> AgentsSpec:
    client = llm_client_factory(orchestrator.config) if llm_client_factory else DeepSeekChatClient.from_config(
        orchestrator.config,
        role="planner",
    )
    response = client.chat(
        [
            ChatMessage(
                role="system",
                content=(
                    "你是 configurator Agent，只负责把用户的自然语言需求转换为 agents spec 草稿。"
                    "只输出 JSON，不要输出 markdown。必须保留所有 protected_agents。"
                    "不要直接应用修改，不要写文件。"
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "user_request": prompt,
                        "builder_plan": builder_plan or _builder_plan(prompt),
                        "current_agents_spec": base.to_dict(),
                        "workflow_nodes": [
                            {"id": node.id, "label": node.label, "agent": node.agent, "work": node.work}
                            for node in workflow.nodes
                        ],
                        "required_schema": {
                            "version": "1.0",
                            "name": "string",
                            "revision": "string",
                            "protected_agents": ["string"],
                            "agents": [
                                {
                                    "id": "snake_case",
                                    "label": "中文短名称",
                                    "description": "一句话职责",
                                    "responsibilities": ["职责"],
                                    "model_role": "planner|worker|critic|grader",
                                    "tools": ["tool_name"],
                                    "permissions": {},
                                    "memory_access": ["short_term:read"],
                                    "workflow_nodes": ["workflow_node_id"],
                                    "handoff_contract": {},
                                    "ui": {},
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        temperature=0.2,
        max_tokens=2200,
    )
    data = _extract_json_object(response.content)
    spec = AgentsSpec.from_dict(data)
    spec = AgentsSpec.from_dict({**spec.to_dict(), "protected_agents": list(base.protected_agents)})
    spec = _local_configurator_agents(prompt=prompt, base=spec, workflow=workflow, builder_plan=builder_plan)
    spec = AgentsSpec.from_dict({**spec.to_dict(), "protected_agents": list(base.protected_agents)})
    protected = set(base.protected_agents)
    candidate_ids = {agent.id for agent in spec.agents}
    if not protected.issubset(candidate_ids):
        missing = protected - candidate_ids
        original = base.agent_map()
        spec = AgentsSpec.from_dict(
            {
                **spec.to_dict(),
                "agents": [*spec.to_dict()["agents"], *[original[item].to_dict() for item in sorted(missing)]],
                "protected_agents": list(base.protected_agents),
            }
        )
    return spec


def _deepseek_configurator_workflow(
    *,
    prompt: str,
    base: WorkflowSpec,
    agents: AgentsSpec,
    orchestrator: MultiAgentOrchestrator,
    llm_client_factory: Callable[[Any], ChatClient] | None,
    builder_plan: dict[str, Any] | None = None,
) -> WorkflowSpec:
    client = llm_client_factory(orchestrator.config) if llm_client_factory else DeepSeekChatClient.from_config(
        orchestrator.config,
        role="planner",
    )
    response = client.chat(
        [
            ChatMessage(
                role="system",
                content=(
                    "你是 configurator Agent，只负责把用户需求转换为 workflow spec 草稿。"
                    "只输出 JSON，不要输出 markdown。必须保留所有 protected_nodes，未知 handler 使用 review_note 或 passthrough。"
                    "不要直接应用修改，不要写文件。"
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "user_request": prompt,
                        "builder_plan": builder_plan or _builder_plan(prompt),
                        "current_workflow_spec": base.to_dict(),
                        "available_agents": [
                            {"id": agent.id, "label": agent.label, "description": agent.description}
                            for agent in agents.agents
                        ],
                        "supported_builtin_handlers": ["passthrough", "review_note"],
                        "required_schema": {
                            "version": "1.0",
                            "name": "string",
                            "revision": "string",
                            "start_node": "node_id",
                            "terminal_nodes": ["node_id"],
                            "protected_nodes": ["node_id"],
                            "nodes": [
                                {
                                    "id": "snake_case",
                                    "label": "中文短名称",
                                    "agent": "agent_id",
                                    "description": "一句话描述",
                                    "work": "工作内容",
                                    "input_contract": "上游交付",
                                    "output_contract": "下游交付",
                                    "handler_kind": "builtin",
                                    "handler": "review_note",
                                    "checkpoint": False,
                                    "gate_policy": {},
                                    "ui": {},
                                }
                            ],
                            "edges": [
                                {
                                    "id": "edge_id",
                                    "from": "node_id",
                                    "to": "node_id",
                                    "type": "flow",
                                    "condition": "condition",
                                    "handoff_contract": {},
                                    "gate_policy": {},
                                    "planner_contract": {},
                                    "reviewer_required": False,
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        temperature=0.2,
        max_tokens=2600,
    )
    spec = WorkflowSpec.from_dict(_extract_json_object(response.content))
    missing = set(base.protected_nodes) - {node.id for node in spec.nodes}
    if missing:
        raise ValueError(f"DeepSeek workflow draft deleted protected nodes: {', '.join(sorted(missing))}")
    return _local_configurator_workflow(prompt=prompt, base=spec, agents=agents, builder_plan=builder_plan)


def _local_configurator_agents(
    *,
    prompt: str,
    base: AgentsSpec,
    workflow: WorkflowSpec,
    builder_plan: dict[str, Any] | None = None,
) -> AgentsSpec:
    data = base.to_dict()
    data["revision"] = "configurator-draft"
    agents = {agent["id"]: dict(agent) for agent in data.get("agents", [])}
    text = prompt.lower()
    plan = builder_plan or _builder_plan(prompt)
    design_count = int(plan["agent_count"])
    protected = set(data.get("protected_agents", []))

    agents = {agent_id: agent for agent_id, agent in agents.items() if agent_id in protected}
    for agent in agents.values():
        _set_builder_visibility(agent, visible=False)

    def ensure(agent: dict[str, Any]) -> None:
        existing = agents.get(agent["id"], {})
        merged = {**agent, **existing}
        merged["responsibilities"] = list(dict.fromkeys([*agent.get("responsibilities", []), *existing.get("responsibilities", [])]))
        merged["tools"] = list(dict.fromkeys([*agent.get("tools", []), *existing.get("tools", [])]))
        merged["workflow_nodes"] = list(dict.fromkeys([*agent.get("workflow_nodes", []), *existing.get("workflow_nodes", [])]))
        merged["ui"] = {**dict(agent.get("ui") or {}), **dict(existing.get("ui") or {})}
        agents[agent["id"]] = merged

    if design_count:
        for agent in _compact_agent_templates(prompt, design_count):
            ensure(agent)
    else:
        ensure(
            {
                "id": "configurator",
                "label": "配置助手",
                "description": "通过问答帮助用户生成 Agent、Workflow、技能和验收测试草稿。",
                "responsibilities": ["理解用户目标", "生成多 Agent 草稿", "推荐工作流节点", "生成验收测试"],
                "model_role": "planner",
                "tools": ["agents_draft", "workflow_draft", "config_schema"],
                "permissions": {"direct_apply": False},
                "memory_access": [],
                "workflow_nodes": [],
                "handoff_contract": {},
                "ui": {},
            }
        )
        if any(term in text for term in ["代码", "实现", "修改", "开发", "code", "repo"]):
            ensure(
                {
                    "id": "code_modifier",
                    "label": "代码修改器",
                    "description": "只负责应用已批准的代码或配置修改，不参与普通对话、检索、记忆或推理。",
                    "responsibilities": ["应用已批准提案", "限制修改路径", "运行验证命令"],
                    "model_role": "worker",
                    "tools": ["opencode:code-modifier"],
                    "permissions": {"direct_apply": False, "allowed_paths": ["src/", "tests/", "configs/"]},
                    "memory_access": [],
                    "workflow_nodes": [],
                    "handoff_contract": {},
                    "ui": {},
                }
            )
        if any(term in text for term in ["学术", "论文", "研究", "材料", "合金", "科研", "paper", "research"]):
            ensure(
                {
                    "id": "domain_researcher",
                    "label": "领域研究员",
                    "description": "面向专业领域问题检索论文、标准和高质量资料，形成证据候选。",
                    "responsibilities": ["形成检索式", "筛选论文证据", "标记证据质量"],
                    "model_role": "worker",
                    "tools": ["external_search", "retrieve_knowledge"],
                    "permissions": {},
                    "memory_access": ["long_term:read"],
                    "workflow_nodes": [],
                    "handoff_contract": {"output": "ranked evidence candidates"},
                    "ui": {},
                }
            )
        if any(term in text for term in ["工作流", "workflow", "节点", "编排", "多agent", "multi"]):
            ensure(
                {
                    "id": "workflow_architect",
                    "label": "工作流架构师",
                    "description": "根据任务类型设计节点、连线、门禁和 Agent 交付契约。",
                    "responsibilities": ["设计节点图", "定义连线契约", "检查可运行性"],
                    "model_role": "planner",
                    "tools": ["workflow_draft"],
                    "permissions": {"direct_apply": False},
                    "memory_access": [],
                    "workflow_nodes": [],
                    "handoff_contract": {"output": "workflow spec draft"},
                    "ui": {},
                }
            )

        workflow_agent_ids = {node.agent for node in workflow.nodes if node.agent}
        for agent_id in workflow_agent_ids:
            if agent_id not in agents:
                ensure(
                    {
                        "id": agent_id,
                        "label": agent_id,
                        "description": f"Workflow 中引用的 {agent_id} Agent。",
                        "responsibilities": ["执行 workflow 节点任务"],
                        "model_role": "worker",
                        "tools": [],
                        "permissions": {},
                        "memory_access": [],
                        "workflow_nodes": [node.id for node in workflow.nodes if node.agent == agent_id],
                        "handoff_contract": {},
                        "ui": {},
                    }
                )
    data["agents"] = list(agents.values())
    return AgentsSpec.from_dict(data)


def _local_configurator_workflow(
    *,
    prompt: str,
    base: WorkflowSpec,
    agents: AgentsSpec,
    builder_plan: dict[str, Any] | None = None,
) -> WorkflowSpec:
    data = base.to_dict()
    data["revision"] = "configurator-draft"
    nodes = list(data.get("nodes", []))
    edges = list(data.get("edges", []))
    agent_ids = {agent.id for agent in agents.agents}
    plan = builder_plan or _builder_plan(prompt)
    design_count = int(plan["node_count"])
    protected = set(data.get("protected_nodes", []))
    compact_nodes = _compact_workflow_nodes(prompt, design_count, agent_ids)
    nodes = _merge_compact_workflow_nodes(nodes, compact_nodes, protected)
    active_node_ids = {node["id"] for node in nodes}
    edges = [
        edge
        for edge in edges
        if edge.get("from") in active_node_ids and edge.get("to") in active_node_ids
    ]
    edges = [
        *edges,
        *_compact_workflow_edges(compact_nodes),
    ]

    data["nodes"] = nodes
    data["edges"] = _dedupe_edges(edges)
    return WorkflowSpec.from_dict(data)


def _compact_agent_templates(prompt: str, count: int) -> list[dict[str, Any]]:
    text = prompt.lower()
    research = any(term in text for term in ["学术", "论文", "研究", "材料", "合金", "科研", "paper", "research"])
    wants_workflow = any(term in text for term in ["工作流", "workflow", "节点", "编排", "多agent", "multi", "门禁", "交付"])
    templates: list[dict[str, Any]] = []
    if research:
        templates.append(
            _builder_agent(
                "domain_researcher",
                "领域研究员",
                "检索和筛选领域资料、论文与知识库证据。",
                ["形成检索式", "筛选领域证据", "标记证据质量"],
                "worker",
                ["external_search", "retrieve_knowledge"],
                ["project_research"],
                {"output": "ranked evidence candidates"},
            )
        )
    if wants_workflow:
        templates.append(
            _builder_agent(
                "workflow_architect",
                "工作流架构师",
                "设计节点、连线、门禁和交付契约。",
                ["设计节点图", "定义连线契约", "检查可运行性"],
                "planner",
                ["workflow_draft"],
                ["project_intake"],
                {"output": "workflow spec draft"},
            )
        )
    templates.extend(
        [
            _builder_agent(
                "solution_builder",
                "方案执行者",
                "根据计划完成核心推理、分析或执行交付。",
                ["执行核心任务", "产出可审查草案", "记录不确定性"],
                "worker",
                [],
                ["project_reason"],
                {"output": "answer or artifact draft"},
            ),
            _builder_agent(
                "quality_reviewer",
                "质量审查",
                "审查输出质量、门禁、证据绑定和交付完整性。",
                ["审查关键结论", "检查门禁条件", "提出修正意见"],
                "critic",
                [],
                ["project_review"],
                {"output": "review notes"},
            ),
            _builder_agent(
                "evidence_specialist",
                "证据专员",
                "专门管理证据收集、引用质量和证据缺口。",
                ["收集证据", "评估来源质量", "维护证据缺口"],
                "worker",
                ["retrieve_knowledge", "external_search"],
                ["project_research"],
                {"output": "evidence ledger candidates"},
            ),
            _builder_agent(
                "tool_operator",
                "工具执行",
                "按审批后的计划执行工具调用并记录结果。",
                ["执行工具", "记录工具输出", "报告失败状态"],
                "worker",
                [],
                ["project_action"],
                {"output": "tool results"},
            ),
        ]
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template in templates:
        if template["id"] in seen:
            continue
        seen.add(template["id"])
        deduped.append(template)
    while len(deduped) < count:
        index = len(deduped) + 1
        deduped.append(
            _builder_agent(
                f"custom_agent_{index}",
                f"自定义 Agent {index}",
                "按用户场景补充的专用 Agent。",
                ["补充专用职责", "形成可审查交付"],
                "worker",
                [],
                [],
                {},
            )
        )
    return deduped[:count]


def _builder_agent(
    agent_id: str,
    label: str,
    description: str,
    responsibilities: list[str],
    model_role: str,
    tools: list[str],
    workflow_nodes: list[str],
    handoff_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": agent_id,
        "label": label,
        "description": description,
        "responsibilities": responsibilities,
        "model_role": model_role,
        "tools": tools,
        "permissions": {"direct_apply": False},
        "memory_access": ["short_term:read"],
        "workflow_nodes": workflow_nodes,
        "handoff_contract": handoff_contract,
        "ui": {"builder_visible": True, "builder_generated": True},
    }


def _compact_workflow_nodes(prompt: str, count: int, agent_ids: set[str]) -> list[dict[str, Any]]:
    text = prompt.lower()
    research_agent = _first_available(agent_ids, ["domain_researcher", "evidence_specialist", "retriever"], "retriever")
    architect_agent = _first_available(agent_ids, ["workflow_architect", "planner", "coordinator"], "planner")
    builder_agent = _first_available(agent_ids, ["solution_builder", "reasoner"], "reasoner")
    reviewer_agent = _first_available(agent_ids, ["quality_reviewer", "critic", "reviewer"], "critic")
    wants_research = any(term in text for term in ["学术", "论文", "研究", "材料", "合金", "科研", "paper", "research", "证据"])
    review_node_id = "domain_review" if wants_research else "project_review"
    review_node_label = "领域审查" if wants_research else "质量审查"
    review_node_work = (
        "审查领域事实、论文证据、证据缺口和输出质量。"
        if wants_research
        else "审查证据、门禁、交付契约和输出质量。"
    )
    base_templates = [
        _builder_node(
            "project_intake",
            "目标建模",
            architect_agent,
            "把用户目标转成项目任务、边界和交付约束。",
            "用户目标",
            "项目任务包",
            "passthrough",
        ),
        _builder_node(
            "project_research",
            "资料检索" if wants_research else "上下文准备",
            research_agent,
            "收集领域资料、知识库内容和必要证据。",
            "项目任务包",
            "上下文与证据候选",
            "passthrough",
            checkpoint=True,
        ),
        _builder_node(
            "project_reason",
            "方案推理",
            builder_agent,
            "生成核心分析、方案草案或任务结果。",
            "上下文与证据候选",
            "方案草案",
            "passthrough",
        ),
        _builder_node(
            review_node_id,
            review_node_label,
            reviewer_agent,
            review_node_work,
            "方案草案",
            "领域审查记录" if wants_research else "审查记录",
            "review_note",
            checkpoint=True,
            gate_policy={"reviewer_required": True, "requires_evidence": wants_research},
        ),
    ]
    if count <= 3:
        templates = base_templates[1:4]
    else:
        templates = list(base_templates)
    while len(templates) < count:
        index = len(templates) + 1
        templates.append(
            _builder_node(
                f"project_step_{index}",
                f"扩展步骤 {index}",
                builder_agent,
                "按用户需求补充的自定义工作步骤。",
                "上游交付",
                "下游交付",
                "passthrough",
            )
        )
    return templates[:count]


def _builder_node(
    node_id: str,
    label: str,
    agent: str,
    work: str,
    input_contract: str,
    output_contract: str,
    handler: str,
    *,
    checkpoint: bool = False,
    gate_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "agent": agent,
        "description": work,
        "work": work,
        "input_contract": input_contract,
        "output_contract": output_contract,
        "handler_kind": "builtin",
        "handler": handler,
        "checkpoint": checkpoint,
        "gate_policy": dict(gate_policy or {}),
        "ui": {"builder_visible": True, "builder_generated": True},
    }


def _merge_compact_workflow_nodes(
    existing_nodes: list[dict[str, Any]],
    compact_nodes: list[dict[str, Any]],
    protected: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    inserted = False
    for node in existing_nodes:
        if node.get("id") not in protected:
            continue
        node = dict(node)
        _set_builder_visibility(node, visible=False)
        result.append(node)
        if node.get("id") == "plan":
            result.extend(compact_nodes)
            inserted = True
    if not inserted:
        result.extend(compact_nodes)
    return result


def _compact_workflow_edges(compact_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not compact_nodes:
        return []
    edges: list[dict[str, Any]] = [
        _builder_edge("plan", compact_nodes[0]["id"], "进入用户设计层", reviewer_required=True)
    ]
    for previous, current in zip(compact_nodes, compact_nodes[1:]):
        edges.append(_builder_edge(previous["id"], current["id"], "设计层转交"))
    edges.append(_builder_edge(compact_nodes[-1]["id"], "evidence_audit", "回到底座证据/门禁链路", reviewer_required=True))
    return edges


def _builder_edge(from_node: str, to_node: str, condition: str, *, reviewer_required: bool = False) -> dict[str, Any]:
    return {
        "id": f"{from_node}_to_{to_node}",
        "from": from_node,
        "to": to_node,
        "type": "flow",
        "condition": condition,
        "handoff_contract": {"payload": condition},
        "gate_policy": {},
        "planner_contract": {},
        "reviewer_required": reviewer_required,
    }


def _first_available(agent_ids: set[str], candidates: list[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate in agent_ids:
            return candidate
    return fallback


def _set_builder_visibility(item: dict[str, Any], *, visible: bool) -> None:
    ui = dict(item.get("ui") or {})
    if visible:
        ui["builder_visible"] = True
        ui.pop("builder_hidden", None)
    else:
        ui["builder_hidden"] = True
        ui.setdefault("system_scaffold", True)
    item["ui"] = ui


def _agents_draft_summary(spec: dict[str, Any]) -> dict[str, int]:
    protected = set(spec.get("protected_agents", []))
    agents = spec.get("agents", [])
    design = [
        agent
        for agent in agents
        if agent.get("id") not in protected and not dict(agent.get("ui") or {}).get("builder_hidden")
    ]
    return {
        "total_count": len(agents),
        "protected_count": len([agent for agent in agents if agent.get("id") in protected]),
        "design_count": len(design),
    }


def _workflow_draft_summary(spec: dict[str, Any]) -> dict[str, int]:
    protected = set(spec.get("protected_nodes", []))
    nodes = spec.get("nodes", [])
    design = [
        node
        for node in nodes
        if node.get("id") not in protected and not dict(node.get("ui") or {}).get("builder_hidden")
    ]
    return {
        "total_count": len(nodes),
        "protected_count": len([node for node in nodes if node.get("id") in protected]),
        "design_count": len(design),
    }


def _insert_after_index(nodes: list[dict[str, Any]], node_id: str) -> int:
    for index, node in enumerate(nodes):
        if node.get("id") == node_id:
            return index + 1
    return len(nodes)


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        edge_id = str(edge.get("id") or f"{edge.get('from')}->{edge.get('to')}")
        if edge_id in seen:
            continue
        seen.add(edge_id)
        result.append(edge)
    return result


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{"):
        return json.loads(text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("configurator response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    config_path: str | Path = "agent.yaml",
    workspace_root: str | Path = ".",
) -> None:
    server = create_server(host=host, port=port, config_path=config_path, workspace_root=workspace_root)
    print(f"Reasoning Agent Debug Console: http://{host}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping debug console.")
    finally:
        server.server_close()
