from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.gates import GatePolicy
from reasoning_agent_template.models import GateDecision, RuntimeHandle, utc_now
from reasoning_agent_template.sessions import SessionStore


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class ToolModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class ToolCallingModel(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolModelResponse:
        ...


@dataclass(frozen=True)
class RuntimeTool:
    name: str
    description: str
    action: Callable[[dict[str, Any]], Any]
    approval_action: str | None = None
    risk_level: str = "low"
    input_schema: dict[str, Any] = field(default_factory=dict)
    target_path_argument: str | None = None
    approved_by: str | None = None
    timeout_seconds: float | None = None
    idempotency_key_argument: str | None = None

    def to_model_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema or {"type": "object"},
        }

    def target_path(self, arguments: dict[str, Any]) -> Path | None:
        if not self.target_path_argument:
            return None
        value = arguments.get(self.target_path_argument)
        if value is None:
            return None
        return Path(str(value))


@dataclass(frozen=True)
class AgentRuntimeResult:
    status: str
    answer: str
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]
    gate_decisions: list[GateDecision] = field(default_factory=list)
    tool_steps: int = 0


class AgentRuntime:
    """Small OpenClaude-inspired tool loop for template agents.

    The class intentionally stays UI-free. CLI, web, tests, and future SDK
    wrappers can consume the same events and messages without importing a
    terminal renderer.
    """

    def __init__(
        self,
        *,
        config: AgentConfig,
        model: ToolCallingModel,
        tools: list[RuntimeTool],
        gate_policy: GatePolicy | None = None,
        max_turns: int = 8,
        max_tool_steps: int = 32,
        approved_by: str | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
    ):
        self.config = config
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.gate_policy = gate_policy or GatePolicy(
            workspace_root=config.workspace_root,
            min_evidence_by_risk=dict(config.gates.get("min_evidence_by_risk", {})),
            approval_required_actions=set(config.gates.get("approval_required_actions", [])),
        )
        self.max_turns = max_turns
        self.max_tool_steps = max_tool_steps
        self.approved_by = approved_by
        self.session_store = session_store
        self.session_id = session_id
        self._idempotency_results: dict[str, str] = {}

    def run(self, prompt: str) -> AgentRuntimeResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        events: list[dict[str, Any]] = [
            {
                "time": utc_now(),
                "kind": "runtime_started",
                "message": "runtime started",
            }
        ]
        gate_decisions: list[GateDecision] = []
        tool_steps = 0
        answer = ""

        for _turn in range(self.max_turns):
            response = self.model.complete(
                messages,
                [tool.to_model_schema() for tool in self.tools.values()],
            )
            answer = response.content
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.to_dict() for tool_call in response.tool_calls
                ]
            messages.append(assistant_message)
            events.append(
                {
                    "time": utc_now(),
                    "kind": "model_completed",
                    "tool_calls": len(response.tool_calls),
                }
            )

            if not response.tool_calls:
                events.append(
                    {
                        "time": utc_now(),
                        "kind": "runtime_completed",
                        "message": "runtime completed",
                    }
                )
                return self._result(
                    status="completed",
                    answer=answer,
                    messages=messages,
                    events=events,
                    gate_decisions=gate_decisions,
                    tool_steps=tool_steps,
                )

            for tool_call in response.tool_calls:
                if tool_steps >= self.max_tool_steps:
                    events.append(
                        {
                            "time": utc_now(),
                            "kind": "tool_step_limit",
                            "tool": tool_call.name,
                            "limit": self.max_tool_steps,
                        }
                    )
                    return self._result(
                        status="step_limit",
                        answer=response.content,
                        messages=messages,
                        events=events,
                        gate_decisions=gate_decisions,
                        tool_steps=tool_steps,
                    )

                tool = self.tools.get(tool_call.name)
                if tool is None:
                    events.append(
                        {
                            "time": utc_now(),
                            "kind": "unknown_tool",
                            "tool": tool_call.name,
                        }
                    )
                    return self._result(
                        status="failed",
                        answer=f"Unknown tool: {tool_call.name}",
                        messages=messages,
                        events=events,
                        gate_decisions=gate_decisions,
                        tool_steps=tool_steps,
                    )

                events.append(
                    {
                        "time": utc_now(),
                        "kind": "tool_requested",
                        "tool": tool.name,
                        "call_id": tool_call.call_id,
                    }
                )
                decision = self._gate(tool, tool_call.arguments)
                if decision is not None:
                    gate_decisions.append(decision)
                    if decision.status != "allow":
                        events.append(
                            {
                                "time": utc_now(),
                                "kind": "permission_interrupted",
                                "tool": tool.name,
                                "status": decision.status,
                                "reasons": list(decision.reasons),
                            }
                        )
                        return self._result(
                            status="interrupted",
                            answer=response.content,
                            messages=messages,
                            events=events,
                            gate_decisions=gate_decisions,
                            tool_steps=tool_steps,
                        )

                validation_errors = _validate_tool_arguments(tool.input_schema, tool_call.arguments)
                if validation_errors:
                    events.append(
                        {
                            "time": utc_now(),
                            "kind": "tool_validation_failed",
                            "tool": tool.name,
                            "errors": validation_errors,
                        }
                    )
                    return self._result(
                        status="failed",
                        answer=f"Invalid arguments for {tool.name}: {'; '.join(validation_errors)}",
                        messages=messages,
                        events=events,
                        gate_decisions=gate_decisions,
                        tool_steps=tool_steps,
                    )

                idempotency_key = None
                if tool.idempotency_key_argument:
                    raw_key = tool_call.arguments.get(tool.idempotency_key_argument)
                    if raw_key is not None and str(raw_key).strip():
                        idempotency_key = f"{tool.name}:{str(raw_key).strip()}"
                if idempotency_key in self._idempotency_results:
                    content = self._idempotency_results[idempotency_key]
                    events.append(
                        {
                            "time": utc_now(),
                            "kind": "tool_deduplicated",
                            "tool": tool.name,
                            "call_id": tool_call.call_id,
                            "idempotency_key": idempotency_key,
                        }
                    )
                    output = content
                else:
                    try:
                        output = _invoke_tool(tool, dict(tool_call.arguments))
                    except FutureTimeoutError:
                        events.append(
                            {
                                "time": utc_now(),
                                "kind": "tool_timed_out",
                                "tool": tool.name,
                                "timeout_seconds": tool.timeout_seconds,
                            }
                        )
                        return self._result(
                            status="failed",
                            answer=f"Tool timed out: {tool.name}",
                            messages=messages,
                            events=events,
                            gate_decisions=gate_decisions,
                            tool_steps=tool_steps,
                        )
                    except Exception as exc:
                        events.append(
                            {
                                "time": utc_now(),
                                "kind": "tool_failed",
                                "tool": tool.name,
                                "error": str(exc),
                            }
                        )
                        return self._result(
                            status="failed",
                            answer=str(exc),
                            messages=messages,
                            events=events,
                            gate_decisions=gate_decisions,
                            tool_steps=tool_steps,
                        )
                    content = _stringify_tool_result(output)
                    if idempotency_key:
                        self._idempotency_results[idempotency_key] = content

                tool_steps += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "name": tool.name,
                        "content": content,
                    }
                )
                events.append(
                    {
                        "time": utc_now(),
                        "kind": "tool_completed",
                        "tool": tool.name,
                        "call_id": tool_call.call_id,
                    }
                )

        events.append(
            {
                "time": utc_now(),
                "kind": "max_turns_reached",
                "limit": self.max_turns,
            }
        )
        return self._result(
            status="max_turns",
            answer=answer,
            messages=messages,
            events=events,
            gate_decisions=gate_decisions,
            tool_steps=tool_steps,
        )

    def _result(
        self,
        *,
        status: str,
        answer: str,
        messages: list[dict[str, Any]],
        events: list[dict[str, Any]],
        gate_decisions: list[GateDecision],
        tool_steps: int,
    ) -> AgentRuntimeResult:
        result = AgentRuntimeResult(
            status=status,
            answer=answer,
            messages=messages,
            events=events,
            gate_decisions=gate_decisions,
            tool_steps=tool_steps,
        )
        if self.session_store is not None and self.session_id is not None:
            self.session_store.record_snapshot(
                session_id=self.session_id,
                messages=result.messages,
                events=result.events,
                status=result.status,
                metadata={"tool_steps": result.tool_steps},
            )
        return result

    def _gate(
        self,
        tool: RuntimeTool,
        arguments: dict[str, Any],
    ) -> GateDecision | None:
        if tool.approval_action is None:
            return None
        return self.gate_policy.evaluate(
            action=tool.approval_action,
            risk_level=tool.risk_level,
            evidence=[],
            target_path=tool.target_path(arguments),
            approved_by=tool.approved_by or self.approved_by,
        )


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _invoke_tool(tool: RuntimeTool, arguments: dict[str, Any]) -> Any:
    if tool.timeout_seconds is None:
        return tool.action(arguments)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(tool.action, arguments)
    try:
        return future.result(timeout=max(0.001, float(tool.timeout_seconds)))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _validate_tool_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    if not schema:
        return []
    errors: list[str] = []
    if schema.get("type", "object") != "object" or not isinstance(arguments, dict):
        return ["arguments must be an object"]
    for name in schema.get("required", []):
        if name not in arguments:
            errors.append(f"missing required field: {name}")
    for name, property_schema in dict(schema.get("properties", {})).items():
        if name not in arguments or not isinstance(property_schema, dict):
            continue
        expected = property_schema.get("type")
        value = arguments[name]
        valid = {
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }.get(expected, True)
        if not valid:
            errors.append(f"{name} must be {expected}")
    return errors


def create_deep_agent_runtime(
    *,
    config: AgentConfig,
    tools: list[Any],
    skills_dir: Path,
    prefer_deepagents: bool | None = None,
) -> RuntimeHandle:
    """Create a Deep Agents runtime when requested, otherwise use a local fallback."""

    should_use_deepagents = (
        config.runtime.get("prefer_deepagents", False) if prefer_deepagents is None else prefer_deepagents
    )
    if should_use_deepagents:
        try:
            from deepagents import create_deep_agent  # type: ignore

            interrupt_on = {
                action: True for action in config.gates.get("approval_required_actions", [])
            }
            agent = create_deep_agent(
                tools=tools,
                instructions=_instructions(config),
                skills=str(skills_dir),
                interrupt_on=interrupt_on,
            )
            return RuntimeHandle(backend="deepagents", invoke=agent.invoke)
        except Exception as exc:
            return RuntimeHandle(
                backend="fallback",
                invoke=lambda payload: {"messages": payload.get("messages", [])},
                degraded_reason=f"deepagents initialization failed: {type(exc).__name__}: {exc}",
            )

    return RuntimeHandle(backend="fallback", invoke=lambda payload: {"messages": payload.get("messages", [])})


def _instructions(config: AgentConfig) -> str:
    return (
        f"You are {config.identity.get('name', 'a reasoning agent')}. "
        "Follow evidence-first reasoning, gate risky actions, preserve minimal changes, "
        "and generate self-evolution proposals instead of directly mutating core skills."
    )
