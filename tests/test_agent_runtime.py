import tempfile
import time
import unittest
from pathlib import Path

from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.runtime import (
    AgentRuntime,
    RuntimeTool,
    ToolCall,
    ToolModelResponse,
)
from reasoning_agent_template.sessions import SessionStore


class FakeToolModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses:
            raise AssertionError("FakeToolModel received more calls than expected")
        return self.responses.pop(0)


class AgentRuntimeTests(unittest.TestCase):
    def test_runtime_executes_tool_and_feeds_result_back_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="I need a tool.",
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="echo",
                                arguments={"text": "hello"},
                            )
                        ],
                    ),
                    ToolModelResponse(content="final: HELLO"),
                ]
            )
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="echo",
                        description="Uppercase text.",
                        action=lambda args: args["text"].upper(),
                    )
                ],
            )

            result = runtime.run("say hello")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "final: HELLO")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(model.calls[1]["messages"][-1]["content"], "HELLO")
        self.assertTrue(any(event["kind"] == "tool_completed" for event in result.events))

    def test_runtime_interrupts_tool_when_gate_requires_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="I need shell access.",
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="shell",
                                arguments={"command": "echo hello"},
                            )
                        ],
                    )
                ]
            )
            executed = []
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="shell",
                        description="Run a shell command.",
                        action=lambda args: executed.append(args["command"]),
                        approval_action="execute_command",
                    )
                ],
            )

            result = runtime.run("run a command")

        self.assertEqual(result.status, "interrupted")
        self.assertEqual(executed, [])
        self.assertEqual(result.gate_decisions[-1].status, "interrupt")
        self.assertTrue(any(event["kind"] == "permission_interrupted" for event in result.events))
        self.assertEqual(len(model.calls), 1)

    def test_runtime_enforces_tool_step_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="first tool",
                        tool_calls=[
                            ToolCall(
                                call_id="call_1",
                                name="echo",
                                arguments={"text": "one"},
                            )
                        ],
                    ),
                    ToolModelResponse(
                        content="second tool",
                        tool_calls=[
                            ToolCall(
                                call_id="call_2",
                                name="echo",
                                arguments={"text": "two"},
                            )
                        ],
                    ),
                ]
            )
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="echo",
                        description="Echo text.",
                        action=lambda args: args["text"],
                    )
                ],
                max_tool_steps=1,
            )

            result = runtime.run("loop")

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.tool_steps, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(any(event["kind"] == "tool_step_limit" for event in result.events))

    def test_runtime_can_persist_session_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AgentConfig.default(workspace_root=root)
            store = SessionStore(root / "sessions")
            model = FakeToolModel([ToolModelResponse(content="final answer")])
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[],
                session_store=store,
                session_id="runtime-session",
            )

            result = runtime.run("persist this")
            loaded = store.load("runtime-session")

        self.assertEqual(loaded.status, "completed")
        self.assertEqual(loaded.messages, result.messages)
        self.assertEqual(loaded.events, result.events)

    def test_runtime_validates_tool_arguments_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="call",
                        tool_calls=[ToolCall(call_id="call_1", name="typed", arguments={})],
                    )
                ]
            )
            executed = []
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="typed",
                        description="typed tool",
                        action=lambda args: executed.append(args),
                        input_schema={
                            "type": "object",
                            "required": ["value"],
                            "properties": {"value": {"type": "string"}},
                        },
                    )
                ],
            )
            result = runtime.run("invalid")

        self.assertEqual(result.status, "failed")
        self.assertEqual(executed, [])
        self.assertTrue(any(event["kind"] == "tool_validation_failed" for event in result.events))

    def test_runtime_deduplicates_idempotent_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="first",
                        tool_calls=[ToolCall(call_id="call_1", name="write", arguments={"key": "k"})],
                    ),
                    ToolModelResponse(
                        content="second",
                        tool_calls=[ToolCall(call_id="call_2", name="write", arguments={"key": "k"})],
                    ),
                    ToolModelResponse(content="done"),
                ]
            )
            executed = []
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="write",
                        description="idempotent write",
                        action=lambda args: (executed.append(args["key"]) or "ok"),
                        input_schema={"type": "object", "required": ["key"]},
                        idempotency_key_argument="key",
                    )
                ],
            )
            result = runtime.run("write twice")

        self.assertEqual(result.status, "completed")
        self.assertEqual(executed, ["k"])
        self.assertTrue(any(event["kind"] == "tool_deduplicated" for event in result.events))

    def test_runtime_reports_tool_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig.default(workspace_root=Path(tmp))
            model = FakeToolModel(
                [
                    ToolModelResponse(
                        content="slow",
                        tool_calls=[ToolCall(call_id="call_1", name="slow", arguments={})],
                    )
                ]
            )
            runtime = AgentRuntime(
                config=config,
                model=model,
                tools=[
                    RuntimeTool(
                        name="slow",
                        description="slow",
                        action=lambda args: time.sleep(0.05),
                        timeout_seconds=0.001,
                    )
                ],
            )
            result = runtime.run("slow")

        self.assertEqual(result.status, "failed")
        self.assertTrue(any(event["kind"] == "tool_timed_out" for event in result.events))


if __name__ == "__main__":
    unittest.main()
