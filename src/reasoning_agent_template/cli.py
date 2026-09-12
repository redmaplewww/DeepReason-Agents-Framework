from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Sequence

from reasoning_agent_template.config import load_agent_config
from reasoning_agent_template.llm import ChatMessage, DeepSeekChatClient, LLMRequestError, MissingApiKeyError
from reasoning_agent_template.multiagent import MultiAgentOrchestrator
from reasoning_agent_template.sessions import SessionStore
from reasoning_agent_template.rag_eval import evaluate_knowledge_base, format_markdown_report, load_cases
from reasoning_agent_template.skills import SkillRegistry
from reasoning_agent_template.web import serve


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reasoning-agent",
        description="CLI for the heavy-reasoning agent template.",
    )
    parser.add_argument("--config", default="agent.yaml", help="Path to agent.yaml.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Run the local evidence/state-machine coordinator.")
    chat.add_argument("question", nargs="+", help="Question to ask the local template coordinator.")
    chat.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    chat.set_defaults(handler=_cmd_chat)

    skills = subparsers.add_parser("skills", help="List enabled local skill packs.")
    skills.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    skills.set_defaults(handler=_cmd_skills)

    smoke = subparsers.add_parser("deepseek-smoke", help="Run a real DeepSeek chat-completions smoke test.")
    smoke.add_argument("--client-role", default="worker", help="Model role from agent.yaml to use.")
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--max-tokens", type=int, default=256)
    smoke.add_argument(
        "--no-env-check",
        action="store_true",
        help="Skip the preflight API-key check; useful when tests inject a fake client.",
    )
    smoke.set_defaults(handler=_cmd_deepseek_smoke)

    test = subparsers.add_parser("test", help="Run unittest discovery.")
    test.add_argument("--tests-dir", default="tests", help="Directory to discover tests from.")
    test.add_argument("--pattern", default="test*.py", help="Unittest discovery pattern.")
    test.set_defaults(handler=_cmd_test)

    rag_eval = subparsers.add_parser("rag-eval", help="Evaluate local RAG recall across retrieval methods.")
    rag_eval.add_argument("--cases", default="configs/rag_benchmark_cases.json", help="Path to RAG benchmark cases.")
    rag_eval.add_argument("--report", default="", help="Optional markdown report output path.")
    rag_eval.add_argument("--min-score", type=float, default=None, help="Override retrieval min_score.")
    rag_eval.add_argument("--top-ks", default="1,3,5", help="Comma-separated K values.")
    rag_eval.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    rag_eval.set_defaults(handler=_cmd_rag_eval)

    web = subparsers.add_parser("web", help="Start the local web chat and debug console.")
    web.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    web.add_argument("--port", type=int, default=8765, help="Port to bind.")
    web.set_defaults(handler=_cmd_web)

    return parser


def _cmd_chat(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    workspace = Path(args.workspace)
    config = load_agent_config(config_path)
    try:
        payload = MultiAgentOrchestrator(
            config=config,
            workspace_root=workspace,
            session_store=SessionStore(workspace / "sessions"),
        ).run(" ".join(args.question))
    except (MissingApiKeyError, LLMRequestError) as exc:
        print(f"DeepSeek chat failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("ANSWER=" + payload["answer"])
    print("TRACE=" + " -> ".join(payload["state_machine"]["trace"]))
    decisions = payload["gates"]["decisions"]
    if decisions:
        print("GATE=" + decisions[-1]["status"])
    print("LLM=" + payload["runtime"]["llm"]["status"])
    print("MODEL=" + payload["runtime"]["llm"]["model"])
    print("EVIDENCE_MODE=" + payload["evidence"]["mode"])
    print("RISK=" + payload["evidence"]["risk_level"])
    print("EVIDENCE=" + ", ".join(item["id"] for item in payload["evidence"]["items"]))
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    config = load_agent_config(Path(args.config))
    skills_dir = Path(config.skills.get("directory", "skills"))
    if not skills_dir.is_absolute():
        skills_dir = Path(args.workspace) / skills_dir
    skills = SkillRegistry(skills_dir).load()
    enabled = config.skills.get("enabled", sorted(skills))
    rows = [
        {
            "name": name,
            "description": skills[name].description if name in skills else "",
            "path": str(skills[name].path) if name in skills else "",
        }
        for name in enabled
    ]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print(f"{row['name']}\t{row['description']}")
    return 0


def _cmd_deepseek_smoke(args: argparse.Namespace) -> int:
    if not args.no_env_check and not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is required for deepseek-smoke.", file=sys.stderr)
        return 2

    try:
        config = load_agent_config(Path(args.config))
        client = DeepSeekChatClient.from_config(config, role=args.client_role)
        result = client.chat(
            [
                ChatMessage(
                    role="system",
                    content="Return exactly one line starting with TEMPLATE_OK: .",
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "用一句中文说明证据优先推理、状态门禁、记忆提案适合作为 Agent 模板约束。"
                    ),
                ),
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if "TEMPLATE_OK" not in result.content:
        print("DeepSeek smoke test failed: TEMPLATE_OK marker missing", file=sys.stderr)
        return 1
    print(f"MODEL={result.model}")
    print("ANSWER=" + result.content.replace("\n", " ").strip())
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    suite = unittest.defaultTestLoader.discover(args.tests_dir, pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _cmd_rag_eval(args: argparse.Namespace) -> int:
    config = load_agent_config(Path(args.config))
    workspace = Path(args.workspace)
    knowledge_dir = Path(str(config.knowledge.get("directory", "knowledge")))
    if not knowledge_dir.is_absolute():
        knowledge_dir = workspace / knowledge_dir
    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = workspace / cases_path
    top_ks = [int(item.strip()) for item in str(args.top_ks).split(",") if item.strip()]
    min_score = float(config.knowledge.get("min_score", 0.0) if args.min_score is None else args.min_score)
    payload = evaluate_knowledge_base(
        knowledge_dir=knowledge_dir,
        cases=load_cases(cases_path),
        top_ks=top_ks,
        min_score=min_score,
        max_chunk_chars=int(config.knowledge.get("chunk_size", 1400)),
    )
    report = format_markdown_report(payload)
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = workspace / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(report)
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    serve(host=args.host, port=args.port, config_path=args.config, workspace_root=args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
