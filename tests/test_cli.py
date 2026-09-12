import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from reasoning_agent_template import cli
from reasoning_agent_template.llm import ChatResult


class CliTests(unittest.TestCase):
    def run_cli(self, args):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_help_lists_core_commands(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with redirect_stdout(stdout):
                cli.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertIn("chat", text)
        self.assertIn("skills", text)
        self.assertIn("deepseek-smoke", text)
        self.assertIn("test", text)
        self.assertIn("web", text)

    def test_chat_runs_local_state_machine_with_optional_evidence_for_routine_chat(self):
        class FakeClient:
            model = "deepseek-v4-flash"

            def chat(self, messages, temperature, max_tokens):
                return ChatResult(content="CLI_DEEPSEEK_MARK", model=self.model, raw={})

        with patch("reasoning_agent_template.multiagent.DeepSeekChatClient.from_config", return_value=FakeClient()):
            code, stdout, stderr = self.run_cli(["chat", "What constraints does this template enforce?"])

        self.assertEqual(code, 0, stderr)
        self.assertIn("ANSWER=CLI_DEEPSEEK_MARK", stdout)
        self.assertIn("TRACE=intake -> plan -> reason -> evidence_audit", stdout)
        self.assertIn("GATE=allow", stdout)
        self.assertIn("LLM=called", stdout)
        self.assertIn("EVIDENCE_MODE=optional", stdout)
        self.assertIn("EVIDENCE=\n", stdout)

    def test_chat_json_output_is_machine_readable(self):
        class FakeClient:
            model = "deepseek-v4-flash"

            def chat(self, messages, temperature, max_tokens):
                return ChatResult(content="JSON_DEEPSEEK_MARK", model=self.model, raw={})

        with patch("reasoning_agent_template.multiagent.DeepSeekChatClient.from_config", return_value=FakeClient()):
            code, stdout, stderr = self.run_cli(["chat", "--json", "What constraints does this template enforce?"])

        self.assertEqual(code, 0, stderr)
        self.assertIn('"answer"', stdout)
        self.assertIn("JSON_DEEPSEEK_MARK", stdout)
        self.assertIn('"state_machine"', stdout)
        self.assertIn('"evidence"', stdout)
        self.assertIn('"llm"', stdout)

    def test_skills_lists_local_skill_metadata(self):
        code, stdout, stderr = self.run_cli(["skills"])

        self.assertEqual(code, 0, stderr)
        self.assertIn("evidence-first", stdout)
        self.assertIn("minimal-change", stdout)

    def test_deepseek_smoke_uses_injected_client(self):
        class FakeClient:
            model = "deepseek-v4-flash"

            def chat(self, messages, temperature, max_tokens):
                return ChatResult(
                    content="TEMPLATE_OK: CLI smoke path works.",
                    model=self.model,
                    raw={},
                )

        with patch("reasoning_agent_template.cli.DeepSeekChatClient.from_config", return_value=FakeClient()):
            code, stdout, stderr = self.run_cli(["deepseek-smoke", "--no-env-check", "--client-role", "worker"])

        self.assertEqual(code, 0, stderr)
        self.assertIn("MODEL=deepseek-v4-flash", stdout)
        self.assertIn("TEMPLATE_OK", stdout)

    def test_test_command_runs_unittest_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            tests_dir = Path(tmp) / "sample_tests"
            tests_dir.mkdir()
            (tests_dir / "test_sample.py").write_text(
                "import unittest\n\nclass Sample(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
                encoding="utf-8",
            )

            code, stdout, stderr = self.run_cli(["test", "--tests-dir", str(tests_dir)])

        self.assertEqual(code, 0, stderr)
        self.assertIn("OK", stderr)

    def test_rag_eval_reports_recall_and_index_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "target.md").write_text(
                "HNSW vector search uses graph layers for approximate nearest neighbor retrieval.",
                encoding="utf-8",
            )
            config_path = root / "agent.json"
            config_path.write_text(
                json.dumps(
                    {
                        "knowledge": {
                            "directory": "knowledge",
                            "index_type": "hybrid",
                            "top_k": 5,
                            "min_score": 0.0,
                            "chunk_size": 1400,
                        }
                    }
                ),
                encoding="utf-8",
            )
            cases_path = root / "cases.json"
            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "hnsw",
                                "query": "HNSW nearest neighbor graph retrieval",
                                "expected_sources": ["knowledge/target.md"],
                                "tags": ["direct"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            code, stdout, stderr = self.run_cli(
                [
                    "--config",
                    str(config_path),
                    "--workspace",
                    str(root),
                    "rag-eval",
                    "--cases",
                    str(cases_path),
                    "--json",
                ]
            )

        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["index"]["chunk_count"], 1)
        self.assertEqual(payload["methods"]["hybrid"]["recall"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
