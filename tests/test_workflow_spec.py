import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib import error, request

from reasoning_agent_template.agent_templates import AgentTemplateStore
from reasoning_agent_template.agents_spec import AgentsSpec, AgentsSpecStore
from reasoning_agent_template.code_modifier import LocalWorkflowSpecCodeModifier
from reasoning_agent_template.config import AgentConfig
from reasoning_agent_template.workflow import TemplateCoordinator
from reasoning_agent_template.workflow_spec import WorkflowSpec, WorkflowSpecStore
from reasoning_agent_template.web import create_server


class WorkflowSpecTests(unittest.TestCase):
    def test_builtin_autoresearch_template_loads_and_validates(self):
        store = AgentTemplateStore(
            Path("."),
            {
                "template_builtin_dir": "configs/templates/builtin",
                "template_user_dir": "configs/templates/user",
            },
        )

        templates = store.list_templates()
        template = store.load_template("autoresearch-agent")
        workflow_agents = {node.agent for node in template.workflow.nodes if node.agent}
        agents_validation = template.agents.validate(workflow_agent_ids=workflow_agents)
        workflow_validation = template.workflow.validate()

        self.assertIn("autoresearch-agent", {item["id"] for item in templates})
        self.assertIn("AutoResearch", template.label)
        self.assertTrue(agents_validation.ok, agents_validation.to_dict())
        self.assertTrue(workflow_validation.ok, workflow_validation.to_dict())
        self.assertIn("research_lead", {agent.id for agent in template.agents.agents})
        self.assertIn("research_plan", {node.id for node in template.workflow.nodes})
        self.assertIn("research_report", {node.id for node in template.workflow.nodes})

    def test_template_api_apply_writes_agents_and_workflow_drafts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            builtin_dir = root / "configs" / "templates" / "builtin"
            builtin_dir.mkdir(parents=True)
            (builtin_dir / "autoresearch-agent.template.json").write_text(
                Path("configs/templates/builtin/autoresearch-agent.template.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            server = create_server(host="127.0.0.1", port=0, config_path=root / "agent.yaml", workspace_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                templates = _get_json(f"{base_url}/api/templates")
                applied = _post_json(f"{base_url}/api/templates/apply", {"template_id": "autoresearch-agent"})
                agents_draft_exists = (root / "configs" / "agents" / "default.agents.draft.json").exists()
                workflow_draft_exists = (root / "configs" / "workflows" / "default.workflow.draft.json").exists()
            finally:
                server.shutdown()
                server.server_close()

        agent_ids = {agent["id"] for agent in applied["agents"]["spec"]["agents"]}
        node_ids = {node["id"] for node in applied["workflow"]["spec"]["nodes"]}
        self.assertIn("autoresearch-agent", {item["id"] for item in templates["templates"]})
        self.assertEqual(applied["status"], "drafted")
        self.assertTrue(agents_draft_exists)
        self.assertTrue(workflow_draft_exists)
        self.assertIn("research_lead", agent_ids)
        self.assertIn("research_report", node_ids)
        self.assertTrue(applied["agents"]["validation"]["ok"], applied["agents"]["validation"])
        self.assertTrue(applied["workflow"]["validation"]["ok"], applied["workflow"]["validation"])

    def test_template_api_save_current_design_creates_user_template_and_can_reapply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(host="127.0.0.1", port=0, config_path=root / "agent.yaml", workspace_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                saved = _post_json(
                    f"{base_url}/api/templates/save",
                    {
                        "template_id": "my-research-v1",
                        "label": "My Research V1",
                        "description": "Saved local research design.",
                        "agents": AgentsSpec.default().to_dict(),
                        "workflow": WorkflowSpec.default().to_dict(),
                    },
                )
                templates = _get_json(f"{base_url}/api/templates")
                applied = _post_json(f"{base_url}/api/templates/apply", {"template_id": "my-research-v1"})
                template_file_exists = (root / "configs" / "templates" / "user" / "my-research-v1.template.json").exists()
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(saved["status"], "saved")
        self.assertEqual(saved["template"]["source"], "user")
        self.assertTrue(template_file_exists)
        self.assertIn("my-research-v1", {item["id"] for item in templates["templates"]})
        self.assertEqual(applied["status"], "drafted")
        self.assertTrue(applied["agents"]["validation"]["ok"], applied["agents"]["validation"])
        self.assertTrue(applied["workflow"]["validation"]["ok"], applied["workflow"]["validation"])

    def test_template_api_rejects_unsafe_template_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(host="127.0.0.1", port=0, config_path=root / "agent.yaml", workspace_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                save_status, save_error = _post_json_expect_error(
                    f"{base_url}/api/templates/save",
                    {"template_id": "../x", "label": "bad"},
                )
                apply_status, apply_error = _post_json_expect_error(
                    f"{base_url}/api/templates/apply",
                    {"template_id": "../x"},
                )
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(save_status, 400)
        self.assertEqual(apply_status, 400)
        self.assertIn("invalid template id", save_error["error"])
        self.assertIn("invalid template id", apply_error["error"])

    def test_default_agents_spec_validates_and_protects_core_agents(self):
        base = AgentsSpec.default()
        data = base.to_dict()
        data["agents"] = [agent for agent in data["agents"] if agent["id"] != "coordinator"]

        validation = AgentsSpec.from_dict(data).validate(base=base, workflow_agent_ids={"coordinator", "planner"})

        self.assertFalse(validation.ok)
        self.assertTrue(any("protected agent cannot be deleted: coordinator" in item for item in validation.errors))

    def test_local_code_modifier_applies_approved_agents_spec_only_inside_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = AgentsSpecStore(root)
            spec = AgentsSpec.default()
            store.save_draft(spec)
            proposal = store.create_proposal(spec)

            result = LocalWorkflowSpecCodeModifier(root).apply_agents_proposal(
                proposal,
                approved_by="tester",
            )

        self.assertEqual(result.status, "applied")
        self.assertIn("configs/agents/default.agents.json", result.modified_files)

    def test_default_workflow_spec_validates_and_drives_existing_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "knowledge").mkdir()
            config = AgentConfig.default(workspace_root=root)
            config.knowledge["directory"] = str(root / "knowledge")

            spec = WorkflowSpec.default()
            validation = spec.validate()
            coordinator = TemplateCoordinator(config=config, workspace_root=root)
            result = coordinator.run("hello")

        self.assertTrue(validation.ok, validation.to_dict())
        self.assertEqual(result.stage_trace, spec.node_ids())
        self.assertEqual(result.state.current_stage, "respond")

    def test_workflow_validation_rejects_missing_edges_and_protected_deletion(self):
        base = WorkflowSpec.default()
        data = base.to_dict()
        data["nodes"] = [node for node in data["nodes"] if node["id"] != "gate"]
        candidate = WorkflowSpec.from_dict(data)

        validation = candidate.validate(base=base)

        self.assertFalse(validation.ok)
        self.assertTrue(any("protected node cannot be deleted: gate" in item for item in validation.errors))
        self.assertTrue(any("references missing" in item for item in validation.errors))

    def test_unknown_builtin_handler_requires_code_before_runtime(self):
        data = WorkflowSpec.default().to_dict()
        data["nodes"].insert(
            4,
            {
                "id": "domain_review",
                "label": "领域审查",
                "agent": "critic",
                "description": "领域专家审查",
                "work": "审查领域事实和交付质量。",
                "input_contract": "答案草案",
                "output_contract": "领域审查记录",
                "handler_kind": "builtin",
                "handler": "domain_review",
                "checkpoint": True,
                "gate_policy": {},
                "ui": {},
            },
        )
        data["edges"].append(
            {
                "id": "reason_to_domain_review",
                "from": "reason",
                "to": "domain_review",
                "type": "flow",
                "condition": "draft ready",
                "handoff_contract": {},
                "gate_policy": {},
                "planner_contract": {},
                "reviewer_required": True,
            }
        )
        data["edges"].append(
            {
                "id": "domain_review_to_evidence_audit",
                "from": "domain_review",
                "to": "evidence_audit",
                "type": "flow",
                "condition": "review complete",
                "handoff_contract": {},
                "gate_policy": {},
                "planner_contract": {},
                "reviewer_required": False,
            }
        )
        spec = WorkflowSpec.from_dict(data)

        validation = spec.validate(base=WorkflowSpec.default())

        self.assertFalse(validation.ok)
        self.assertTrue(any("domain_review" in item for item in validation.requires_code))

    def test_local_code_modifier_applies_approved_spec_only_inside_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowSpecStore(root)
            spec = WorkflowSpec.default()
            store.save_draft(spec)
            proposal = store.create_proposal(spec)

            result = LocalWorkflowSpecCodeModifier(root).apply_workflow_proposal(
                proposal,
                approved_by="tester",
            )

        self.assertEqual(result.status, "applied")
        self.assertIn("configs/workflows/default.workflow.json", result.modified_files)

    def test_local_code_modifier_interrupts_unapproved_or_denied_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = WorkflowSpecStore(root)
            proposal = store.create_proposal(WorkflowSpec.default())
            modifier = LocalWorkflowSpecCodeModifier(root)

            unapproved = modifier.apply_workflow_proposal(proposal)
            proposal["target_path"] = "evidence/ledger.jsonl"
            proposal["modified_files"] = ["evidence/ledger.jsonl"]
            denied_path = modifier.apply_workflow_proposal(proposal, approved_by="tester")

        self.assertEqual(unapproved.status, "interrupt")
        self.assertTrue(any("approval is required" in reason for reason in unapproved.gate_decision["reasons"]))
        self.assertEqual(denied_path.status, "interrupt")
        self.assertTrue(any("denied path" in reason for reason in denied_path.gate_decision["reasons"]))

    def test_workflow_api_draft_proposal_and_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(host="127.0.0.1", port=0, config_path=root / "agent.yaml", workspace_root=root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                spec_payload = _get_json(f"{base_url}/api/workflow/spec")
                draft = spec_payload["spec"]
                draft["revision"] = "api-test"

                draft_payload = _post_json(f"{base_url}/api/workflow/draft", {"spec": draft})
                proposal = _post_json(f"{base_url}/api/workflow/proposal", {})
                applied = _post_json(
                    f"{base_url}/api/workflow/apply",
                    {"proposal_id": proposal["proposal_id"], "approved": True, "approved_by": "tester"},
                )
            finally:
                server.shutdown()
                server.server_close()

        self.assertTrue(draft_payload["validation"]["ok"])
        self.assertEqual(applied["status"], "applied")
        self.assertEqual(applied["workflow"]["spec"]["revision"], "api-test")

    def test_configurator_compose_draft_proposal_and_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: _FakeConfiguratorClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                spec_payload = _get_json(f"{base_url}/api/agents/spec")
                self.assertTrue(spec_payload["validation"]["ok"])

                composed = _post_json(
                    f"{base_url}/api/configurator/compose",
                    {"prompt": "/配置 我要一个材料学研究多 Agent，带领域研究员，并在推理后加入领域审查"},
                )
                proposal = _post_json(f"{base_url}/api/agents/proposal", {})
                applied = _post_json(
                    f"{base_url}/api/agents/apply",
                    {"proposal_id": proposal["proposal_id"], "approved": True, "approved_by": "tester"},
                )
            finally:
                server.shutdown()
                server.server_close()

        configured = composed["agents"]
        workflow_configured = composed["workflow"]
        ids = {agent["id"] for agent in configured["spec"]["agents"]}
        self.assertEqual(composed["status"], "completed")
        self.assertEqual(composed["targets"], ["agents", "workflow"])
        self.assertIn("配置助手已生成草稿", composed["answer"])
        self.assertEqual(configured["source"], "deepseek")
        self.assertIn("domain_researcher", ids)
        self.assertTrue(configured["validation"]["ok"])
        self.assertTrue(workflow_configured["validation"]["ok"])
        self.assertTrue(any(node["id"] == "domain_review" for node in workflow_configured["spec"]["nodes"]))
        self.assertEqual(applied["status"], "applied")
        self.assertTrue(any(agent["id"] == "domain_researcher" for agent in applied["agents"]["spec"]["agents"]))

    def test_configurator_honors_explicit_design_counts_without_deleting_scaffold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: _FakeConfiguratorClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                composed = _post_json(
                    f"{base_url}/api/configurator/compose",
                    {"prompt": "/配置 生成 3 个 Agent 和 3-4 个节点，材料学研究用途"},
                )
            finally:
                server.shutdown()
                server.server_close()

        agents_spec = composed["agents"]["spec"]
        workflow_spec = composed["workflow"]["spec"]
        protected_agents = set(agents_spec["protected_agents"])
        protected_nodes = set(workflow_spec["protected_nodes"])
        design_agents = [
            agent
            for agent in agents_spec["agents"]
            if agent["id"] not in protected_agents and not agent.get("ui", {}).get("builder_hidden")
        ]
        hidden_agents = [
            agent for agent in agents_spec["agents"] if agent["id"] in protected_agents and agent.get("ui", {}).get("builder_hidden")
        ]
        design_nodes = [
            node
            for node in workflow_spec["nodes"]
            if node["id"] not in protected_nodes and not node.get("ui", {}).get("builder_hidden")
        ]
        hidden_nodes = [
            node for node in workflow_spec["nodes"] if node["id"] in protected_nodes and node.get("ui", {}).get("builder_hidden")
        ]

        self.assertEqual(composed["agents"]["summary"]["design_count"], 3)
        self.assertEqual(composed["workflow"]["summary"]["design_count"], 4)
        self.assertEqual(len(design_agents), 3)
        self.assertEqual(len(design_nodes), 4)
        self.assertGreaterEqual(len(hidden_agents), 1)
        self.assertGreaterEqual(len(hidden_nodes), 1)
        self.assertTrue(composed["agents"]["validation"]["ok"])
        self.assertTrue(composed["workflow"]["validation"]["ok"])
        self.assertIn("设计层 3 个", composed["answer"])
        self.assertIn("设计层 4 个节点", composed["answer"])

    def test_configurator_infers_small_design_for_simple_goals_without_explicit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: _FakeConfiguratorClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                composed = _post_json(
                    f"{base_url}/api/configurator/compose",
                    {"prompt": "/配置 做一个只回答常见问题的轻量客服 Agent"},
                )
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(composed["builder_plan"]["scale"], "simple")
        self.assertFalse(composed["builder_plan"]["explicit_agent_count"])
        self.assertFalse(composed["builder_plan"]["explicit_node_count"])
        self.assertEqual(composed["agents"]["summary"]["design_count"], 1)
        self.assertEqual(composed["workflow"]["summary"]["design_count"], 2)
        self.assertIn("规模判断: simple", composed["answer"])

    def test_configurator_infers_larger_design_for_complex_goals_without_explicit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: _FakeConfiguratorClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                composed = _post_json(
                    f"{base_url}/api/configurator/compose",
                    {
                        "prompt": "/配置 做一个材料学研究多 Agent，自动检索论文和 RAG，带证据门禁、reviewer 审查、长期记忆沉淀和自进化提案"
                    },
                )
            finally:
                server.shutdown()
                server.server_close()

        self.assertEqual(composed["builder_plan"]["scale"], "complex")
        self.assertGreaterEqual(composed["agents"]["summary"]["design_count"], 3)
        self.assertGreaterEqual(composed["workflow"]["summary"]["design_count"], 4)
        self.assertLessEqual(composed["agents"]["summary"]["design_count"], 4)
        self.assertLessEqual(composed["workflow"]["summary"]["design_count"], 6)
        self.assertIn("规模判断: complex", composed["answer"])

    def test_configurator_replaces_prior_design_layer_without_stale_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent.yaml").write_text(
                Path("agent.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "knowledge").mkdir()
            server = create_server(
                host="127.0.0.1",
                port=0,
                config_path=root / "agent.yaml",
                workspace_root=root,
                llm_client_factory=lambda _config: _FakeConfiguratorClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                _post_json(
                    f"{base_url}/api/configurator/compose",
                    {"prompt": "/配置 生成 3 个 Agent 和 3-4 个节点，材料学研究用途"},
                )
                composed = _post_json(
                    f"{base_url}/api/configurator/compose",
                    {"prompt": "/配置 做一个只回答常见问题的轻量客服 Agent"},
                )
            finally:
                server.shutdown()
                server.server_close()

        workflow_spec = composed["workflow"]["spec"]
        node_ids = {node["id"] for node in workflow_spec["nodes"]}
        self.assertTrue(composed["workflow"]["validation"]["ok"], composed["workflow"]["validation"])
        self.assertEqual(composed["workflow"]["summary"]["design_count"], 2)
        self.assertTrue(all(edge["from"] in node_ids and edge["to"] in node_ids for edge in workflow_spec["edges"]))


def _get_json(url: str):
    return json.loads(request.urlopen(url, timeout=10).read().decode("utf-8"))


def _post_json(url: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(request.urlopen(req, timeout=10).read().decode("utf-8"))


def _post_json_expect_error(url: str, body: dict):
    data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        request.urlopen(req, timeout=10)
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))
    raise AssertionError("request unexpectedly succeeded")


class _FakeConfiguratorClient:
    model = "fake-configurator"

    def chat(self, messages, *, temperature, max_tokens):
        data = AgentsSpec.default().to_dict()
        data["revision"] = "fake-configurator"
        data["agents"].append(
            {
                "id": "domain_researcher",
                "label": "领域研究员",
                "description": "检索材料学论文并形成证据候选。",
                "responsibilities": ["论文检索", "证据筛选"],
                "model_role": "worker",
                "tools": ["external_search"],
                "permissions": {},
                "memory_access": ["long_term:read"],
                "workflow_nodes": [],
                "handoff_contract": {},
                "ui": {},
            }
        )
        return type("Result", (), {"content": json.dumps(data, ensure_ascii=False)})()


if __name__ == "__main__":
    unittest.main()
