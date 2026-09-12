import json
import tempfile
import unittest
from pathlib import Path

from reasoning_agent_template.evidence import EvidenceLedger
from reasoning_agent_template.evolution import SelfEvolutionEngine
from reasoning_agent_template.gates import GatePolicy
from reasoning_agent_template.knowledge import LocalKnowledgeBase, WikipediaKnowledgeSource
from reasoning_agent_template.memory import LongTermMemoryStore


class KnowledgeMemoryEvolutionTests(unittest.TestCase):
    def test_local_knowledge_retrieval_returns_citable_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "policy.md").write_text(
                "# Policy\nEvidence-first answers must cite source spans.",
                encoding="utf-8",
            )
            (knowledge_dir / "memory.txt").write_text(
                "Long-term memory writes require a gate decision.",
                encoding="utf-8",
            )
            ledger = EvidenceLedger(root / "evidence" / "ledger.jsonl")
            kb = LocalKnowledgeBase(knowledge_dir, ledger=ledger)

            kb.ingest()
            results = kb.retrieve("Which policy requires evidence citations?", top_k=2)

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].source, str(knowledge_dir / "policy.md"))
            self.assertIn("lines", results[0].span)
            self.assertEqual(len(results[0].content_hash), 64)
            self.assertGreater(results[0].score, 0)
            self.assertTrue(results[0].evidence_id.startswith("ev_"))

    def test_knowledge_base_does_not_ingest_memory_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            memory_dir = root / "memory"
            knowledge_dir.mkdir()
            memory_dir.mkdir()
            (knowledge_dir / "kb.md").write_text(
                "Knowledge base documents are external or project source material.",
                encoding="utf-8",
            )
            (memory_dir / "user.jsonl").write_text(
                '{"partition":"user","key":"secret","value":"memory-only-vector-database-fact"}\n',
                encoding="utf-8",
            )
            ledger = EvidenceLedger(root / "evidence" / "ledger.jsonl")
            kb = LocalKnowledgeBase(knowledge_dir, ledger=ledger)

            results = kb.retrieve("memory-only-vector-database-fact", top_k=3)

            self.assertEqual(results, [])

    def test_hybrid_retrieval_supports_cross_lingual_semantic_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            target = knowledge_dir / "hea.md"
            target.write_text(
                "High entropy alloy strength depends on solid solution strengthening, grain size, "
                "precipitation hardening, dislocation density, phase stability, microstructure, and processing history.",
                encoding="utf-8",
            )
            (knowledge_dir / "rag.md").write_text(
                "BM25 keyword search works best when query and document share exact lexical terms.",
                encoding="utf-8",
            )
            kb = LocalKnowledgeBase(knowledge_dir)

            keyword_results = kb.retrieve("高熵合金强度受哪些微观组织因素影响", top_k=3, methods=["keyword"])
            hybrid_results = kb.retrieve(
                "高熵合金强度受哪些微观组织因素影响",
                top_k=3,
                methods=["bm25", "semantic", "graph"],
            )

            self.assertFalse(any(result.source == str(target) for result in keyword_results))
            self.assertGreaterEqual(len(hybrid_results), 1)
            self.assertEqual(hybrid_results[0].source, str(target))
            self.assertIn("semantic", hybrid_results[0].score_breakdown)
            self.assertIn("graph", hybrid_results[0].retrieval_method)

    def test_graph_retrieval_can_be_used_as_standalone_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            target = knowledge_dir / "workflow.md"
            target.write_text(
                "The evidence gate connects failed audits to retrieve. Retrieve then returns support to reason.",
                encoding="utf-8",
            )
            (knowledge_dir / "memory.md").write_text(
                "Memory proposals require approval before persistent storage changes.",
                encoding="utf-8",
            )
            kb = LocalKnowledgeBase(knowledge_dir)

            results = kb.retrieve("gate retrieve support", top_k=2, methods=["graph"])

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].source, str(target))
            self.assertEqual(results[0].retrieval_method, "graph")
            self.assertGreater(results[0].score_breakdown["graph"], 0)

    def test_wiki_fallback_can_be_switched_on_and_combined(self):
        class FakeWikiSource:
            def retrieve(self, query, *, top_k):
                return [
                    {
                        "title": "Hierarchical navigable small world",
                        "url": "https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world",
                        "summary": "HNSW is a graph-based approximate nearest neighbor search algorithm.",
                        "score": 0.68,
                    }
                ][:top_k]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "memory.md").write_text(
                "Memory gate policies are unrelated to approximate nearest neighbor search.",
                encoding="utf-8",
            )
            ledger = EvidenceLedger(root / "evidence" / "ledger.jsonl")
            kb = LocalKnowledgeBase(knowledge_dir, ledger=ledger, wiki_source=FakeWikiSource())

            local_only = kb.retrieve("HNSW approximate nearest neighbor graph", top_k=3, methods=["bm25"])
            with_wiki = kb.retrieve(
                "HNSW approximate nearest neighbor graph",
                top_k=3,
                methods=["bm25", "wiki"],
                wiki_top_k=1,
            )

            self.assertFalse(any(result.source.startswith("https://en.wikipedia.org/") for result in local_only))
            self.assertTrue(any(result.source.startswith("https://en.wikipedia.org/") for result in with_wiki))
            wiki_result = next(result for result in with_wiki if result.source.startswith("https://en.wikipedia.org/"))
            self.assertEqual(wiki_result.retrieval_method, "wiki")
            self.assertTrue(wiki_result.evidence_id.startswith("ev_"))

    def test_wikipedia_source_sends_user_agent_and_records_errors(self):
        captured = {}

        class FakeResponse:
            def read(self):
                return json.dumps(
                    {
                        "query": {
                            "search": [
                                {
                                    "title": "Hierarchical navigable small world",
                                    "snippet": "HNSW is a graph-based approximate nearest neighbor algorithm.",
                                }
                            ]
                        }
                    }
                ).encode("utf-8")

        def fake_urlopen(request, *, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return FakeResponse()

        source = WikipediaKnowledgeSource(urlopen=fake_urlopen, timeout_seconds=7)
        results = source.retrieve("HNSW approximate nearest neighbor graph", top_k=1)

        self.assertEqual(results[0]["title"], "Hierarchical navigable small world")
        self.assertIn("wikipedia.org/w/api.php", captured["url"])
        self.assertEqual(captured["timeout"], 7)
        self.assertTrue(
            any(key.lower() == "user-agent" and value for key, value in captured["headers"].items())
        )

        def failing_urlopen(request, *, timeout):
            raise RuntimeError("blocked by provider")

        failing_source = WikipediaKnowledgeSource(urlopen=failing_urlopen)
        self.assertEqual(failing_source.retrieve("HNSW", top_k=1), [])
        self.assertEqual(failing_source.diagnostics[-1]["status"], "error")
        self.assertIn("blocked by provider", failing_source.diagnostics[-1]["message"])

    def test_long_term_memory_requires_evidence_and_never_writes_shared_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = EvidenceLedger(root / "evidence.jsonl")
            policy = GatePolicy(
                workspace_root=root,
                min_evidence_by_risk={"medium": 1},
                approval_required_actions={"write_memory"},
            )
            store = LongTermMemoryStore(root / "memory", gate_policy=policy)

            missing = store.write(
                partition="project",
                key="principle",
                value="Use evidence-first reasoning.",
                evidence=[],
            )
            self.assertEqual(missing.decision.status, "interrupt")
            self.assertEqual(store.read("project", "principle"), None)

            evidence = ledger.record(
                source_type="user",
                uri="conversation",
                locator="turn:1",
                content="User approved project memory.",
                summary="User approved project memory.",
                confidence=1.0,
                used_for=["memory:project"],
            )
            written = store.write(
                partition="project",
                key="principle",
                value="Use evidence-first reasoning.",
                evidence=[evidence],
                approved_by="human",
            )
            self.assertEqual(written.decision.status, "allow")
            self.assertEqual(store.read("project", "principle"), "Use evidence-first reasoning.")

            shared = store.write(
                partition="shared",
                key="global_rule",
                value="Do not mutate shared memory.",
                evidence=[evidence],
                approved_by="human",
            )
            self.assertEqual(shared.decision.status, "deny")
            self.assertEqual(store.read("shared", "global_rule"), None)

    def test_long_term_memory_rejects_path_traversal_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = GatePolicy(workspace_root=root, approval_required_actions=[])
            store = LongTermMemoryStore(root / "memory", gate_policy=gate)

            for partition in ["../outside", "C:\\outside", "shared/..", ""]:
                with self.subTest(partition=partition), self.assertRaises(ValueError):
                    store.list_partition(partition)

    def test_self_evolution_generates_proposals_without_mutating_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills_dir = root / "skills"
            proposals_dir = root / "evolution" / "proposals"
            skills_dir.mkdir()
            (skills_dir / "minimal-change").mkdir()
            skill_file = skills_dir / "minimal-change" / "SKILL.md"
            skill_file.write_text("---\nname: minimal-change\ndescription: keep changes small\n---\n", encoding="utf-8")
            ledger = EvidenceLedger(root / "evidence.jsonl")
            evidence = ledger.record(
                source_type="test",
                uri="golden",
                locator="case:minimal-change",
                content="The agent over-edited unrelated files.",
                summary="Over-edit failure.",
                confidence=0.87,
                used_for=["evolution:minimal-change"],
            )

            engine = SelfEvolutionEngine(proposals_dir=proposals_dir, skills_dir=skills_dir)
            proposal = engine.propose_skill_update(
                skill_name="minimal-change",
                rationale="Tighten scope checks after an over-edit failure.",
                evidence=[evidence],
                suggested_change="Require an explicit unrelated-file scan before editing.",
            )

            self.assertTrue(proposal.path.exists())
            self.assertEqual(skill_file.read_text(encoding="utf-8"), "---\nname: minimal-change\ndescription: keep changes small\n---\n")
            data = json.loads(proposal.path.read_text(encoding="utf-8"))
            self.assertEqual(data["target"], "skills/minimal-change/SKILL.md")
            self.assertEqual(data["status"], "proposed")
            self.assertEqual(data["evidence_ids"], [evidence.id])


if __name__ == "__main__":
    unittest.main()
