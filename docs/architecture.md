# DeepReason Agents Framework Architecture

DeepReason Agents Framework is local-first. The deterministic coordinator lets tests and starter projects run without API keys, while `runtime.create_deep_agent_runtime` can opt into Deep Agents when `runtime.prefer_deepagents` is true and the dependency is installed.

The OpenClaude reference is distilled as architecture, not copied source. See
`docs/openclaude-distillation.md` for the keep/plugin/leave-out map and the
first lightweight runtime boundary.

The default graph is not a fixed required path. The normal path is:

`intake -> plan -> reason -> evidence_audit -> gate -> act_or_answer -> verify -> consolidate -> respond`

When evidence is required, `plan` branches to `retrieve`. Strict evidence gaps and blocked gates can take one configured retry edge before the run terminates. The runtime enforces `workflow_max_steps` and `workflow_retry_limit`; graph edges are telemetry-backed execution decisions, not decorative metadata.

Core constraints live in local skills under `skills/`. They are loaded by metadata first and opened only when relevant. Evidence is stored in `evidence/ledger.jsonl`, memory is partitioned under `memory/`, and self-evolution writes proposals under `evolution/proposals/`.

High-risk actions should be mediated through `GatePolicy`. The starter policy checks evidence minimums, human approval requirements, and workspace boundaries.

Optional capabilities should enter through `plugins.PluginLoader`: discovery reads only lightweight manifests, runtime sees manifest-backed `RuntimeTool` proxies, and heavy implementations are imported only when a capability is activated or a tool is actually called.
