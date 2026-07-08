# DeepReason Agents Framework

DeepReason Agents Framework is a developer framework for building evidence-first, stateful, skillized, heavy-reasoning multi-agent systems. It provides local implementations for evidence ledgers, gates, knowledge retrieval, memory partitions, self-evolution proposals, skill metadata, workflow graphs, and a deterministic coordinator.

The intended runtime is LangGraph OSS plus Deep Agents. The default template stays runnable without API keys; enable Deep Agents in `agent.yaml` after installing optional dependencies.

Run the local test suite:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

Use the CLI without installing the package:

```powershell
$env:PYTHONPATH='src'
python -m reasoning_agent_template chat "What constraints does this template enforce?"
python -m reasoning_agent_template skills
python -m reasoning_agent_template test
python -m reasoning_agent_template web
```

After installing the package, the same commands are available through:

```powershell
reasoning-agent chat "What constraints does this template enforce?"
reasoning-agent skills
reasoning-agent test
reasoning-agent web
deepreason-agent web
```

Open the local web chat and debug console:

```powershell
$env:PYTHONPATH='src'
python -m reasoning_agent_template web --host 127.0.0.1 --port 8765
```

The web console includes conversation, multi-agent runtime status, state-machine trace, evidence ledger, RAG results, gate decisions, memory policy, skill loading, event stream, and raw JSON.

Run a real DeepSeek smoke test without storing secrets in the repo:

```powershell
$env:PYTHONPATH='src'
$env:DEEPSEEK_API_KEY='<your key>'
python -m reasoning_agent_template deepseek-smoke
Remove-Item Env:\DEEPSEEK_API_KEY
```
