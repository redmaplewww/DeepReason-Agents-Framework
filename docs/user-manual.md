# DeepReason Agents Framework 用户手册与学习手册

这份手册面向两类用户：

- **使用者**：想打开 Web 调试台，理解每个按钮、面板、状态和输出代表什么。
- **开发者**：想基于 DeepReason Agents Framework 搭建自己的重推理 Agent，理解每个模块为什么存在、怎么扩展、怎么验证。

建议阅读顺序：

1. 先看第 1-3 章，理解整体界面和运行方式。
2. 再看第 4-8 章，理解多 Agent、工作流、结构化交付、证据、RAG、门禁和记忆。
3. 最后看第 9-14 章，学习如何基于框架做开发和排错。

## 1. 框架的核心定位

DeepReason Agents Framework 不是普通聊天机器人模板，而是一个“重推理 Agent 底层框架”。它解决的问题是：

- 简单任务不必过度工程化，可以快速回答。
- 中等任务可以启用有限证据和轻量审查。
- 困难、学术、事实判断、高风险任务必须进入证据、审查、门禁和验证流程。
- Agent 的工作过程必须可观测，节点之间的交付必须结构化，失败后能知道是哪一步坏了。
- 配置、工作流、Agent 角色、技能和自进化都要能被人工审查后再应用。

框架的主线是：

```text
用户目标
  -> Coordinator 路由
  -> Planner 规划
  -> Retriever 检索证据
  -> Reasoner 生成草案
  -> Critic 审计证据和质量
  -> Gate 决定 allow / interrupt / deny
  -> Verify 验证一致性
  -> Memory 生成沉淀提案
  -> Respond 输出最终回答和调试遥测
```

## 2. Web 调试台总览

启动：

```powershell
$env:PYTHONPATH='src'
python -m reasoning_agent_template web --host 127.0.0.1 --port 8767
```

打开：

```text
http://127.0.0.1:8767/
```

![DeepReason 调试控制台总览](assets/deepreason-console-overview.png)

页面分成四层：

| 层级 | 功能区 | 作用 |
|---|---|---|
| 顶部栏 | 标题、运行时、DeepSeek 状态、刷新按钮 | 确认当前项目、工作目录和 LLM 配置状态。 |
| 左侧对话区 | 普通对话、搭建助手、消息流、输入框 | 与 Agent 对话，或用自然语言生成 Agent/Workflow 草稿。 |
| 右侧监控区 | 状态条、进度提示、多 Agent、工作流图 | 观察当前是谁在工作、工作流走到哪里、是否启用证据系统。 |
| 详情区 | 证据、路由/RAG/门禁/记忆/事件/原始 JSON | 诊断每一步的真实输入、输出、证据和 gate 决策。 |

## 3. 顶部栏和对话区

### 3.1 顶部栏

| 元素 | 显示内容 | 你应该怎么用 |
|---|---|---|
| `Agent 调试控制台` | 当前 Web 控制台标题 | 用于确认你在本地调试台，而不是普通网页。 |
| `runtimeLine` | runtime agent 名称和 workspace 路径 | 确认服务是否加载了正确项目。路径错了时，知识库和配置也会错。 |
| `DeepSeek 必须` | 当前系统要求真实 LLM 输出 | 如果缺少 key，普通对话会报明确错误，而不是用模板假回答。 |
| `刷新` | 重新读取 `/api/status`、workflow spec、agents spec | 当你手动改配置或后台运行完成后，用它同步 UI。 |

### 3.2 对话区

| 元素 | 作用 |
|---|---|
| `普通对话` | 让 Coordinator 自主判断任务难度，并选择普通路径、软证据路径或严格证据路径。 |
| `搭建助手` | 让 Configurator 根据自然语言生成多 Agent 和工作流草稿。 |
| `runId` | 显示当前运行 ID 和 thread ID。thread ID 用于短期记忆隔离。 |
| 消息流 | 展示用户消息和最终回答。证据不直接塞进回答正文，而是在证据栏以参考文献索引展示。 |
| 输入框 | 普通对话时输入任务；搭建助手模式下输入你想创建的 Agent 架构。 |

普通对话中也可以临时触发搭建助手：

```text
/配置 ...
/搭建 ...
#配置 ...
#搭建 ...
配置助手: ...
```

为什么这样设计：

- 普通对话保持自然，不要求用户手动选择工作流。
- 搭建助手作为顶层能力存在，不藏在“多 Agent”或“工作流”模块里。
- 右侧编辑器只保留人工干预、审查、保存、生成提案和批准应用能力。

## 4. 顶部状态条和进度提示

状态条包含五个核心指标：

| 指标 | 来源 | 含义 |
|---|---|---|
| 多 Agent 运行时 | `payload.status` | 当前任务处于 ready、running、completed、failed 等状态。 |
| 当前节点 | `payload.workflow.current` | 当前工作流执行到哪个节点。 |
| 路由 / 工作流 | `payload.routing.difficulty` + `payload.workflow.variant` | Coordinator/Reviewer 判断的难度和工作流类型。 |
| 证据系统 | `mode / strictness / status / count` | 证据是否启用、严格程度、是否合格、证据数量。 |
| RAG / 外部证据 | `rag.count / external_evidence.count` | 本地知识库和外部证据检索结果数量。 |

进度提示区显示：

- 当前激活 Agent。
- 当前节点或搭建目标。
- 当前动作说明，例如“正在检索 RAG”“正在生成草稿”“本轮运行已完成”。

为什么这样设计：

- 重推理任务很容易“看起来瞬间完成”，用户不知道 Agent 是否真的工作。
- 状态条和进度提示让每个阶段变成可观察过程。
- 你可以快速判断：任务是不是被错分为简单任务，证据系统有没有启动，RAG 有没有结果。

## 5. 多 Agent 功能区

多 Agent 面板展示当前加载的 Agent 角色。默认角色来自：

```text
configs/agents/default.agents.json
```

### 5.1 默认 Agent 角色

| Agent | 中文名 | 职责 | 典型节点 |
|---|---|---|---|
| `coordinator` | 协调器 | 接收输入、调度流程、输出最终回答 | `intake`, `act_or_answer`, `respond` |
| `planner` | 规划器 | 生成检索和推理计划 | `plan` |
| `retriever` | 检索器 | 本地 RAG、外部搜索、证据归一化 | `retrieve` |
| `reasoner` | 推理器 | 基于上下文和证据生成答案草案 | `reason` |
| `critic` | 审查器 | 证据审计、门禁、验证一致性 | `evidence_audit`, `gate`, `verify` |
| `memory` | 记忆管理 | 短期上下文整理、长期记忆候选 | `consolidate` |
| `reviewer` | 路由复核 | 复核 Coordinator 难度和证据判断 | `intake -> plan` 等关键转交 |
| `configurator` | 配置助手 | 生成 Agent/Workflow 草稿 | 顶层搭建模式 |
| `code_modifier` | 代码修改器 | 只应用已批准的代码/配置修改 | proposal apply |

### 5.2 Agent 卡片怎么看

每张卡片通常包含：

- 显示名。
- 状态 badge。
- Agent ID。
- protected 标记。
- 描述。
- 最近事件或当前阶段。

protected 表示底座角色，默认不能直接删除。原因是这些角色组成了最小可运行链路：路由、计划、检索、推理、审查、门禁、响应。

### 5.3 Agent 编辑器

点击 `编辑 Agent` 后可以修改：

| 字段 | 含义 | 使用建议 |
|---|---|---|
| Agent ID | 稳定 ID | 用英文小写和下划线，避免频繁改名。 |
| 显示名称 | UI 上展示的名字 | 中文即可，便于非技术用户理解。 |
| 模型角色 | planner / worker / critic / grader | 不同角色可以映射不同模型。 |
| 描述 | 这个 Agent 的边界 | 写清楚“负责什么”和“不负责什么”。 |
| 职责 | 每行一条责任 | 越具体越好，避免一个 Agent 什么都管。 |
| 工具 | 每行一个工具名 | 只给它真正需要的工具。 |
| 记忆权限 | 如 `short_term:read` | 长期写入要谨慎，必须经过 gate。 |
| 绑定 Workflow 节点 | 每行一个节点 ID | 让 UI 和运行时知道这个 Agent 出现在哪些节点。 |
| 权限 JSON | 工具和路径权限 | 用于限制高风险动作。 |
| 交付契约 JSON | Agent 对外输出格式 | 让下游节点能稳定消费。 |

编辑流程：

```text
编辑 Agent
  -> 修改字段
  -> 保存草稿
  -> 生成提案
  -> 审查 diff
  -> 批准应用
```

为什么不直接保存到运行配置：

- Agent 架构变化会影响整个推理链路。
- proposal 让修改可审查、可回滚、可测试。
- code_modifier 只能应用已批准提案，避免普通对话 Agent 自己改系统。

## 6. 工作流点线图

工作流图来自：

```text
configs/workflows/default.workflow.json
```

图中节点代表工作阶段，边代表结构化转交。它不是固定线性链路，而是包含分支和局部回路：

- `plan -> retrieve`: 需要证据时进入检索。
- `plan -> reason`: 简单任务可跳过检索。
- `evidence_audit -> retrieve`: 证据不足时补证。
- `gate -> retrieve`: 门禁阻断时补证或修正。
- `verify -> reason`: 验证失败时回到推理。
- `consolidate -> evidence_audit`: 沉淀提案也要重新审计。

点击节点会显示该节点的真实输入、真实输出、过程摘要、handoff 和 Agent/事件。

点击连线会显示上游输出、下游输入、审查、门禁、交付契约和相关 Agent。

![工作流节点详情](assets/deepreason-workflow-detail.png)

## 7. 节点与节点之间的结构化输出

这是框架最重要的设计之一。

### 7.1 每个节点都有两个层次的契约

第一层是设计时契约：

| 字段 | 含义 |
|---|---|
| `input_contract` | 这个节点期望从上游收到什么。 |
| `output_contract` | 这个节点必须向下游交付什么。 |
| `work` | 这个节点实际负责的工作。 |
| `handler` | 运行时调用哪个处理器。 |
| `gate_policy` | 当前节点是否需要特殊门禁。 |

第二层是运行时产物：

| 字段 | 含义 |
|---|---|
| `actual_input` | 本轮运行中节点真实收到的数据。 |
| `actual_output` | 本轮运行中节点真实产出的数据。 |
| `process` | 可审计的过程摘要和关键判断。 |
| `handoff` | 传给下游的结构化交付包。 |

### 7.2 为什么要结构化输出

如果每个 Agent 只用自然语言把结果丢给下一个 Agent，会出现四类问题：

1. **不可审计**：不知道上游到底交付了什么。
2. **不可替换**：换一个 Agent 或模型，下游就可能读不懂。
3. **不可门禁**：门禁只能看一段散文，无法检查证据数量、风险等级、审批状态。
4. **不可调试**：任务失败时，不知道是路由错、检索错、推理错还是审查错。

结构化输出的意义是：

- 让每一步都有明确输入和输出。
- 让 Reviewer、Critic、Gate 能检查具体字段。
- 让 UI 可以展示真实交付，而不是只显示“已完成”。
- 让 workflow spec 可以被动态编辑，节点也可以替换。
- 让失败恢复、checkpoint、测试和回归评估变得可行。

### 7.3 一个标准节点交付示例

```json
{
  "actual_input": {
    "question": "高熵合金的强度受哪些因素影响？",
    "routing": {
      "difficulty": "hard",
      "workflow": "evidence_strict"
    }
  },
  "actual_output": {
    "claim_outline": [
      "固溶强化",
      "析出强化",
      "晶粒尺寸",
      "相稳定性"
    ],
    "required_evidence": 2
  },
  "process": [
    "任务被判断为材料科学解释问题，需要证据支持。",
    "Planner 要求 Retriever 同时使用本地 RAG 和外部论文来源。"
  ],
  "handoff": {
    "to": "retrieve",
    "payload": "query + evidence policy + risk level"
  }
}
```

### 7.4 边上的结构化交付

边不仅是箭头，还可以有自己的契约：

| 字段 | 作用 |
|---|---|
| `condition` | 什么时候走这条边。 |
| `handoff_contract` | 上游必须交给下游的 payload。 |
| `gate_policy` | 转交时是否需要门禁。 |
| `planner_contract` | Planner 对这条边的规划要求。 |
| `reviewer_required` | 是否需要 Reviewer 审查。 |

例如 `intake -> plan` 默认要求：

```json
{
  "payload": "routing + normalized_goal"
}
```

它的意义是：Planner 不应该直接猜用户意图，而应该消费 Coordinator 已经归一化的目标、难度、风险和证据策略。

## 8. 详情 Tab 完整说明

### 8.1 证据

显示：

- 证据模式：idle、optional、soft、strict 等。
- 严格程度：none、soft、strict。
- 风险等级。
- 合格证据 ID。
- 证据策略原因。
- 参考文献索引。
- 每条 EvidenceItem 的 source、uri、locator、confidence、summary。

怎么用：

- 如果一个学术或技术问题没有证据，先看这里是否进入 evidence mode。
- 如果证据数量够但回答仍被 gate 打回，检查 confidence 和 qualified evidence。
- 回答正文不直接堆证据，证据在这里以参考文献形式索引。

### 8.2 路由/审查

显示：

- Coordinator 的难度判断、工作流判断、confidence。
- Coordinator 的原始 decision JSON。
- Reviewer 的状态、发现和复核 decision。

怎么用：

- 如果系统没有启动证据，先看 difficulty 是否被判断为 routine。
- 如果 Coordinator 过松，Reviewer 应该能升级难度或证据要求。
- 调参时重点看这里，而不是只看最终回答。

### 8.3 RAG

显示：

- 查询框。
- 检索方法开关：BM25、语义、Graph、Wiki。
- 当前 query。
- 结果数量。
- diagnostics。
- 每条 chunk 的 source、span、score、evidence_id、retrieval_method、score_breakdown、text。

![RAG 检索实验面板](assets/deepreason-rag-lab.png)

怎么用：

- BM25：看关键词命中。
- 语义：看中英文别名、近义表达、概念匹配。
- Graph：看局部关系扩展和相邻概念。
- Wiki：本地知识不足时的 fallback。

如果召回差：

1. 检查知识库是否放在 `knowledge/`。
2. 检查文档是否是 `.md`、`.txt`、`.json`。
3. 检查 query 是否有中英文别名。
4. 切换单一方法，看是哪一类检索弱。
5. 跑 `rag-eval` 看 Recall@K。

### 8.4 外部证据

显示：

- 尝试来源。
- 外部 query。
- 相关性过滤后的结果。
- diagnostics 和 provider 错误。

怎么用：

- 学术、事实、时效问题应尽量有外部证据。
- 如果外部结果为空，不一定是没搜索，可能是结果被相关性过滤掉。
- provider 失败会出现在 diagnostics，不应该静默消失。

### 8.5 工作流

显示：

- 工作流状态。
- 当前节点。
- 工作流类型。
- checkpoint 列表。
- 工作流编辑器。
- 当前选中节点/边的工作过程与结果。
- 所有节点摘要。
- 控制流边列表。

怎么用：

- 看任务具体走了哪些节点。
- 看某个节点是否真的执行，还是被跳过。
- 点击节点查看真实输入输出。
- 点击边查看交付过程和门禁。

### 8.6 门禁

显示每条 GateDecision：

- `gate_id`
- 状态：allow / interrupt / deny
- 风险等级。
- required_evidence。
- reasons。

怎么用：

- 高风险动作没有证据时应 interrupt 或 deny。
- 修改文件、写长期记忆、更新 skill、执行命令等动作需要单独 gate。
- 如果你认为系统太松或太严，应改 gate policy 和测试，而不是只改提示词。

### 8.7 记忆

显示：

- 分区列表。
- 只读分区。
- 当前 thread。
- 短期记忆轮数。
- 长期记忆条数。
- 待沉淀内容。
- 记忆硬边界。
- 短期记忆内容。
- 长期记忆内容。
- 长期写入门禁结果。

记忆与知识库的区别：

| 类型 | 存什么 | 存在哪里 | 怎么进入 |
|---|---|---|---|
| 短期记忆 | 当前 thread 对话上下文 | runtime/thread state | 自动记录，按 thread 隔离。 |
| 长期记忆 | 用户偏好、项目事实、流程经验 | `memory/` 分区 | 需要证据和 write_memory gate。 |
| 知识库 | 文档、论文、规范、API、数据集 | `knowledge/` | 用户放入文件或导入。 |

为什么要分开：

- 知识库是外部材料，不代表用户偏好。
- 长期记忆是系统对用户/项目的沉淀，必须审查。
- 短期记忆只是当前会话上下文，不应该污染长期知识。

### 8.8 技能

显示已加载技能包，例如：

- `project-intake`
- `evidence-first`
- `state-gates`
- `knowledge-rag`
- `memory-consolidation`
- `minimal-change`
- `self-evolution`
- `configurator`
- `testing-verification`

怎么用：

- 强约束优先写成 skill，而不是塞进一个巨型 system prompt。
- 技能适合沉淀“项目通用规则”“证据标准”“最小修改原则”“测试验收流程”。
- 新技能放在 `skills/<name>/SKILL.md`，再在 `agent.yaml` 启用。

### 8.9 事件

显示 runtime events：

- Agent 名称。
- event kind。
- message。
- time。
- duration_ms。
- evidence_id。

怎么用：

- 判断某个 Agent 是否真的被调用。
- 判断某一步耗时是否异常。
- 排查“瞬间跳过”的节点。

### 8.10 原始 JSON

展示完整 `/api/status` payload。

怎么用：

- UI 没展示的字段，可以在这里直接看。
- 写测试和调试前端时，以 Raw JSON 为准。
- 如果你要对接外部 dashboard，可以先参考这里的数据结构。

## 9. 工作流编辑器

工作流编辑器的目标不是做低代码玩具，而是让工作流 spec 可视化、可审查、可提案化。

### 9.1 按钮说明

| 按钮 | 作用 |
|---|---|
| `编辑工作流` | 进入编辑模式。 |
| `刷新 Spec` | 重新读取当前 spec/draft。 |
| `保存草稿` | 把当前编辑内容保存为 draft，并触发校验。 |
| `生成提案` | 根据 draft 生成 diff preview 和 proposal。 |
| `批准应用` | 调用 code_modifier 应用已批准 proposal。 |
| `添加节点` | 添加一个通用 passthrough 节点。 |
| `添加领域审查节点` | 在 reason 和 evidence_audit 之间插入 domain_review 节点。 |
| `添加连线` | 添加一条可编辑边。 |
| `删除所选` | 删除当前选中的非保护节点或边。 |

### 9.2 节点字段

| 字段 | 为什么重要 |
|---|---|
| 节点 ID | 所有边、事件、测试、handler 都依赖稳定 ID。 |
| 显示名称 | 给用户看的中文标签。 |
| 激活 Agent | 决定哪个 Agent 对此节点负责。 |
| Handler 类型 | 决定运行时调用 builtin 还是 plugin_tool。 |
| Handler | 决定具体执行逻辑。 |
| 检查点 | 长任务恢复和审计的稳定位置。 |
| 描述 | 给用户理解节点职责。 |
| 工作内容 | 给 Agent 理解节点任务。 |
| 输入契约 | 定义上游必须交付什么。 |
| 输出契约 | 定义下游可以依赖什么。 |
| 门禁策略 JSON | 节点级风险、证据和审批规则。 |
| UI JSON | 前端展示和 builder 层的扩展元数据。 |

### 9.3 连线字段

| 字段 | 为什么重要 |
|---|---|
| 连线 ID | 稳定识别一条边。 |
| 类型 | 区分主流程、分支、重试、修订和循环。 |
| 起点/终点 | 决定控制流结构。 |
| 条件/标签 | 解释为什么走这条边。 |
| 需要 Reviewer | 关键转交是否必须复核。 |
| 交付契约 JSON | 定义边上的 payload。 |
| 门禁策略 JSON | 在节点之间加一道门。 |
| 规划契约 JSON | 要求 Planner 对这条边提供特定计划或输出。 |

### 9.4 校验和提案

保存草稿会校验：

- 节点 ID 是否重复。
- 边是否指向不存在的节点。
- protected 节点是否被删除。
- handler 是否已知。
- 工作流是否断开。

生成提案会得到：

- proposal id。
- draft hash。
- diff preview。
- modified files。
- test command。
- gate decision。

这套流程的意义是：允许用户大胆编辑工作流，但不让修改绕过审查直接进入运行态。

## 10. 搭建助手

搭建助手是一个顶层 Agent，它负责把自然语言目标转换为：

- Agent draft。
- Workflow draft。
- 规模判断。
- 推荐节点和角色。
- 基础交付契约。

它不负责：

- 直接应用修改。
- 绕过人工审查。
- 替代 code_modifier 写代码。
- 更新长期记忆或证据 ledger。

示例：

```text
/搭建 我要做一个材料科学论文审查 Agent。
需要判断论文贡献、证据强度和实验可复现性。
简单论文走轻量流程，复杂论文加入 reviewer 和领域审查节点。
```

你应该做的下一步：

1. 到 `多 Agent` 面板看生成了哪些角色。
2. 到 `工作流` 面板看生成了哪些节点和边。
3. 修正过重或过轻的设计。
4. 保存草稿。
5. 生成提案。
6. 审查 diff。
7. 批准应用。

## 11. RAG 与知识库

### 11.1 知识库初始化

把文件放入：

```text
knowledge/
```

支持：

- `.md`
- `.txt`
- `.json`

运行时会：

1. 扫描文件。
2. 按段落和大小切 chunk。
3. 记录 source path。
4. 记录 line span。
5. 计算 content hash。
6. 建立本地检索索引。
7. 查询时返回 score、score_breakdown、evidence_id。

### 11.2 检索方式

| 方法 | 适合 | 局限 |
|---|---|---|
| BM25 | 精确术语、关键词、专有名词 | 对改写和跨语言较弱。 |
| 语义 | 近义表达、中英文别名、概念匹配 | 当前是本地确定性近似，不是真 embedding。 |
| Graph | 概念关系、相邻术语扩展 | 可能过度扩展，适合辅助而非唯一依据。 |
| Wiki | 本地知识不足时 fallback | 外部来源质量不稳定，需结合 gate。 |

### 11.3 召回评测

```powershell
$env:PYTHONPATH='src'
python -m reasoning_agent_template rag-eval --cases configs/rag_benchmark_cases.json --report docs/rag-benchmark-latest.md
```

评测看：

- Recall@1。
- Recall@3。
- Recall@5。
- 哪些问题命中错误文档。
- 哪些方法的 score_breakdown 不合理。

## 12. 证据系统与门禁

### 12.1 EvidenceItem

证据项包含：

- `id`
- `source_type`
- `uri`
- `locator`
- `content_hash`
- `summary`
- `confidence`
- `collected_at`
- `used_for`

为什么要有 hash：

- 防止引用漂移。
- 让同一段证据可复核。
- 让证据 ledger 能去重和追踪。

为什么回答里不直接输出 evidence id：

- 用户阅读答案时不应被内部 ID 打断。
- 证据栏提供参考文献索引，更适合审查。
- 内部 evidence id 仍保留给系统调试和测试。

### 12.2 GateDecision

门禁结果包含：

- `gate_id`
- `risk_level`
- `status`
- `reasons`
- `required_evidence`
- `approved_by`
- `state_snapshot_id`

状态含义：

| 状态 | 含义 |
|---|---|
| `allow` | 可继续执行或回答。 |
| `interrupt` | 需要补证、人工审批或更多信息。 |
| `deny` | 明确拒绝高风险或越权动作。 |

哪些动作应该 gate：

- 高风险回答。
- 写文件。
- 写长期记忆。
- 更新 skill。
- 应用 workflow/agent proposal。
- 执行命令。
- 外部调用。

## 13. 如何基于框架开发自己的 Agent

### 13.1 最小开发路径

1. 改 `agent.yaml`：项目身份、模型、知识库、门禁、技能。
2. 放资料到 `knowledge/`。
3. 用搭建助手生成初版 Agent 和 Workflow。
4. 人工审查草稿。
5. 保存草稿、生成提案、批准应用。
6. 写测试。
7. 跑测试和 RAG 评测。
8. 打开 Web 调试台验证真实任务。

### 13.2 新增 Agent

改：

```text
configs/agents/default.agents.json
```

重点写：

- 这个 Agent 为什么存在。
- 它负责哪些节点。
- 它能用哪些工具。
- 它能读写哪些记忆。
- 它对下游交付什么结构。

### 13.3 新增工作流节点

改：

```text
configs/workflows/default.workflow.json
```

新增节点时必须回答：

- 它解决哪个明确问题？
- 上游交付什么？
- 下游需要什么？
- 谁负责？
- 是否需要 checkpoint？
- 是否需要 gate？
- 失败后回到哪里？

### 13.4 新增 Skill

新建：

```text
skills/<skill-name>/SKILL.md
```

一个好 skill 应该包含：

- 触发条件。
- 执行流程。
- 强约束。
- 禁止事项。
- 验收标准。
- 失败时如何处理。

### 13.5 新增工具或外部系统

建议做成薄 adapter：

- 工具输入输出要结构化。
- 高风险工具必须进 gate。
- 工具结果要能转成 evidence 或 event。
- 不要把第三方框架大段重写进内核。

## 14. 学习路径

### 初级：会使用

目标：

- 会启动 Web。
- 会普通对话。
- 会看证据栏和 RAG 栏。
- 会知道任务为什么被判简单或困难。

练习：

```text
问：你是谁？
观察：证据系统不应强制启动。

问：高熵合金的强度受哪些因素影响？
观察：RAG/证据应启动，证据栏应出现参考文献。
```

### 中级：会调试

目标：

- 会点击节点和边。
- 会看真实输入和真实输出。
- 会判断哪个 Agent 工作了。
- 会用 RAG 面板检查召回质量。

练习：

```text
点击 plan 节点，看它是否产生证据策略。
点击 plan -> retrieve 边，看为什么进入检索。
切换 RAG 方法，比较 BM25、语义、Graph 的结果。
```

### 高级：会开发

目标：

- 会新增 Agent。
- 会新增工作流节点。
- 会写 handoff contract。
- 会写 gate policy。
- 会新增 skill。
- 会补测试。

练习：

```text
新增 domain_review 节点：
reason -> domain_review -> evidence_audit

要求：
domain_review 由 critic 负责；
输入是 answer_draft + evidence_context；
输出是 domain_review_notes；
边上 reviewer_required=true。
```

## 15. 常见问题

### 为什么简单问题不走证据系统？

因为证据系统是成本较高的严肃流程。简单身份、闲聊、格式转换、低风险问题如果强制检索，会让 Agent 变慢、变笨、用户体验变差。框架目标是“按任务复杂度自适应”，不是所有问题都重流程。

### 为什么学术问题必须尽可能找证据？

学术和技术判断通常具有事实性、可验证性和误导风险。没有证据时，模型容易给出流畅但不可靠的答案。证据系统会迫使 Agent 先检索、再推理、再审计、再回答。

### 为什么要有 Reviewer？

Coordinator 可能把复杂任务误判为简单任务。Reviewer 是第二道判断，用来纠正过松的路由、风险和证据策略。

### 为什么搭建助手不能直接应用修改？

因为搭建助手也是 LLM。它可以生成草稿，但不能绕过人工审查直接改工作流或 Agent 权限。真正应用必须走 proposal 和 code_modifier。

### 为什么长期记忆不能自动写？

长期记忆会改变未来行为。如果把一次错误对话自动沉淀，就会长期污染 Agent。因此长期记忆必须有证据、有 gate、有审查。

### 为什么 code_modifier 只能改代码？

这是职责隔离。code_modifier 不参与聊天、不检索证据、不写记忆，只负责应用已批准修改。这样可以降低“会话 Agent 自己改自己”的风险。

## 16. 推荐检查清单

每次新增功能前问自己：

- 这是新 Agent、新节点、新工具、新 skill，还是只需要改配置？
- 这个功能有没有清晰输入和输出？
- 下游节点是否能稳定消费它的输出？
- 是否需要 Reviewer？
- 是否需要 Gate？
- 是否会写文件、记忆、技能或外部系统？
- UI 是否能看到它真的运行了？
- 是否补了测试？

每次完成修改后运行：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
node --check src\reasoning_agent_template\web_static\app.js
```

如果改了 RAG，再运行：

```powershell
python -m reasoning_agent_template rag-eval --cases configs/rag_benchmark_cases.json --report docs/rag-benchmark-latest.md
```
