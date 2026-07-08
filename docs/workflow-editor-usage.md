# Workflow Editor Usage

本文说明如何使用 Web 调试台里的工作流和多 Agent 编辑能力。它们是人工干预面板：AI 搭建助手可以生成草稿，但最终保存、生成提案和批准应用仍由用户完成。

## 日常调试

1. 打开 `http://127.0.0.1:8767/`。
2. 在左侧对话框提问。
3. 观察顶部状态条：当前 Agent、当前工作流节点、证据模式、RAG 数量和外部证据数量。
4. 点击工作流点线图中的节点或边。
5. 在 `工作流` Tab 的 `工作过程与结果` 区域查看该节点或边的真实输入、真实输出、交付数据、事件和启用 Agent。

## 查看节点详情

- `真实输入`: 节点从上游收到的 payload。
- `真实输出`: 节点产生的结果。
- `思考摘要 / 交付过程`: 可审计的过程摘要、handoff payload 和关键判断。
- `启用 Agent / 事件`: 本阶段涉及的 Agent 和 runtime events。

## 编辑工作流

1. 打开 `工作流` Tab。
2. 点击 `编辑工作流`。
3. 选择图上的节点或边。
4. 节点可编辑：Agent、工作内容、输入契约、输出契约、handler、checkpoint、gate policy。
5. 边可编辑：source、target、condition、handoff contract、gate policy、planner contract、reviewer requirement。
6. 点击 `保存草稿`。
7. 点击 `生成提案`。
8. 审查 diff preview。
9. 确认无误后点击 `批准应用`。

## 工作流字段说明

节点字段：

- `id`: 稳定节点 ID，非保护节点可以新增/删除。
- `label`: UI 显示名。
- `agent`: 执行该节点的 Agent ID。
- `work`: 节点要完成的工作。
- `input_contract`: 上游必须交付的内容。
- `output_contract`: 节点必须输出的内容。
- `handler_kind`: `builtin` 或 `plugin_tool`。
- `handler`: 实际处理器名称。
- `checkpoint`: 是否作为恢复点。
- `gate_policy`: 节点级门禁。

边字段：

- `from` / `to`: 源节点和目标节点。
- `type`: `flow`、`branch`、`retry`、`revise`、`loop` 等。
- `condition`: 走这条边的条件。
- `handoff_contract`: 边上的交付契约。
- `planner_contract`: planner 对该边的规划要求。
- `reviewer_required`: 是否需要 reviewer 审查。

## 重要边界

- 草稿不会改变正在运行的工作流。
- 未知 handler 必须进入 code proposal，不能直接运行。
- protected base nodes 不能从编辑器删除。
- code_modifier 只能改允许范围内的代码、workflow、agents 和测试；不能写 secrets、logs、evidence 或 memory。

## 编辑 Multi-Agent 角色

1. 打开 `多 Agent` 面板。
2. 点击 `编辑 Agent`。
3. 点击一个 Agent 卡片，编辑显示名、描述、模型角色、职责、工具、记忆权限、绑定节点、权限和交付契约。
4. 点击 `添加 Agent` 新增非保护角色。
5. 只对非保护角色使用 `删除所选`。
6. 点击 `保存草稿`，再点击 `生成提案`，审查 diff preview。
7. 确认后点击 `批准应用`。

## 顶层配置助手

`配置助手` 位于左侧对话面板，是自然语言搭建入口。

- 点击 `搭建助手` 可以持续使用 builder 模式。
- 在普通对话中，以 `/配置`、`/搭建`、`#配置`、`#搭建` 开头可以临时触发。
- 描述目标、Agent 数量倾向、工作流节点、交付契约、门禁、reviewer、工具和记忆边界。
- 配置助手只写 draft，不自动应用代码或修改运行中工作流。

草稿生成后：

1. 打开 `多 Agent` 或 `工作流`。
2. 人工审查并微调字段。
3. 点击 `保存草稿`。
4. 点击 `生成提案`。
5. 审查 diff preview。
6. 点击 `批准应用`。

配置助手优先使用 DeepSeek 生成草稿。DeepSeek 不可用时，会生成保守的本地 fallback 草稿，并在对话面板说明 fallback 状态。
