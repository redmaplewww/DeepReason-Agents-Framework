# AgentOps Hardening Review

本文件记录基于本地 AgentOps 审查后的修复结果和当前边界。它不是生产就绪声明；它是可复核的本地基座验收记录。

## 已修复

- 工作流现在按 `WorkflowSpec.edges` 选择下一节点，支持条件分支、受控重试和显式步数上限。普通任务会跳过 `retrieve`，严格证据缺口最多按 `runtime.workflow_retry_limit` 重试。
- Reviewer 路由合并改为单调安全策略。Reviewer 只能提高证据、风险、难度或严格性，不能把 Coordinator 已作出的高安全判断降级。
- 长期记忆分区名经过字符白名单和解析后边界校验，拒绝 `../`、绝对路径和路径折返。
- CLI/Web 使用 `sessions/` 下的 `SessionStore` 保存线程快照，启动新编排器时恢复短期上下文；`sessions/` 被 git 忽略，并提供显式删除 API。
- 工具调用增加基础 JSON Schema 参数校验、可选超时和幂等键去重。门禁仍先于参数校验执行，避免未经批准的工具触发加载或副作用。
- Deep Agents 初始化失败不再静默吞掉；fallback 句柄暴露 `degraded_reason`。
- Evidence Ledger 对相同 source、locator 和内容哈希去重，避免重复检索污染账本。
- Web 默认拒绝非 localhost 绑定；只有明确设置 `runtime.allow_non_localhost=true` 才会开放绑定。

## 验收命令

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m compileall -q src
node --check src\reasoning_agent_template\web_static\app.js
```

## 仍需外部条件

- DeepSeek API 密钥、网络可用性和外部检索服务不由本地测试替代；没有密钥时应看到结构化失败，而不是假装完成。
- `RuntimeTool` 的超时是调用方超时保护，无法强制终止已经在第三方线程中运行的非协作代码。高风险外部动作仍应提供可验证的 postcondition 和人工回滚流程。
- `SessionStore` 是本地 JSON 快照，不是多进程数据库；需要共享部署时应替换为带锁和访问控制的持久化后端。
- Web 当前是本地调试台，不提供远程认证、CSRF 防护或多租户隔离；不要直接暴露到公网。

## 当前操作边界

工作流配置、Agent 配置、自进化和记忆写入仍然是 proposal/gate/人工批准边界。审查修复不会自动修改知识库、技能包、秘密文件或外部运营系统。
