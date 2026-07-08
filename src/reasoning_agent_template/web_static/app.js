const state = {
  payload: null,
  activeTab: "evidence",
  threadId: getOrCreateThreadId(),
  pendingQuestion: null,
  selectedWorkflowElement: null,
  workflowSpecPayload: null,
  workflowEditMode: false,
  workflowProposal: null,
  selectedAgentName: null,
  agentSpecPayload: null,
  agentEditMode: false,
  agentProposal: null,
  templatePayload: null,
  selectedTemplateId: "autoresearch-agent",
  chatMode: "chat",
  ragLab: { query: "", methods: ["bm25", "semantic", "graph"], result: null, error: "" },
};

const $ = (id) => document.getElementById(id);

function getOrCreateThreadId() {
  const key = "reasoning-agent-thread-id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(key, created);
  return created;
}

function labelStatus(value) {
  const labels = {
    ready: "就绪",
    completed: "已完成",
    running: "运行中",
    active: "工作中",
    accepted: "已接收",
    failed: "失败",
    idle: "空闲",
    pending: "待执行",
    configured: "已配置",
    allow: "允许",
    interrupt: "中断",
    deny: "拒绝",
    none: "无",
    skipped: "已跳过",
    noop: "空转",
    called: "已调用",
    missing_api_key: "缺少 Key",
    error: "错误",
    empty: "无结果",
    filtered: "已过滤",
    fallback: "兜底",
    respond: "响应完成",
    optional: "可选",
    required: "必须",
    soft: "软证据",
    strict: "严格证据",
    not_required: "不需要",
    sufficient: "充足",
    partial: "部分不足",
    unqualified: "证据不合格",
    protected_denied: "保护性拒绝",
    exhausted: "已检索但证据不足",
    missing: "未检索",
    simple: "简单",
    medium: "中等",
    hard: "困难",
    routine: "普通对话",
    evidence_soft: "软证据工作流",
    evidence_strict: "严格证据工作流",
    protected_action: "受保护动作",
    approve: "通过",
    escalate: "升级",
    revise: "修正",
    not_available: "不可用",
    llm: "LLM 判断",
    "llm+reviewer": "LLM + Reviewer",
    rules: "规则兜底",
    rules_fallback_after_invalid_llm_json: "规则兜底",
    stage_started: "阶段开始",
    stage_completed: "阶段完成",
    llm_completed: "LLM 完成",
    llm_routing: "LLM 路由",
    route_review: "路由审查",
    review_decision: "审查结论",
    routing_retry: "路由重试",
    reviewer_retry: "审查重试",
    routing_fallback: "路由兜底",
    reviewer_fallback: "审查兜底",
    external_evidence: "外部证据",
    external_search_attempted: "外部搜索",
    short_term_memory: "短期记忆",
    long_term_memory: "长期记忆",
    flow: "主流程",
    branch: "分支",
    retry: "回检",
    loop: "环路",
  };
  return labels[value] || value || "未知";
}

function labelRisk(value) {
  const labels = {
    none: "无",
    low: "低",
    medium: "中",
    high: "高",
    critical: "严重",
  };
  return labels[value] || value || "未知";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDuration(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "";
  return `${Number(value).toFixed(1)} ms`;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function renderStatus(payload) {
  state.payload = payload;
  $("runtimeLine").textContent = `${payload.runtime?.agent || "agent"} | ${payload.runtime?.workspace || "workspace"}`;
  $("agentStatus").textContent = labelStatus(payload.status || "ready");
  $("stateStatus").textContent = labelStatus(payload.workflow?.current || payload.state_machine?.current || "idle");
  $("routingStatus").textContent = `${labelStatus(payload.routing?.difficulty || "idle")} / ${labelStatus(payload.workflow?.variant || payload.routing?.workflow || "idle")}`;
  $("evidenceCount").textContent = `${labelStatus(payload.evidence?.mode || "idle")} / ${labelStatus(payload.evidence?.strictness || "none")} / ${labelStatus(payload.evidence?.status || "idle")} / ${payload.evidence?.count ?? 0}`;
  $("ragCount").textContent = `${payload.rag?.count ?? 0} / ${payload.external_evidence?.count ?? 0}`;
  $("agentCount").textContent = agentCountLabel(payload);
  $("currentStage").textContent = labelStatus(payload.workflow?.current || payload.state_machine?.current || "idle");
  $("runId").textContent = payload.run_id ? `${payload.run_id} | ${payload.thread_id || state.threadId}` : `暂无运行 | ${state.threadId}`;

  renderWorkingHint(payload);
  renderAgents(payload.agents || []);
  renderWorkflowGraph(workflowDisplayGraph(payload));
  renderTab();
}

function renderWorkingHint(payload) {
  const hint = payload.working_hint || {};
  const status = payload.status || "ready";
  $("progressBanner").className = `progress-banner ${escapeHtml(status)}`;
  $("workingAgent").textContent = hint.agent ? `Agent: ${hint.agent}` : "等待运行";
  $("workingStage").textContent = hint.stage ? `节点: ${hint.stage}` : labelStatus(payload.workflow?.current || payload.state_machine?.current || "idle");
  $("workingHint").textContent = hint.message || (status === "completed" ? "本轮运行已完成。" : "发送消息后，这里会显示当前 Agent、工作流节点和执行提示。");
}

function renderAgents(agents) {
  const displayAgents = agentDisplayList(agents);
  $("agentsList").innerHTML = `
    ${renderTemplateLibrary()}
    ${renderAgentsEditor()}
    ${displayAgents
    .map((agent) => {
      const status = agent.status || "idle";
      return `
        <article class="agent-row ${state.selectedAgentName === agent.name ? "selected" : ""}" data-agent-select="${escapeHtml(agent.name)}">
          <header>
            <h3>${escapeHtml(agent.label || agent.name)}</h3>
            <span class="badge ${escapeHtml(status)}">${escapeHtml(labelStatus(status))}</span>
          </header>
          <p class="mono">${escapeHtml(agent.name)}${agent.protected ? " | protected" : ""}</p>
          <p>${escapeHtml(agent.description || "")}</p>
          <p class="mono">${escapeHtml(agent.last_event || agent.current_stage || "")}</p>
        </article>
      `;
    })
    .join("")}
  `;
}

function agentDisplayList(runtimeAgents) {
  const specPayload = state.agentSpecPayload || {};
  const spec = state.agentEditMode ? specPayload.draft || specPayload.spec : null;
  if (!spec) return runtimeAgents;
  const runtimeByName = new Map((runtimeAgents || []).map((agent) => [agent.name, agent]));
  const protectedAgents = new Set(spec.protected_agents || []);
  const agents = builderVisibleAgents(spec);
  return agents.map((agent) => {
    const runtime = runtimeByName.get(agent.id) || {};
    return {
      ...runtime,
      name: agent.id,
      label: agent.label || agent.id,
      description: agent.description || "",
      responsibilities: agent.responsibilities || [],
      model_role: agent.model_role || "worker",
      tools: agent.tools || [],
      memory_access: agent.memory_access || [],
      workflow_nodes: agent.workflow_nodes || [],
      ui: agent.ui || {},
      protected: protectedAgents.has(agent.id),
      status: runtime.status || "idle",
      current_stage: runtime.current_stage || agent.id,
    };
  });
}

function builderVisibleAgents(spec) {
  const agents = spec.agents || [];
  const hasBuilderLayer = agents.some((agent) => agent.ui?.builder_visible || agent.ui?.builder_hidden);
  if (!hasBuilderLayer) return agents;
  return agents.filter((agent) => !agent.ui?.builder_hidden);
}

function agentCountLabel(payload) {
  const specPayload = state.agentSpecPayload || {};
  const spec = state.agentEditMode ? specPayload.draft || specPayload.spec : null;
  if (!spec) return `${payload.agents?.length || 0} 个已加载`;
  const agents = spec.agents || [];
  const protectedAgents = new Set(spec.protected_agents || []);
  const visible = builderVisibleAgents(spec);
  const protectedCount = agents.filter((agent) => protectedAgents.has(agent.id)).length;
  const designCount = visible.filter((agent) => !protectedAgents.has(agent.id)).length;
  if (agents.some((agent) => agent.ui?.builder_visible || agent.ui?.builder_hidden)) {
    return `${designCount} 个设计 / ${protectedCount} 个底座`;
  }
  return `${agents.length} 个草稿`;
}

function renderTemplateLibrary() {
  const payload = state.templatePayload;
  const templates = payload?.templates || [];
  const selectedId = state.selectedTemplateId || templates[0]?.id || "";
  const selected = templates.find((item) => item.id === selectedId) || templates[0];
  if (!payload) {
    return `
      <article class="agent-row template-library">
        <header>
          <h3>Agent 模板库</h3>
          <button type="button" class="secondary" data-template-action="refresh">加载模板</button>
        </header>
        <p>模板库会同时管理 Agent Spec 和 Workflow Spec，适合保存多轮开发中的稳定设计。</p>
      </article>
    `;
  }
  return `
    <article class="agent-row template-library">
      <header>
        <h3>Agent 模板库 <span class="badge ${templates.length ? "completed" : "idle"}">${escapeHtml(templates.length)} 套</span></h3>
        <button type="button" class="secondary" data-template-action="refresh">刷新</button>
      </header>
      <div class="template-controls">
        <label>快速切换
          <select id="templateSelect">
            ${templates
              .map(
                (item) => `
                  <option value="${escapeHtml(item.id)}" ${item.id === selectedId ? "selected" : ""}>
                    ${escapeHtml(item.label || item.id)} · ${escapeHtml(item.source || "template")}
                  </option>
                `,
              )
              .join("")}
          </select>
        </label>
        <button type="button" data-template-action="apply" ${selected ? "" : "disabled"}>加载到草稿</button>
      </div>
      ${
        selected
          ? `
        <p>${escapeHtml(selected.description || "")}</p>
        <p class="mono">${escapeHtml(selected.id)} | ${escapeHtml(selected.source)} | Agent ${escapeHtml(selected.agents_count || 0)} / 节点 ${escapeHtml(selected.workflow_nodes_count || 0)}</p>
      `
          : `<p class="empty">暂无模板。可以先保存当前设计为用户模板。</p>`
      }
      <div class="template-save-form">
        <label>模板 ID<input id="templateSaveId" placeholder="例如 my-research-v1"></label>
        <label>显示名称<input id="templateSaveLabel" placeholder="例如 我的研究 Agent v1"></label>
        <label class="span-2">说明<input id="templateSaveDescription" placeholder="记录这个模板适合什么任务、有哪些边界"></label>
        <button type="button" data-template-action="save">保存当前为模板</button>
      </div>
      <p class="mono">内置目录: ${escapeHtml(payload.paths?.builtin || "")}</p>
      <p class="mono">用户目录: ${escapeHtml(payload.paths?.user || "")}</p>
    </article>
  `;
}

function renderAgentsEditor() {
  const specPayload = state.agentSpecPayload;
  if (!specPayload) {
    return `
      <article class="agent-editor agent-row">
        <header>
          <h3>多 Agent 编辑器</h3>
          <button type="button" class="secondary" data-agent-action="load-spec">加载 Agent Spec</button>
        </header>
        <p>加载后可以新增、删除、修改 Agent；AI 搭建由左侧对话区的顶层搭建助手负责。</p>
      </article>
    `;
  }
  const spec = editableAgentsSpec();
  const validation = specPayload.validation || {};
  const selected = state.selectedAgentName;
  const protectedAgents = new Set(spec.protected_agents || []);
  const proposal = state.agentProposal;
  return `
    <article class="agent-editor agent-row ${state.agentEditMode ? "editing" : ""}">
      <header>
        <h3>多 Agent 编辑器 <span class="badge ${validation.ok ? "completed" : "interrupt"}">${validation.ok ? "校验通过" : "需要处理"}</span></h3>
        <button type="button" class="secondary" data-agent-action="toggle-edit">${state.agentEditMode ? "退出编辑" : "编辑 Agent"}</button>
      </header>
      <p class="mono">Spec: ${escapeHtml(spec.name || "agents")} | revision=${escapeHtml(spec.revision || "dev")} | draft=${specPayload.draft ? "yes" : "no"}</p>
      ${renderAgentBuilderScope(spec)}
      ${renderAgentValidation(validation)}
      <div class="agent-editor-actions">
        <button type="button" class="secondary" data-agent-action="load-spec">刷新 Spec</button>
        <button type="button" data-agent-action="save-draft" ${state.agentEditMode ? "" : "disabled"}>保存草稿</button>
        <button type="button" data-agent-action="create-proposal" ${state.agentEditMode ? "" : "disabled"}>生成提案</button>
        <button type="button" class="danger" data-agent-action="apply-proposal" ${proposal?.proposal_id ? "" : "disabled"}>批准应用</button>
      </div>
      ${
        state.agentEditMode
          ? `
        <div class="agent-editor-actions">
          <button type="button" data-agent-action="add-agent">添加 Agent</button>
          <button type="button" class="danger" data-agent-action="delete-agent" ${selected ? "" : "disabled"}>删除所选</button>
        </div>
        ${renderAgentObjectEditor(spec, selected, protectedAgents)}
      `
          : `<p class="empty">点击“编辑 Agent”后，可以点下面的 Agent 卡片修改；AI 搭建请使用左侧对话区。</p>`
      }
      ${proposal ? renderAgentProposal(proposal) : ""}
      <p class="empty">需要 AI 生成草稿时，请到左侧对话区切换“搭建助手”，或在普通对话中用 /配置、/搭建 触发。</p>
    </article>
  `;
}

function renderAgentBuilderScope(spec) {
  const agents = spec.agents || [];
  if (!agents.some((agent) => agent.ui?.builder_visible || agent.ui?.builder_hidden)) return "";
  const protectedAgents = new Set(spec.protected_agents || []);
  const designCount = builderVisibleAgents(spec).filter((agent) => !protectedAgents.has(agent.id)).length;
  const protectedCount = agents.filter((agent) => protectedAgents.has(agent.id)).length;
  return `<p class="mono">搭建视图: ${designCount} 个设计 Agent；${protectedCount} 个底座 Agent 已隐藏但会随提案保留。</p>`;
}

function renderAgentValidation(validation) {
  const errors = validation.errors || [];
  const warnings = validation.warnings || [];
  if (!errors.length && !warnings.length) return `<p class="mono">校验: agents spec 可运行。</p>`;
  return `
    <div class="workflow-validation">
      ${errors.map((item) => `<p class="error">错误: ${escapeHtml(item)}</p>`).join("")}
      ${warnings.map((item) => `<p class="warn">提醒: ${escapeHtml(item)}</p>`).join("")}
    </div>
  `;
}

function renderAgentObjectEditor(spec, selected, protectedAgents) {
  if (!selected) return `<p class="empty">点击下面的 Agent 卡片后，这里会显示可编辑属性。</p>`;
  const agent = (spec.agents || []).find((item) => item.id === selected);
  if (!agent) return `<p class="empty">未找到所选 Agent。</p>`;
  const isProtected = protectedAgents.has(agent.id);
  return `
    <div class="agent-edit-grid">
      <label>Agent ID<input id="agentId" value="${escapeHtml(agent.id)}" ${isProtected ? "readonly" : ""}></label>
      <label>显示名称<input id="agentLabel" value="${escapeHtml(agent.label || "")}"></label>
      <label>模型角色
        <select id="agentModelRole">
          ${["planner", "worker", "critic", "grader"].map((item) => `<option value="${item}" ${item === agent.model_role ? "selected" : ""}>${item}</option>`).join("")}
        </select>
      </label>
      <label class="span-2">描述<input id="agentDescription" value="${escapeHtml(agent.description || "")}"></label>
      <label class="span-2">职责，每行一条<textarea id="agentResponsibilities">${escapeHtml((agent.responsibilities || []).join("\n"))}</textarea></label>
      <label>工具，每行一条<textarea id="agentTools">${escapeHtml((agent.tools || []).join("\n"))}</textarea></label>
      <label>记忆权限，每行一条<textarea id="agentMemoryAccess">${escapeHtml((agent.memory_access || []).join("\n"))}</textarea></label>
      <label>绑定 Workflow 节点，每行一条<textarea id="agentWorkflowNodes">${escapeHtml((agent.workflow_nodes || []).join("\n"))}</textarea></label>
      <label>权限 JSON<textarea id="agentPermissions">${escapeHtml(JSON.stringify(agent.permissions || {}, null, 2))}</textarea></label>
      <label class="span-2">交付契约 JSON<textarea id="agentHandoff">${escapeHtml(JSON.stringify(agent.handoff_contract || {}, null, 2))}</textarea></label>
      <div class="span-2 agent-editor-actions">
        <button type="button" data-agent-action="save-agent">更新 Agent</button>
        <button type="button" class="danger" data-agent-action="delete-agent" ${isProtected ? "disabled" : ""}>删除 Agent</button>
      </div>
    </div>
  `;
}

function renderAgentProposal(proposal) {
  return `
    <article class="proposal-preview">
      <p><strong>待应用 Agent 提案</strong> <span class="badge pending">${escapeHtml(proposal.proposal_id)}</span></p>
      <p class="mono">draft_hash=${escapeHtml(proposal.draft_hash || "")} | files=${escapeHtml((proposal.modified_files || []).join(", "))}</p>
      <pre class="mono artifact-json">${escapeHtml(proposal.diff_preview || "暂无 diff。")}</pre>
    </article>
  `;
}

function workflowDisplayGraph(payload) {
  const runtimeWorkflow = payload.workflow || {};
  const specPayload = state.workflowSpecPayload || {};
  const spec = state.workflowEditMode ? specPayload.draft || specPayload.spec : null;
  if (!spec) return runtimeWorkflow;
  const runtimeNodes = new Map((runtimeWorkflow.nodes || []).map((node) => [node.id, node]));
  const specNodes = builderVisibleWorkflowNodes(spec);
  const visibleNodeIds = new Set(specNodes.map((node) => node.id));
  const specEdges = builderWorkflowLayerActive(spec)
    ? (spec.edges || []).filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to))
    : spec.edges || [];
  return {
    ...runtimeWorkflow,
    status: runtimeWorkflow.status || "idle",
    nodes: specNodes.map((node) => {
      const runtimeNode = runtimeNodes.get(node.id) || {};
      return {
        ...runtimeNode,
        id: node.id,
        label: node.label || node.id,
        agent: node.agent || runtimeNode.agent || "coordinator",
        description: node.description || "",
        work: node.work || node.description || "",
        input: node.input_contract || "",
        output: node.output_contract || "",
        checkpoint: Boolean(node.checkpoint),
        handler_kind: node.handler_kind || "builtin",
        handler: node.handler || node.id,
        gate_policy: node.gate_policy || {},
        status: runtimeNode.status || "pending",
        effective_status: runtimeNode.effective_status || "pending",
        work_done: Boolean(runtimeNode.work_done),
        skip_reason: runtimeNode.skip_reason || "",
        observed: runtimeNode.observed || "draft",
        duration_ms: runtimeNode.duration_ms,
        artifacts: runtimeNode.artifacts || { actual_input: {}, actual_output: {}, process: [], handoff: {} },
      };
    }),
    edges: specEdges.map((edge) => ({
      id: edge.id || `${edge.from}->${edge.to}`,
      from: edge.from,
      to: edge.to,
      type: edge.type || "flow",
      label: edge.condition || edge.label || "",
      condition: edge.condition || "",
      handoff_contract: edge.handoff_contract || {},
      gate_policy: edge.gate_policy || {},
      planner_contract: edge.planner_contract || {},
      reviewer_required: Boolean(edge.reviewer_required),
    })),
    checkpoints: specNodes.filter((node) => node.checkpoint).map((node) => node.id),
    spec: {
      name: spec.name,
      revision: spec.revision,
      version: spec.version,
      start_node: spec.start_node,
      terminal_nodes: spec.terminal_nodes || [],
    },
  };
}

function builderWorkflowLayerActive(spec) {
  return (spec.nodes || []).some((node) => node.ui?.builder_visible || node.ui?.builder_hidden);
}

function builderVisibleWorkflowNodes(spec) {
  const nodes = spec.nodes || [];
  if (!builderWorkflowLayerActive(spec)) return nodes;
  return nodes.filter((node) => !node.ui?.builder_hidden);
}

function renderWorkflowGraph(workflow) {
  renderCytoscapeWorkflowGraph($("workflowGraph"), workflow, { compact: false });
}

function renderCytoscapeWorkflowGraph(container, workflow, options = {}) {
  const nodes = workflow.nodes || [];
  const edges = workflow.edges || [];
  if (container._workflowCy) {
    container._workflowCy.destroy();
    container._workflowCy = null;
  }
  if (!nodes.length) {
    container.innerHTML = `<p class="empty">暂无工作流图。</p>`;
    return;
  }
  if (typeof cytoscape !== "function") {
    container.innerHTML = `<p class="empty">图渲染库未加载，无法绘制工作流。</p>`;
    return;
  }
  container.innerHTML = `
    <div class="workflow-graph-canvas"></div>
    <div class="workflow-graph-legend">
      <span><i class="legend-dot active"></i>工作中</span>
      <span><i class="legend-dot completed"></i>已完成</span>
      <span><i class="legend-dot skipped"></i>跳过/空转</span>
      <span><i class="legend-line retry"></i>回路/重试</span>
    </div>
  `;
  const canvas = container.querySelector(".workflow-graph-canvas");
  const nodeIds = new Set(nodes.map((node) => node.id));
  const elements = [
    ...nodes.map((node, index) => {
      const status = node.status || "pending";
      const effect = node.effective_status || status;
      const duration = formatDuration(node.duration_ms);
      return {
        data: {
          id: node.id,
          label: node.id,
          order: index,
          agent: node.agent || "-",
          effect,
          displayLabel: `${node.label || workflowNodeLabel(node.id)}\n${node.agent || "-"}`,
          status,
          description: node.description || "",
          input: node.input || "",
          output: node.output || "",
          observed: node.observed || "",
          duration: duration || (node.work_done ? "有产出" : "无可见产出"),
          skip: node.skip_reason || "",
        },
        classes: [status, effect, node.checkpoint ? "checkpoint" : ""].filter(Boolean).join(" "),
        position: workflowNodePosition(node.id, index, nodes, canvas, options),
      };
    }),
    ...edges
      .filter((edge) => nodeIds.has(edge.from) && nodeIds.has(edge.to))
      .map((edge, index) => ({
        data: {
          id: `edge-${index}-${edge.from}-${edge.to}`,
          source: edge.from,
          target: edge.to,
          label: "",
          type: edge.type || "flow",
          description: edge.label || "",
          from: edge.from,
          to: edge.to,
        },
        classes: edge.type || "flow",
      })),
  ];

  const cy = cytoscape({
    container: canvas,
    elements,
    minZoom: 0.35,
    maxZoom: 1.7,
    autoungrabify: false,
    style: cytoscapeWorkflowStyle(),
    layout: workflowHybridLayoutOptions(options),
  });

  cy.one("layoutstop", () => {
    cy.fit(undefined, options.compact ? 20 : 30);
  });
  applyWorkflowSelectionToGraph(cy);
  cy.on("mouseover", "node", (event) => {
    const item = event.target.data();
    canvas.title = `${item.label} | Agent: ${item.agent} | ${labelStatus(item.effect)} | ${item.observed}`;
  });
  cy.on("mouseout", "node", () => {
    canvas.title = "";
  });
  cy.on("tap", "node", (event) => {
    handleWorkflowGraphSelection(workflow, event.target, "node");
  });
  cy.on("tap", "edge", (event) => {
    handleWorkflowGraphSelection(workflow, event.target, "edge");
  });
  container._workflowCy = cy;
}

function handleWorkflowGraphSelection(workflow, element, kind) {
  const data = element.data();
  if (kind === "edge") {
    state.selectedWorkflowElement = {
      kind: "edge",
      id: data.id,
      from: data.from || data.source,
      to: data.to || data.target,
      type: data.type || "flow",
    };
  } else {
    state.selectedWorkflowElement = {
      kind: "node",
      id: data.id,
    };
  }
  setActiveTab("workflow");
  applyWorkflowSelectionToOpenGraphs();
  renderTab();
  applyWorkflowSelectionToOpenGraphs();
  focusWorkflowSelectionPanel();
}

function applyWorkflowSelectionToOpenGraphs() {
  document.querySelectorAll(".workflow-graph").forEach((container) => {
    if (container._workflowCy) applyWorkflowSelectionToGraph(container._workflowCy);
  });
}

function focusWorkflowSelectionPanel() {
  window.setTimeout(() => {
    const panel = document.querySelector(".workflow-selection");
    const scroller = $("tabContent");
    if (!panel || !scroller) return;
    const scrollerRect = scroller.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const targetTop = scroller.scrollTop + (panelRect.top - scrollerRect.top) - 10;
    panel.classList.add("selection-focus");
    scroller.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
    window.setTimeout(() => panel.classList.remove("selection-focus"), 1400);
  }, 0);
}

function applyWorkflowSelectionToGraph(cy) {
  const selection = state.selectedWorkflowElement;
  cy.elements().removeClass("selected");
  if (!selection) return;
  const match =
    selection.kind === "edge"
      ? cy
          .edges()
          .filter(
            (edge) =>
              edge.data("id") === selection.id ||
              (edge.data("from") === selection.from &&
                edge.data("to") === selection.to &&
                edge.data("type") === selection.type),
          )
      : cy.nodes().filter((node) => node.id() === selection.id);
  match.addClass("selected");
}

function workflowHybridLayoutOptions(options) {
  return {
    name: "preset",
    fit: true,
    padding: options.compact ? 18 : 26,
    animate: true,
    animationDuration: 320,
  };
}

function workflowLoopNodeOrder() {
  return ["retrieve", "reason", "evidence_audit", "gate", "act_or_answer", "verify", "consolidate"];
}

function workflowEvidenceLoopNodeOrder() {
  return ["retrieve", "reason", "evidence_audit"];
}

function workflowReviewLoopNodeOrder() {
  return ["gate", "act_or_answer", "verify", "consolidate"];
}

function workflowLoopClusterNodes(nodes) {
  const available = new Set(nodes.map((node) => node.id));
  return workflowLoopNodeOrder().filter((id) => available.has(id));
}

function workflowAvailableLoopNodes(nodes, orderedIds) {
  const available = new Set(nodes.map((node) => node.id));
  return orderedIds.filter((nodeId) => available.has(nodeId));
}

function workflowNodePosition(id, index, nodes, canvas, options) {
  const width = Math.max(canvas.clientWidth || 760, options.compact ? 640 : 760);
  const height = Math.max(canvas.clientHeight || 360, options.compact ? 340 : 360);
  const centerY = height * 0.5;
  const evidenceLoopNodes = workflowAvailableLoopNodes(nodes, workflowEvidenceLoopNodeOrder());
  const evidenceLoopIndex = evidenceLoopNodes.indexOf(id);
  if (evidenceLoopIndex >= 0 && evidenceLoopNodes.length >= 3) {
    const angles = {
      retrieve: Math.PI,
      reason: -Math.PI / 2,
      evidence_audit: 0,
    };
    const loopCenterX = width * 0.42;
    const radiusX = Math.min(width * 0.09, options.compact ? 58 : 72);
    const radiusY = Math.min(height * 0.18, options.compact ? 48 : 62);
    const angle = angles[id] ?? (-Math.PI / 2 + (evidenceLoopIndex / evidenceLoopNodes.length) * 2 * Math.PI);
    return {
      x: loopCenterX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  }

  const reviewLoopNodes = workflowAvailableLoopNodes(nodes, workflowReviewLoopNodeOrder());
  const reviewLoopIndex = reviewLoopNodes.indexOf(id);
  if (reviewLoopIndex >= 0 && reviewLoopNodes.length >= 3) {
    const angles = {
      gate: Math.PI,
      act_or_answer: -Math.PI / 2,
      verify: 0,
      consolidate: Math.PI / 2,
    };
    const loopCenterX = width * 0.68;
    const radiusX = Math.min(width * 0.09, options.compact ? 58 : 72);
    const radiusY = Math.min(height * 0.18, options.compact ? 48 : 62);
    const angle = angles[id] ?? (-Math.PI / 2 + (reviewLoopIndex / reviewLoopNodes.length) * 2 * Math.PI);
    return {
      x: loopCenterX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  }

  const anchors = {
    intake: { x: options.compact ? 60 : 68, y: centerY },
    plan: { x: options.compact ? 176 : 198, y: centerY },
    respond: { x: width - (options.compact ? 58 : 68), y: centerY },
  };
  if (anchors[id]) return anchors[id];

  const span = Math.max(width - 160, 1);
  const denom = Math.max(nodes.length - 1, 1);
  return {
    x: 80 + (span * index) / denom,
    y: centerY,
  };
}

function workflowNodeLabel(id) {
  const labels = {
    intake: "接收",
    plan: "计划",
    retrieve: "检索",
    reason: "推理",
    evidence_audit: "证据审计",
    gate: "门禁",
    act_or_answer: "行动/回答",
    verify: "验证",
    consolidate: "沉淀",
    respond: "响应",
  };
  return labels[id] || id;
}

function cytoscapeWorkflowStyle() {
  return [
    {
      selector: "node",
      style: {
        "shape": "round-rectangle",
        "width": 98,
        "height": 46,
        "background-color": "#ffffff",
        "border-width": 1.4,
        "border-color": "#c2ccd9",
        "label": "data(displayLabel)",
        "font-size": 9,
        "font-weight": 700,
        "color": "#151922",
        "text-valign": "center",
        "text-halign": "center",
        "line-height": 1.14,
        "text-wrap": "wrap",
        "text-max-width": 88,
        "overlay-opacity": 0,
      },
    },
    {
      selector: "node.active",
      style: {
        "background-color": "#ecfdfb",
        "border-color": "#0f766e",
        "border-width": 3,
      },
    },
    {
      selector: "node.selected",
      style: {
        "background-color": "#f5fffd",
        "border-color": "#14a39a",
        "border-width": 4,
      },
    },
    {
      selector: "node.completed",
      style: {
        "background-color": "#f0faf1",
        "border-color": "#2f7d32",
      },
    },
    {
      selector: "node.skipped, node.noop",
      style: {
        "background-color": "#fff8eb",
        "border-color": "#b7791f",
        "border-style": "dashed",
      },
    },
    {
      selector: "node.checkpoint",
      style: {
        "border-width": 2.4,
      },
    },
    {
      selector: "edge",
      style: {
        "width": 2.2,
        "line-color": "#c2ccd9",
        "target-arrow-color": "#c2ccd9",
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "control-point-step-size": 34,
        "label": "data(label)",
        "font-size": 8,
        "color": "#667085",
        "text-background-color": "#fbfdff",
        "text-background-opacity": 0.82,
        "text-background-padding": 2,
        "text-rotation": "none",
        "opacity": 0.86,
        "overlay-opacity": 0,
      },
    },
    {
      selector: "edge.branch",
      style: {
        "line-style": "dashed",
        "line-color": "#64748b",
        "target-arrow-color": "#64748b",
      },
    },
    {
      selector: "edge.retry, edge.revise, edge.loop",
      style: {
        "curve-style": "bezier",
        "control-point-step-size": 48,
        "line-style": "dashed",
        "line-color": "#b7791f",
        "target-arrow-color": "#b7791f",
        "color": "#b7791f",
      },
    },
    {
      selector: "edge.selected",
      style: {
        "width": 3.8,
        "line-color": "#14a39a",
        "target-arrow-color": "#14a39a",
        "opacity": 1,
      },
    },
  ];
}
function referenceIndexByEvidenceId(payload) {
  const references = payload.evidence?.references || [];
  return new Map(references.map((item) => [item.id, item.index]));
}

function referenceLabel(payload, evidenceId) {
  const index = referenceIndexByEvidenceId(payload).get(evidenceId);
  return index ? `参考文献[${index}]` : evidenceId || "未绑定参考文献";
}

function renderTab() {
  const payload = state.payload || {};
  const target = $("tabContent");
  if (state.activeTab === "evidence") return renderEvidenceTab(target, payload);
  if (state.activeTab === "routing") return renderRoutingTab(target, payload);
  if (state.activeTab === "rag") return renderRagTab(target, payload);
  if (state.activeTab === "external") return renderExternalTab(target, payload);
  if (state.activeTab === "workflow") return renderWorkflowTab(target, payload);
  if (state.activeTab === "gates") return renderGatesTab(target, payload);
  if (state.activeTab === "memory") return renderMemoryTab(target, payload);
  if (state.activeTab === "skills") return renderSkillsTab(target, payload);
  if (state.activeTab === "events") return renderEventsTab(target, payload);
  target.innerHTML = `<pre class="mono">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });
}

function renderEvidenceTab(target, payload) {
  const items = payload.evidence?.items || [];
  const references = payload.evidence?.references || [];
  const qualifiedReferences = (payload.evidence?.qualified_evidence_ids || [])
    .map((id) => referenceLabel(payload, id))
    .join(", ");
  const referenceList = references.length
    ? `
      <article class="detail-item reference-list">
        <h3>参考文献索引</h3>
        ${references
          .map(
            (item) => `
          <p class="reference-item">
            <strong>[${escapeHtml(item.index)}]</strong>
            ${escapeHtml(item.source_type)} | ${escapeHtml(item.uri)} | ${escapeHtml(item.locator)}
          </p>
          <p>${escapeHtml(item.summary || "")}</p>
        `,
          )
          .join("")}
      </article>
    `
    : "";
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>模式</strong> ${escapeHtml(labelStatus(payload.evidence?.mode || "idle"))} / ${escapeHtml(labelStatus(payload.evidence?.strictness || "none"))}</p>
      <p>状态: ${escapeHtml(labelStatus(payload.evidence?.status || "idle"))} | 风险: ${escapeHtml(labelRisk(payload.evidence?.risk_level || "none"))}</p>
      <p>合格证据: ${escapeHtml(qualifiedReferences || "暂无")}</p>
      <p class="mono">${escapeHtml((payload.evidence?.reasons || []).join("\n") || "暂无证据策略原因。")}</p>
    </article>
    ${referenceList}
    ${
      items.length
        ? items
            .map(
              (item) => `
          <article class="detail-item">
            <p><strong>${escapeHtml(referenceLabel(payload, item.id))}</strong> <span class="badge completed">${escapeHtml(item.source_type)}</span></p>
            <p class="mono">${escapeHtml(item.uri)} | ${escapeHtml(item.locator)} | confidence=${escapeHtml(item.confidence)}</p>
            <p>${escapeHtml(item.summary || "")}</p>
          </article>
        `,
            )
            .join("")
        : `<p class="empty">暂无证据。简单对话不会强制检索；中高难或高风险任务会在这里沉淀证据。</p>`
    }
  `;
}

function renderRoutingTab(target, payload) {
  const routing = payload.routing || {};
  const reviewer = payload.reviewer || {};
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>Coordinator</strong> <span class="badge">${escapeHtml(labelStatus(routing.source || "idle"))}</span></p>
      <p>难度: ${escapeHtml(labelStatus(routing.difficulty || "idle"))} | 工作流: ${escapeHtml(labelStatus(routing.workflow || "idle"))} | confidence=${escapeHtml(routing.confidence ?? 0)}</p>
      <pre class="mono">${escapeHtml(JSON.stringify(routing.decision || {}, null, 2))}</pre>
    </article>
    <article class="detail-item">
      <p><strong>Reviewer</strong> <span class="badge ${escapeHtml(reviewer.status || "idle")}">${escapeHtml(labelStatus(reviewer.status || "idle"))}</span></p>
      <p class="mono">${escapeHtml((reviewer.findings || []).join("\n") || "暂无审查意见。")}</p>
      <pre class="mono">${escapeHtml(JSON.stringify(reviewer.decision || {}, null, 2))}</pre>
    </article>
  `;
}

function renderRagTab(target, payload) {
  const liveRag = payload.rag || {};
  const lab = state.ragLab?.result;
  const rag = lab || liveRag;
  const items = rag.results || [];
  const selected = new Set(state.ragLab?.methods || ["bm25", "semantic", "graph"]);
  const queryValue = state.ragLab?.query || liveRag.query || "";
  const methodToggle = (value, label) => `
    <label class="method-toggle">
      <input type="checkbox" name="ragMethod" value="${escapeHtml(value)}" ${selected.has(value) ? "checked" : ""}>
      ${escapeHtml(label)}
    </label>
  `;
  const diagnostics = rag.diagnostics || [];
  const diagnosticsBlock = diagnostics.length
    ? diagnostics
        .map(
          (item) => `
        <article class="detail-item compact">
          <p><strong>${escapeHtml(item.source || "rag")}</strong> <span class="badge ${escapeHtml(item.status || "completed")}">${escapeHtml(labelStatus(item.status || "completed"))}</span></p>
          <p class="mono">${escapeHtml(item.message || "")}</p>
        </article>
      `,
        )
        .join("")
    : "";
  target.innerHTML = `
    <form id="ragQueryForm" class="rag-query-panel">
      <input name="query" value="${escapeHtml(queryValue)}" placeholder="输入一个查询来测试 RAG 召回">
      <div class="method-toggles">
        ${methodToggle("bm25", "BM25")}
        ${methodToggle("semantic", "语义")}
        ${methodToggle("graph", "Graph")}
        ${methodToggle("wiki", "Wiki")}
      </div>
      <button type="submit">检索</button>
    </form>
    <article class="detail-item">
      <p><strong>RAG 检索</strong> ${escapeHtml(items.length)} 条</p>
      <p>方法: ${escapeHtml((rag.methods || liveRag.methods || []).join(" + ") || "暂无")}</p>
      <p class="mono">query=${escapeHtml(rag.query || queryValue || "")}</p>
      ${state.ragLab?.error ? `<p class="error">${escapeHtml(state.ragLab.error)}</p>` : ""}
    </article>
    ${diagnosticsBlock}
    ${
      items.length
        ? items
            .map((item) => {
              const breakdown = item.score_breakdown || {};
              const breakdownText = Object.entries(breakdown)
                .map(([key, value]) => `${key}=${Number(value || 0).toFixed(2)}`)
                .join(" | ");
              return `
        <article class="detail-item">
          <p><strong>${escapeHtml(item.source)}</strong> <span class="badge completed">${Number(item.score || 0).toFixed(2)}</span></p>
          <p class="mono">${escapeHtml(item.span)} | ${escapeHtml(referenceLabel(payload, item.evidence_id))}</p>
          <p class="mono">method=${escapeHtml(item.retrieval_method || "keyword")}${breakdownText ? ` | ${escapeHtml(breakdownText)}` : ""}</p>
          <p>${escapeHtml(item.text || "")}</p>
        </article>
      `;
            })
            .join("")
        : `<p class="empty">暂无本地 RAG 结果。</p>`
    }
  `;
}

function renderExternalTab(target, payload) {
  const external = payload.external_evidence || {};
  const items = external.results || [];
  const diagnostics = external.diagnostics || [];
  const diagnosticsBlock = diagnostics.length
    ? diagnostics
        .map(
          (item) => `
        <article class="detail-item">
          <p><strong>${escapeHtml(item.source || "source")}</strong> <span class="badge ${escapeHtml(item.status || "idle")}">${escapeHtml(labelStatus(item.status || "idle"))}</span></p>
          <p class="mono">${escapeHtml(item.message || "")}</p>
          <p class="mono">${escapeHtml(item.url || "")}</p>
        </article>
      `,
        )
        .join("")
    : "";
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>外部证据检索</strong> ${escapeHtml(items.length)} 条</p>
      <p>尝试来源: ${escapeHtml((external.attempted_sources || []).join(", ") || "暂无")}</p>
      <p class="mono">query=${escapeHtml(external.query || "")}</p>
    </article>
    ${
      items.length
        ? items
            .map(
              (item) => `
          <article class="detail-item">
            <p><strong>${escapeHtml(item.source)}</strong> <span class="badge completed">${Number(item.score || 0).toFixed(2)}</span></p>
            <p class="mono">${escapeHtml(item.span)} | ${escapeHtml(referenceLabel(payload, item.evidence_id))}</p>
            <p>${escapeHtml(item.text || "")}</p>
          </article>
        `,
            )
            .join("")
        : `<p class="empty">暂无通过相关性过滤的外部论文、网络或用户经验证据。</p>`
    }
    ${diagnosticsBlock}
  `;
}

function renderWorkflowSelectionPanel(payload) {
  const workflow = workflowDisplayGraph(payload);
  const selection = state.selectedWorkflowElement;
  if (!selection) {
    return `
      <article class="detail-item workflow-selection">
        <p><strong>工作过程与结果</strong></p>
        <p>未选择工作流对象。选中图上的节点或连线后，这里会显示对应模块的输入、输出、执行摘要、启用 Agent、审查和门禁。</p>
      </article>
    `;
  }
  if (selection.kind === "edge") {
    const edge = workflowSelectedEdge(workflow, selection);
    return edge ? renderWorkflowEdgeSelection(payload, edge) : renderMissingWorkflowSelection(selection);
  }
  const node = workflowSelectedNode(workflow, selection.id);
  return node ? renderWorkflowNodeSelection(payload, node) : renderMissingWorkflowSelection(selection);
}

function renderMissingWorkflowSelection(selection) {
  return `
    <article class="detail-item workflow-selection">
      <p><strong>工作过程与结果</strong></p>
      <p>当前选择 ${escapeHtml(selection.kind || "unknown")}:${escapeHtml(selection.id || selection.from || "")} 已不在最新工作流中。</p>
    </article>
  `;
}

function workflowSelectedNode(workflow, id) {
  return (workflow.nodes || []).find((node) => node.id === id);
}

function workflowSelectedEdge(workflow, selection) {
  return (workflow.edges || []).find(
    (edge, index) =>
      `edge-${index}-${edge.from}-${edge.to}` === selection.id ||
      (edge.from === selection.from && edge.to === selection.to && (edge.type || "flow") === selection.type),
  );
}

function renderWorkflowNodeSelection(payload, node) {
  const agents = workflowAgentsForNode(payload, node);
  const events = workflowEventsForNode(payload, node);
  const artifacts = node.artifacts || {};
  return `
    <article class="detail-item workflow-selection">
      <header class="workflow-selection-head">
        <div>
          <p><strong>工作过程与结果</strong></p>
          <h3>${escapeHtml(workflowNodeLabel(node.id))} <span class="mono">${escapeHtml(node.id)}</span></h3>
        </div>
        <div class="workflow-selection-badges">
          <span class="badge ${escapeHtml(node.status || "pending")}">${escapeHtml(labelStatus(node.status || "pending"))}</span>
          <span class="badge ${escapeHtml(node.effective_status || node.status || "pending")}">${escapeHtml(labelStatus(node.effective_status || node.status || "pending"))}</span>
        </div>
      </header>
      <div class="workflow-selection-grid">
        <section>
          <h4>真实输入</h4>
          ${renderArtifactBlock("input contract", node.input || "未声明")}
          ${renderArtifactBlock("actual_input", artifacts.actual_input || {})}
          <p class="mono">耗时: ${escapeHtml(formatDuration(node.duration_ms) || "未执行")}</p>
        </section>
        <section>
          <h4>真实输出</h4>
          ${renderArtifactBlock("output contract", node.output || "未声明")}
          ${renderArtifactBlock("actual_output", artifacts.actual_output || {})}
        </section>
        <section>
          <h4>思考 / 交付过程</h4>
          ${renderArtifactBlock("process", artifacts.process || node.observed || "暂无执行观察。")}
          ${renderArtifactBlock("handoff", artifacts.handoff || {})}
        </section>
        <section>
          <h4>启用 Agent / 事件</h4>
          ${renderWorkflowAgentList(agents)}
          ${renderWorkflowEventList(events)}
        </section>
      </div>
    </article>
  `;
}

function renderWorkflowEdgeSelection(payload, edge) {
  const workflow = workflowDisplayGraph(payload);
  const source = workflowSelectedNode(workflow, edge.from) || { id: edge.from };
  const target = workflowSelectedNode(workflow, edge.to) || { id: edge.to };
  const details = workflowEdgeTransitionDetails(payload, edge, source, target);
  const agents = workflowAgentsForEdge(payload, source, target, edge);
  const events = workflowEventsForEdge(payload, edge, source, target);
  const sourceArtifacts = source.artifacts || {};
  const targetArtifacts = target.artifacts || {};
  return `
    <article class="detail-item workflow-selection">
      <header class="workflow-selection-head">
        <div>
          <p><strong>转交流程审查</strong></p>
          <h3>${escapeHtml(workflowNodeLabel(edge.from))} -> ${escapeHtml(workflowNodeLabel(edge.to))}</h3>
        </div>
        <div class="workflow-selection-badges">
          <span class="badge ${escapeHtml(edge.type || "flow")}">${escapeHtml(labelStatus(edge.type || "flow"))}</span>
        </div>
      </header>
      <div class="workflow-selection-grid">
        <section>
          <h4>上游真实输出</h4>
          ${renderArtifactBlock(`${edge.from}.actual_output`, sourceArtifacts.actual_output || source.output || {})}
          <p class="mono">条件: ${escapeHtml(edge.label || "默认主流程")}</p>
        </section>
        <section>
          <h4>下游真实输入</h4>
          ${renderArtifactBlock(`${edge.to}.actual_input`, targetArtifacts.actual_input || target.input || {})}
        </section>
        <section>
          <h4>审查 / 门禁 / 交付</h4>
          ${details.map((item) => `<p>${escapeHtml(item)}</p>`).join("")}
          ${renderArtifactBlock("source_handoff", sourceArtifacts.handoff || {})}
          ${renderArtifactBlock("target_handoff", targetArtifacts.handoff || {})}
        </section>
        <section>
          <h4>启用 Agent / 事件</h4>
          ${renderWorkflowAgentList(agents)}
          ${renderWorkflowEventList(events)}
        </section>
      </div>
    </article>
  `;
}

function renderArtifactBlock(title, value) {
  return `
    <div class="artifact-block">
      <h5>${escapeHtml(title)}</h5>
      <pre class="mono artifact-json">${escapeHtml(formatArtifactValue(value))}</pre>
    </div>
  `;
}

function formatArtifactValue(value) {
  if (value === undefined || value === null || value === "") return "暂无";
  if (typeof value === "string") return value;
  if (Array.isArray(value) && !value.length) return "[]";
  if (typeof value === "object" && !Object.keys(value).length) return "{}";
  try {
    return JSON.stringify(value, null, 2);
  } catch (_error) {
    return String(value);
  }
}

function workflowNodeReasoningSummary(payload, node) {
  if (node.id === "intake") {
    return `Coordinator 将用户目标归一化，并形成难度 ${labelStatus(payload.routing?.difficulty || "idle")}、工作流 ${labelStatus(payload.workflow?.variant || "idle")} 的初步上下文。`;
  }
  if (node.id === "plan") {
    const reviewer = payload.reviewer || {};
    return `Planner 使用 Coordinator 路由结果生成执行计划；Reviewer 状态为 ${labelStatus(reviewer.status || "idle")}，用于纠正过松的难度、风险或证据判断。`;
  }
  if (node.id === "retrieve") {
    const external = payload.external_evidence || {};
    return `Retriever 按证据策略检索本地 RAG 和外部来源；已尝试 ${escapeHtml((external.attempted_sources || []).join(", ") || "无外部来源")}。`;
  }
  if (node.id === "reason") {
    return `Reasoner 只基于当前路由、记忆和证据上下文生成可发布答案草案；证据不足时会输出受限答案或等待门禁处理。`;
  }
  if (node.id === "evidence_audit") {
    return `Critic 审计关键结论是否有证据支撑；当前证据状态为 ${labelStatus(payload.evidence?.status || "idle")}。`;
  }
  if (node.id === "gate") {
    return `Critic 执行风险、证据和权限门禁；当前已有 ${payload.gates?.count || 0} 条门禁决策。`;
  }
  if (node.id === "act_or_answer") {
    return `Coordinator 根据门禁结果决定发布答案、给出受限答案、拒绝高风险动作，或等待补充证据。`;
  }
  if (node.id === "verify") {
    return `Critic 对输出、证据状态和门禁状态做一致性验证；若失败会回到推理或检索环节。`;
  }
  if (node.id === "consolidate") {
    return `Memory Agent 只生成证据沉淀或记忆候选提案，不直接修改长期记忆、知识库或技能。`;
  }
  if (node.id === "respond") {
    return `Coordinator 汇总最终答案和调试遥测，把参考文献索引留在证据栏中。`;
  }
  return node.observed || "暂无可审计摘要。";
}

function workflowEdgeTransitionDetails(payload, edge, source, target) {
  const details = [];
  details.push(`边类型: ${labelStatus(edge.type || "flow")}；${edge.label || "默认进入下一节点"}`);
  if (edge.from === "intake" || edge.to === "plan" || edge.from === "plan") {
    details.push(
      `路由审查: Coordinator=${labelStatus(payload.routing?.source || "idle")}，Reviewer=${labelStatus(payload.reviewer?.status || "idle")}。`,
    );
  }
  if (edge.from === "plan" && edge.to === "retrieve") {
    details.push(`证据触发: ${labelStatus(payload.evidence?.mode || "idle")} / ${labelStatus(payload.evidence?.strictness || "none")}。`);
  }
  if (edge.from === "plan" && edge.to === "reason") {
    details.push("普通路径: 当证据不强制时可跳过检索，但仍保留 Reviewer 的升级能力。");
  }
  if (edge.from === "evidence_audit" || edge.to === "gate") {
    details.push(`证据审计: ${labelStatus(payload.evidence?.status || "idle")}；合格证据 ${payload.evidence?.qualified_evidence_ids?.length || 0} 条。`);
  }
  if (edge.from === "gate" || edge.to === "gate" || edge.to === "act_or_answer") {
    details.push(workflowGateSummary(payload));
  }
  if (["retry", "revise", "loop"].includes(edge.type || "")) {
    details.push(`回路原因: ${edge.label || "需要补证、复审或修正后再进入主流程"}`);
  }
  details.push(`Agent 转交: ${source.agent || "-"} -> ${target.agent || "-"}`);
  return details;
}

function workflowGateSummary(payload) {
  const decisions = payload.gates?.decisions || [];
  if (!decisions.length) return "门禁: 当前还没有产生门禁决策。";
  return decisions
    .map((decision) => {
      const reasons = (decision.reasons || []).join("; ");
      return `门禁 ${decision.gate_id}: ${labelStatus(decision.status)} / 风险 ${labelRisk(decision.risk_level)}${reasons ? ` / ${reasons}` : ""}`;
    })
    .join("\n");
}

function workflowAgentsForNode(payload, node) {
  const names = new Set([node.agent].filter(Boolean));
  if (node.id === "intake" || node.id === "plan") {
    names.add("coordinator");
    names.add("reviewer");
  }
  if (["evidence_audit", "gate", "verify"].includes(node.id)) names.add("critic");
  if (node.id === "retrieve") names.add("retriever");
  if (node.id === "consolidate") names.add("memory");
  return workflowAgentRecords(payload, names);
}

function workflowAgentsForEdge(payload, source, target, edge) {
  const names = new Set([source.agent, target.agent].filter(Boolean));
  if (edge.from === "intake" || edge.from === "plan" || edge.to === "plan") names.add("reviewer");
  if (edge.from === "evidence_audit" || edge.to === "gate" || edge.from === "gate" || edge.to === "verify") names.add("critic");
  if (["retry", "loop"].includes(edge.type || "")) names.add("retriever");
  return workflowAgentRecords(payload, names);
}

function workflowAgentRecords(payload, names) {
  const byName = new Map((payload.agents || []).map((agent) => [agent.name, agent]));
  return [...names].map((name) => byName.get(name) || { name, status: "not_available", description: "" });
}

function workflowEventsForNode(payload, node) {
  const events = payload.events || [];
  const stageEvents = events.filter((event) => event.stage === node.id);
  const agentEvents = events.filter((event) => !event.stage && event.agent === node.agent);
  return [...stageEvents, ...agentEvents].slice(-8);
}

function workflowEventsForEdge(payload, edge, source, target) {
  const events = payload.events || [];
  const agents = new Set([source.agent, target.agent, "reviewer", "critic"].filter(Boolean));
  return events
    .filter((event) => event.stage === edge.from || event.stage === edge.to || (!event.stage && agents.has(event.agent)))
    .slice(-10);
}

function renderWorkflowAgentList(agents) {
  if (!agents.length) return `<p class="empty">暂无 Agent 记录。</p>`;
  return agents
    .map(
      (agent) => `
        <p>
          <strong>${escapeHtml(agent.name)}</strong>
          <span class="badge ${escapeHtml(agent.status || "idle")}">${escapeHtml(labelStatus(agent.status || "idle"))}</span>
          <span class="mono">${escapeHtml(agent.last_event || agent.current_stage || agent.description || "")}</span>
        </p>
      `,
    )
    .join("");
}

function renderWorkflowEventList(events) {
  if (!events.length) return `<p class="empty">暂无相关事件。</p>`;
  return events
    .map(
      (event) => `
        <p>
          <strong>${escapeHtml(event.agent || "-")}</strong>
          <span class="badge">${escapeHtml(labelStatus(event.kind || "event"))}</span>
          <span class="mono">${escapeHtml(event.stage || "")}${event.duration_ms !== undefined ? ` | ${escapeHtml(formatDuration(event.duration_ms))}` : ""}</span>
          ${escapeHtml(event.message || "")}
        </p>
      `,
    )
    .join("");
}

function renderWorkflowTab(target, payload) {
  const workflow = workflowDisplayGraph(payload);
  const nodes = workflow.nodes || [];
  const edges = workflow.edges || [];
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>工作流状态</strong> <span class="badge ${escapeHtml(workflow.status || "idle")}">${escapeHtml(labelStatus(workflow.status || "idle"))}</span></p>
      <p>当前节点: ${escapeHtml(workflow.current || "idle")} | 工作流类型: ${escapeHtml(labelStatus(workflow.variant || "idle"))}</p>
      <p class="mono">检查点: ${escapeHtml((workflow.checkpoints || []).join(", "))}</p>
    </article>
    ${renderWorkflowEditor(payload)}
    ${renderWorkflowSelectionPanel(payload)}
    ${nodes
      .map(
        (node) => `
      <article class="detail-item">
        <p>
          <strong>${escapeHtml(node.id)}</strong>
          <span class="badge ${escapeHtml(node.status || "pending")}">${escapeHtml(labelStatus(node.status || "pending"))}</span>
          <span class="badge ${escapeHtml(node.effective_status || node.status || "pending")}">${escapeHtml(labelStatus(node.effective_status || node.status || "pending"))}</span>
        </p>
        <p>${escapeHtml(node.description || "")}</p>
        <p class="mono">负责 Agent: ${escapeHtml(node.agent)} | 输入: ${escapeHtml(node.input)} | 输出: ${escapeHtml(node.output)}</p>
        <p class="mono">实际产出: ${escapeHtml(node.work_done ? "有" : "无")} | ${escapeHtml(node.skip_reason || "无跳过原因")} | ${escapeHtml(node.observed || "")}${node.duration_ms !== undefined ? ` | ${escapeHtml(formatDuration(node.duration_ms))}` : ""}</p>
      </article>
    `,
      )
      .join("")}
    <article class="detail-item">
      <p><strong>控制流边</strong></p>
      <p class="mono">${escapeHtml(edges.map((edge) => `${edge.from} -> ${edge.to} [${edge.type || "flow"}] ${edge.label || ""}`).join("\n"))}</p>
    </article>
  `;
}

function renderWorkflowEditor(payload) {
  const specPayload = state.workflowSpecPayload;
  if (!specPayload) {
    return `
      <article class="detail-item workflow-editor">
        <header class="workflow-editor-head">
          <div>
            <p><strong>工作流编辑器</strong></p>
            <p>尚未加载 workflow spec。</p>
          </div>
          <button type="button" class="secondary" data-workflow-action="load-spec">加载 Spec</button>
        </header>
      </article>
    `;
  }
  const spec = editableWorkflowSpec(payload);
  const validation = specPayload.validation || {};
  const proposal = state.workflowProposal;
  const selected = state.selectedWorkflowElement;
  const protectedNodes = new Set(spec.protected_nodes || []);
  return `
    <article class="detail-item workflow-editor ${state.workflowEditMode ? "editing" : ""}">
      <header class="workflow-editor-head">
        <div>
          <p><strong>工作流编辑器</strong> <span class="badge ${validation.ok ? "completed" : "interrupt"}">${validation.ok ? "校验通过" : "需要处理"}</span></p>
          <p class="mono">Spec: ${escapeHtml(spec.name || "workflow")} | revision=${escapeHtml(spec.revision || "dev")} | draft=${specPayload.draft ? "yes" : "no"}</p>
          ${renderWorkflowBuilderScope(spec)}
        </div>
        <div class="workflow-editor-actions">
          <button type="button" class="secondary" data-workflow-action="toggle-edit">${state.workflowEditMode ? "退出编辑" : "编辑工作流"}</button>
          <button type="button" class="secondary" data-workflow-action="load-spec">刷新 Spec</button>
          <button type="button" data-workflow-action="save-draft" ${state.workflowEditMode ? "" : "disabled"}>保存草稿</button>
          <button type="button" data-workflow-action="create-proposal" ${state.workflowEditMode ? "" : "disabled"}>生成提案</button>
          <button type="button" class="danger" data-workflow-action="apply-proposal" ${proposal?.proposal_id ? "" : "disabled"}>批准应用</button>
        </div>
      </header>
      ${renderWorkflowValidation(validation)}
      ${
        state.workflowEditMode
          ? `
        <div class="workflow-editor-toolbar">
          <button type="button" data-workflow-action="add-node">添加节点</button>
          <button type="button" data-workflow-action="add-domain-review">添加领域审查节点</button>
          <button type="button" data-workflow-action="add-edge">添加连线</button>
          <button type="button" class="danger" data-workflow-action="delete-selected" ${selected ? "" : "disabled"}>删除所选</button>
        </div>
        ${renderWorkflowObjectEditor(spec, selected, protectedNodes)}
      `
          : `<p class="empty">进入编辑模式后，可以在图上选择节点或连线，修改 Agent、工作内容、门禁、规划化交付和连接关系。</p>`
      }
      ${proposal ? renderWorkflowProposal(proposal) : ""}
      <p class="empty">需要 AI 生成草稿时，请到左侧对话区切换“搭建助手”，或在普通对话中用 /配置、/搭建 触发。</p>
    </article>
  `;
}

function renderWorkflowBuilderScope(spec) {
  if (!builderWorkflowLayerActive(spec)) return "";
  const protectedNodes = new Set(spec.protected_nodes || []);
  const visible = builderVisibleWorkflowNodes(spec);
  const designCount = visible.filter((node) => !protectedNodes.has(node.id)).length;
  const protectedCount = (spec.nodes || []).filter((node) => protectedNodes.has(node.id)).length;
  return `<p class="mono">搭建视图: ${designCount} 个设计节点；${protectedCount} 个底座节点已隐藏但会随提案保留。</p>`;
}

function renderWorkflowValidation(validation) {
  const errors = validation.errors || [];
  const warnings = validation.warnings || [];
  const requiresCode = validation.requires_code || [];
  if (!errors.length && !warnings.length && !requiresCode.length) {
    return `<p class="mono">校验: workflow spec 可运行。</p>`;
  }
  return `
    <div class="workflow-validation">
      ${errors.map((item) => `<p class="error">错误: ${escapeHtml(item)}</p>`).join("")}
      ${requiresCode.map((item) => `<p class="warn">需要代码: ${escapeHtml(item)}</p>`).join("")}
      ${warnings.map((item) => `<p class="warn">提醒: ${escapeHtml(item)}</p>`).join("")}
    </div>
  `;
}

function renderWorkflowObjectEditor(spec, selection, protectedNodes) {
  if (!selection) return `<p class="empty">点击图上的节点或连线后，这里会显示可编辑属性。</p>`;
  if (selection.kind === "edge") {
    const edge = (spec.edges || []).find(
      (item) =>
        item.id === selection.id ||
        (item.from === selection.from && item.to === selection.to && (item.type || "flow") === selection.type),
    );
    return edge ? renderWorkflowEdgeEditor(spec, edge) : `<p class="empty">未找到所选连线。</p>`;
  }
  const node = (spec.nodes || []).find((item) => item.id === selection.id);
  return node ? renderWorkflowNodeEditor(node, protectedNodes.has(node.id)) : `<p class="empty">未找到所选节点。</p>`;
}

function renderWorkflowNodeEditor(node, isProtected) {
  return `
    <div class="workflow-edit-grid" data-editor-kind="node">
      <label>节点 ID<input id="workflowNodeId" value="${escapeHtml(node.id)}" ${isProtected ? "readonly" : ""}></label>
      <label>显示名称<input id="workflowNodeLabel" value="${escapeHtml(node.label || "")}"></label>
      <label>激活 Agent<input id="workflowNodeAgent" value="${escapeHtml(node.agent || "coordinator")}"></label>
      <label>Handler 类型
        <select id="workflowNodeHandlerKind">
          ${["builtin", "plugin_tool"].map((item) => `<option value="${item}" ${item === node.handler_kind ? "selected" : ""}>${item}</option>`).join("")}
        </select>
      </label>
      <label>Handler<input id="workflowNodeHandler" value="${escapeHtml(node.handler || node.id)}"></label>
      <label class="checkbox-row"><input id="workflowNodeCheckpoint" type="checkbox" ${node.checkpoint ? "checked" : ""}> 检查点</label>
      <label class="span-2">描述<input id="workflowNodeDescription" value="${escapeHtml(node.description || "")}"></label>
      <label class="span-2">工作内容<textarea id="workflowNodeWork">${escapeHtml(node.work || "")}</textarea></label>
      <label>输入契约<textarea id="workflowNodeInput">${escapeHtml(node.input_contract || "")}</textarea></label>
      <label>输出契约<textarea id="workflowNodeOutput">${escapeHtml(node.output_contract || "")}</textarea></label>
      <label>门禁策略 JSON<textarea id="workflowNodeGate">${escapeHtml(JSON.stringify(node.gate_policy || {}, null, 2))}</textarea></label>
      <label>UI JSON<textarea id="workflowNodeUi">${escapeHtml(JSON.stringify(node.ui || {}, null, 2))}</textarea></label>
      <div class="span-2 workflow-editor-actions">
        <button type="button" data-workflow-action="save-node">更新节点</button>
        <button type="button" class="danger" data-workflow-action="delete-selected" ${isProtected ? "disabled" : ""}>删除节点</button>
      </div>
    </div>
  `;
}

function renderWorkflowEdgeEditor(spec, edge) {
  const nodeOptions = (selected) =>
    (spec.nodes || [])
      .map((node) => `<option value="${escapeHtml(node.id)}" ${node.id === selected ? "selected" : ""}>${escapeHtml(node.id)}</option>`)
      .join("");
  return `
    <div class="workflow-edit-grid" data-editor-kind="edge">
      <label>连线 ID<input id="workflowEdgeId" value="${escapeHtml(edge.id || `${edge.from}->${edge.to}`)}"></label>
      <label>类型<input id="workflowEdgeType" value="${escapeHtml(edge.type || "flow")}"></label>
      <label>起点<select id="workflowEdgeFrom">${nodeOptions(edge.from)}</select></label>
      <label>终点<select id="workflowEdgeTo">${nodeOptions(edge.to)}</select></label>
      <label class="span-2">条件 / 标签<input id="workflowEdgeCondition" value="${escapeHtml(edge.condition || edge.label || "")}"></label>
      <label class="checkbox-row"><input id="workflowEdgeReviewer" type="checkbox" ${edge.reviewer_required ? "checked" : ""}> 需要 Reviewer</label>
      <label>交付契约 JSON<textarea id="workflowEdgeHandoff">${escapeHtml(JSON.stringify(edge.handoff_contract || {}, null, 2))}</textarea></label>
      <label>门禁策略 JSON<textarea id="workflowEdgeGate">${escapeHtml(JSON.stringify(edge.gate_policy || {}, null, 2))}</textarea></label>
      <label>规划契约 JSON<textarea id="workflowEdgePlanner">${escapeHtml(JSON.stringify(edge.planner_contract || {}, null, 2))}</textarea></label>
      <div class="span-2 workflow-editor-actions">
        <button type="button" data-workflow-action="save-edge">更新连线</button>
        <button type="button" class="danger" data-workflow-action="delete-selected">删除连线</button>
      </div>
    </div>
  `;
}

function renderWorkflowProposal(proposal) {
  return `
    <article class="proposal-preview">
      <p><strong>待应用提案</strong> <span class="badge pending">${escapeHtml(proposal.proposal_id)}</span></p>
      <p class="mono">draft_hash=${escapeHtml(proposal.draft_hash || "")} | files=${escapeHtml((proposal.modified_files || []).join(", "))}</p>
      <pre class="mono artifact-json">${escapeHtml(proposal.diff_preview || "暂无 diff。")}</pre>
    </article>
  `;
}

function renderGatesTab(target, payload) {
  const items = payload.gates?.decisions || [];
  target.innerHTML = items.length
    ? items
        .map(
          (item) => `
        <article class="detail-item">
          <p><strong>${escapeHtml(item.gate_id)}</strong> <span class="badge ${escapeHtml(item.status)}">${escapeHtml(labelStatus(item.status))}</span></p>
          <p>风险: ${escapeHtml(labelRisk(item.risk_level))} | 证据: ${escapeHtml((item.required_evidence || []).join(", "))}</p>
          <p class="mono">${escapeHtml((item.reasons || []).join("; ") || "门禁无异常。")}</p>
        </article>
      `,
        )
        .join("")
    : `<p class="empty">暂无门禁决策。</p>`;
}

function renderMemoryTab(target, payload) {
  const memory = payload.memory || {};
  const shortTerm = memory.short_term || {};
  const longTerm = memory.long_term || {};
  const boundaries = memory.boundaries || {};
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>策略</strong> ${escapeHtml(memory.policy || "configured")}</p>
      <p>分区: ${escapeHtml((memory.partitions || []).join(", "))}</p>
      <p>只读: ${escapeHtml((memory.read_only || []).join(", "))}</p>
      <p>Thread: ${escapeHtml(shortTerm.thread_id || payload.thread_id || state.threadId)}</p>
      <p>短期记忆: ${escapeHtml(shortTerm.turns || 0)} 轮 | 长期记忆: ${escapeHtml(longTerm.records || 0)} 条 | 本轮写入: ${escapeHtml(longTerm.writes || 0)} 条</p>
      <p class="mono">${escapeHtml((memory.pending_consolidation || []).join("\n") || "暂无待沉淀内容。")}</p>
    </article>
    <article class="detail-item">
      <p><strong>硬边界</strong></p>
      <p class="mono">${escapeHtml(Object.entries(boundaries).map(([key, value]) => `${key}: ${value}`).join("\n") || "暂无边界说明。")}</p>
      <p class="mono">已知 thread: ${escapeHtml((shortTerm.threads || []).join(", ") || shortTerm.thread_id || state.threadId)}</p>
    </article>
    <article class="detail-item">
      <p><strong>短期记忆</strong></p>
      <p class="mono">${escapeHtml((shortTerm.items || []).map((item) => `用户: ${item.user}\nAgent: ${item.assistant}`).join("\n\n") || "暂无短期对话记忆。")}</p>
    </article>
    <article class="detail-item">
      <p><strong>长期记忆</strong></p>
      <p class="mono">${escapeHtml((longTerm.items || []).map((item) => `${item.partition}/${item.key}: ${item.value}`).join("\n") || "暂无长期记忆。")}</p>
    </article>
    <article class="detail-item">
      <p><strong>记忆写入门禁</strong></p>
      <p class="mono">${escapeHtml((longTerm.write_decisions || []).map((item) => `${item.partition}/${item.key}: ${item.status} ${(item.reasons || []).join("; ")}`).join("\n") || "本轮没有长期记忆写入。")}</p>
    </article>
  `;
}

function renderSkillsTab(target, payload) {
  const skills = payload.skills || {};
  target.innerHTML = `
    <article class="detail-item">
      <p><strong>${skills.count || 0} 个技能包已加载</strong></p>
      <p class="mono">${escapeHtml((skills.enabled || skills.loaded || []).join("\n"))}</p>
    </article>
  `;
}

function renderEventsTab(target, payload) {
  const events = payload.events || [];
  target.innerHTML = events.length
    ? events
        .map(
          (event) => `
        <article class="detail-item">
          <p><strong>${escapeHtml(event.agent)}</strong> <span class="badge">${escapeHtml(labelStatus(event.kind))}</span></p>
          <p>${escapeHtml(event.message)}</p>
          <p class="mono">${escapeHtml(event.time)}${event.duration_ms !== undefined ? ` | ${escapeHtml(formatDuration(event.duration_ms))}` : ""}${event.evidence_id ? ` | ${escapeHtml(event.evidence_id)}` : ""}</p>
        </article>
      `,
        )
        .join("")
    : `<p class="empty">暂无事件。</p>`;
}

function editableWorkflowSpec(payload = state.payload || {}) {
  const specPayload = state.workflowSpecPayload || {};
  return deepClone(specPayload.draft || specPayload.spec || workflowSpecFromRuntime(payload.workflow || {}));
}

function workflowSpecFromRuntime(workflow) {
  const nodes = workflow.nodes || [];
  return {
    version: "1.0",
    name: workflow.spec?.name || "runtime-workflow",
    revision: workflow.spec?.revision || "runtime",
    start_node: workflow.spec?.start_node || nodes[0]?.id || "intake",
    terminal_nodes: workflow.spec?.terminal_nodes || ["respond"],
    protected_nodes: nodes.map((node) => node.id),
    nodes: nodes.map((node) => ({
      id: node.id,
      label: node.label || workflowNodeLabel(node.id),
      agent: node.agent || "coordinator",
      description: node.description || "",
      work: node.work || node.description || "",
      input_contract: node.input || "",
      output_contract: node.output || "",
      handler_kind: node.handler_kind || "builtin",
      handler: node.handler || node.id,
      checkpoint: Boolean(node.checkpoint),
      gate_policy: node.gate_policy || {},
      ui: node.ui || {},
    })),
    edges: (workflow.edges || []).map((edge) => ({
      id: edge.id || `${edge.from}->${edge.to}`,
      from: edge.from,
      to: edge.to,
      type: edge.type || "flow",
      condition: edge.condition || edge.label || "",
      handoff_contract: edge.handoff_contract || {},
      gate_policy: edge.gate_policy || {},
      planner_contract: edge.planner_contract || {},
      reviewer_required: Boolean(edge.reviewer_required),
    })),
  };
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function linesFromTextarea(id) {
  const node = $(id);
  return node ? node.value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean) : [];
}

async function loadWorkflowSpec({ render = false } = {}) {
  state.workflowSpecPayload = await api("/api/workflow/spec");
  if (render && state.payload) renderStatus(state.payload);
  return state.workflowSpecPayload;
}

async function loadAgentSpec({ render = false } = {}) {
  state.agentSpecPayload = await api("/api/agents/spec");
  if (render && state.payload) renderStatus(state.payload);
  return state.agentSpecPayload;
}

async function loadTemplates({ render = false } = {}) {
  state.templatePayload = await api("/api/templates");
  const ids = new Set((state.templatePayload.templates || []).map((item) => item.id));
  if (!ids.has(state.selectedTemplateId)) {
    state.selectedTemplateId = ids.has("autoresearch-agent")
      ? "autoresearch-agent"
      : state.templatePayload.templates?.[0]?.id || "";
  }
  if (render && state.payload) renderStatus(state.payload);
  return state.templatePayload;
}

function setWorkflowDraft(spec) {
  state.workflowSpecPayload = state.workflowSpecPayload || {};
  state.workflowSpecPayload.draft = spec;
  state.workflowSpecPayload.validation = state.workflowSpecPayload.validation || { ok: false, errors: [], warnings: ["草稿尚未保存校验"], requires_code: [] };
  if (state.payload) renderStatus(state.payload);
}

function editableAgentsSpec() {
  const specPayload = state.agentSpecPayload || {};
  return deepClone(specPayload.draft || specPayload.spec || { version: "1.0", name: "agents", revision: "draft", protected_agents: [], agents: [] });
}

function setAgentDraft(spec, validation = null) {
  state.agentSpecPayload = state.agentSpecPayload || {};
  state.agentSpecPayload.draft = spec;
  state.agentSpecPayload.validation = validation || state.agentSpecPayload.validation || { ok: false, errors: [], warnings: ["草稿尚未保存校验"] };
  if (state.payload) renderStatus(state.payload);
}

function templateSlug(value) {
  const ascii = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
  if (/^[a-z]/.test(ascii)) return ascii;
  return `template-${Date.now().toString(36)}`;
}

function currentTemplateAgentsSpec() {
  const spec = editableAgentsSpec();
  return state.agentEditMode ? updateSelectedAgentObject(spec) : spec;
}

function currentTemplateWorkflowSpec() {
  const spec = editableWorkflowSpec(state.payload || {});
  return state.workflowEditMode ? updateSelectedWorkflowObject(spec) : spec;
}

async function applySelectedTemplate() {
  const templateId = $("templateSelect")?.value || state.selectedTemplateId;
  if (!templateId) return;
  const payload = await api("/api/templates/apply", {
    method: "POST",
    body: JSON.stringify({ template_id: templateId }),
  });
  state.selectedTemplateId = templateId;
  state.agentSpecPayload = payload.agents;
  state.workflowSpecPayload = payload.workflow;
  state.agentEditMode = true;
  state.workflowEditMode = true;
  state.agentProposal = null;
  state.workflowProposal = null;
  state.selectedAgentName = chooseGeneratedAgent(payload.agents.spec);
  state.selectedWorkflowElement = chooseGeneratedWorkflowNode(payload.workflow.spec);
  appendMessage("agent", `已加载模板“${payload.template?.label || templateId}”到 Agent/Workflow 草稿。请人工检查后再生成提案并批准应用。`);
  await refreshStatus();
}

async function saveCurrentTemplate() {
  const label = $("templateSaveLabel")?.value.trim() || "未命名 Agent 模板";
  const templateId = $("templateSaveId")?.value.trim() || templateSlug(label);
  const description = $("templateSaveDescription")?.value.trim() || "从调试台保存的 Agent/Workflow 草稿。";
  const payload = await api("/api/templates/save", {
    method: "POST",
    body: JSON.stringify({
      template_id: templateId,
      label,
      description,
      agents: currentTemplateAgentsSpec(),
      workflow: currentTemplateWorkflowSpec(),
    }),
  });
  state.selectedTemplateId = payload.template?.id || templateId;
  await loadTemplates();
  appendMessage("agent", `已保存模板“${payload.template?.label || label}”。之后可以在模板库里快速切换。`);
  renderStatus(state.payload || {});
}

async function handleTemplateAction(action) {
  try {
    if (action === "refresh") return loadTemplates({ render: true });
    if (action === "apply") return applySelectedTemplate();
    if (action === "save") return saveCurrentTemplate();
  } catch (error) {
    appendMessage("agent", `模板库错误: ${error.message}`);
  }
}

function updateSelectedAgentObject(spec) {
  const selected = state.selectedAgentName;
  if (!selected || !$("agentId")) return spec;
  const index = (spec.agents || []).findIndex((agent) => agent.id === selected);
  if (index < 0) return spec;
  const previousId = spec.agents[index].id;
  const nextId = $("agentId").value.trim() || previousId;
  const agent = {
    ...spec.agents[index],
    id: nextId,
    label: $("agentLabel").value.trim() || nextId,
    description: $("agentDescription").value.trim(),
    model_role: $("agentModelRole").value,
    responsibilities: linesFromTextarea("agentResponsibilities"),
    tools: linesFromTextarea("agentTools"),
    memory_access: linesFromTextarea("agentMemoryAccess"),
    workflow_nodes: linesFromTextarea("agentWorkflowNodes"),
    permissions: parseJsonField("agentPermissions", {}),
    handoff_contract: parseJsonField("agentHandoff", {}),
  };
  spec.agents[index] = agent;
  if (nextId !== previousId) {
    spec.protected_agents = (spec.protected_agents || []).map((item) => (item === previousId ? nextId : item));
    state.selectedAgentName = nextId;
  }
  return spec;
}

function addAgent() {
  const spec = editableAgentsSpec();
  const id = nextAgentId(spec, "custom_agent");
  spec.agents = spec.agents || [];
  spec.agents.push({
    id,
    label: "自定义 Agent",
    description: "新的可配置 Agent。",
    responsibilities: ["补充这个 Agent 的职责"],
    model_role: "worker",
    tools: [],
    permissions: {},
    memory_access: [],
    workflow_nodes: [],
    handoff_contract: {},
    ui: {},
  });
  state.selectedAgentName = id;
  state.agentEditMode = true;
  setAgentDraft(spec);
}

function deleteSelectedAgent() {
  const selected = state.selectedAgentName;
  if (!selected) return;
  const spec = editableAgentsSpec();
  if ((spec.protected_agents || []).includes(selected)) {
    appendMessage("agent", `受保护 Agent ${selected} 不能删除。`);
    return;
  }
  spec.agents = (spec.agents || []).filter((agent) => agent.id !== selected);
  state.selectedAgentName = null;
  setAgentDraft(spec);
}

function nextAgentId(spec, base) {
  const ids = new Set((spec.agents || []).map((agent) => agent.id));
  if (!ids.has(base)) return base;
  let index = 2;
  while (ids.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

async function saveAgentDraft() {
  const spec = updateSelectedAgentObject(editableAgentsSpec());
  const payload = await api("/api/agents/draft", { method: "POST", body: JSON.stringify({ spec }) });
  state.agentSpecPayload = { ...(state.agentSpecPayload || {}), draft: payload.spec, validation: payload.validation };
  renderStatus(state.payload || {});
}

async function createAgentProposal() {
  await saveAgentDraft();
  state.agentProposal = await api("/api/agents/proposal", { method: "POST", body: JSON.stringify({}) });
  renderStatus(state.payload || {});
}

async function applyAgentProposal() {
  if (!state.agentProposal?.proposal_id) return;
  const result = await api("/api/agents/apply", {
    method: "POST",
    body: JSON.stringify({
      proposal_id: state.agentProposal.proposal_id,
      approved: true,
      approved_by: "web-debugger",
    }),
  });
  state.agentProposal = null;
  state.agentSpecPayload = result.agents;
  state.agentEditMode = false;
  await refreshStatus();
}

async function handleAgentEditorAction(action) {
  try {
    if (action === "load-spec") return loadAgentSpec({ render: true });
    if (action === "toggle-edit") {
      state.agentEditMode = !state.agentEditMode;
      if (!state.agentSpecPayload) await loadAgentSpec();
      renderStatus(state.payload || {});
      return;
    }
    if (action === "add-agent") return addAgent();
    if (action === "delete-agent") return deleteSelectedAgent();
    if (action === "save-agent") {
      setAgentDraft(updateSelectedAgentObject(editableAgentsSpec()));
      return;
    }
    if (action === "save-draft") return saveAgentDraft();
    if (action === "create-proposal") return createAgentProposal();
    if (action === "apply-proposal") return applyAgentProposal();
  } catch (error) {
    appendMessage("agent", `Agent 编辑错误: ${error.message}`);
  }
}

function updateSelectedWorkflowObject(spec) {
  const selection = state.selectedWorkflowElement;
  if (!selection) return spec;
  if (selection.kind === "edge") return updateSelectedWorkflowEdge(spec, selection);
  return updateSelectedWorkflowNode(spec, selection);
}

function updateSelectedWorkflowNode(spec, selection) {
  const index = (spec.nodes || []).findIndex((node) => node.id === selection.id);
  if (index < 0 || !$("workflowNodeId")) return spec;
  const previousId = spec.nodes[index].id;
  const nextId = $("workflowNodeId").value.trim() || previousId;
  const node = {
    ...spec.nodes[index],
    id: nextId,
    label: $("workflowNodeLabel").value.trim(),
    agent: $("workflowNodeAgent").value.trim() || "coordinator",
    description: $("workflowNodeDescription").value.trim(),
    work: $("workflowNodeWork").value.trim(),
    input_contract: $("workflowNodeInput").value.trim(),
    output_contract: $("workflowNodeOutput").value.trim(),
    handler_kind: $("workflowNodeHandlerKind").value,
    handler: $("workflowNodeHandler").value.trim() || nextId,
    checkpoint: $("workflowNodeCheckpoint").checked,
    gate_policy: parseJsonField("workflowNodeGate"),
    ui: parseJsonField("workflowNodeUi"),
  };
  spec.nodes[index] = node;
  if (previousId !== nextId) {
    spec.edges = (spec.edges || []).map((edge) => ({
      ...edge,
      from: edge.from === previousId ? nextId : edge.from,
      to: edge.to === previousId ? nextId : edge.to,
    }));
    spec.protected_nodes = (spec.protected_nodes || []).map((id) => (id === previousId ? nextId : id));
    spec.start_node = spec.start_node === previousId ? nextId : spec.start_node;
    spec.terminal_nodes = (spec.terminal_nodes || []).map((id) => (id === previousId ? nextId : id));
    state.selectedWorkflowElement = { kind: "node", id: nextId };
  }
  return spec;
}

function updateSelectedWorkflowEdge(spec, selection) {
  const index = (spec.edges || []).findIndex(
    (edge) =>
      edge.id === selection.id ||
      (edge.from === selection.from && edge.to === selection.to && (edge.type || "flow") === selection.type),
  );
  if (index < 0 || !$("workflowEdgeId")) return spec;
  const edge = {
    ...spec.edges[index],
    id: $("workflowEdgeId").value.trim() || `${$("workflowEdgeFrom").value}->${$("workflowEdgeTo").value}`,
    from: $("workflowEdgeFrom").value,
    to: $("workflowEdgeTo").value,
    type: $("workflowEdgeType").value.trim() || "flow",
    condition: $("workflowEdgeCondition").value.trim(),
    reviewer_required: $("workflowEdgeReviewer").checked,
    handoff_contract: parseJsonField("workflowEdgeHandoff"),
    gate_policy: parseJsonField("workflowEdgeGate"),
    planner_contract: parseJsonField("workflowEdgePlanner"),
  };
  spec.edges[index] = edge;
  state.selectedWorkflowElement = { kind: "edge", id: edge.id, from: edge.from, to: edge.to, type: edge.type };
  return spec;
}

function parseJsonField(id) {
  const raw = $(id)?.value.trim();
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${id} 不是合法 JSON: ${error.message}`);
  }
}

function addWorkflowNode(kind = "generic") {
  const spec = editableWorkflowSpec();
  const id = nextWorkflowNodeId(spec, kind === "domain_review" ? "domain_review" : "custom_node");
  const node = {
    id,
    label: kind === "domain_review" ? "领域审查" : "新节点",
    agent: kind === "domain_review" ? "critic" : "coordinator",
    description: kind === "domain_review" ? "领域事实与交付质量审查" : "自定义工作流节点",
    work: kind === "domain_review" ? "审查领域事实、证据绑定和交付质量。" : "描述这个节点负责的工作。",
    input_contract: kind === "domain_review" ? "答案草案" : "上游交付",
    output_contract: kind === "domain_review" ? "领域审查记录" : "下游交付",
    handler_kind: "builtin",
    handler: kind === "domain_review" ? "review_note" : "passthrough",
    checkpoint: kind === "domain_review",
    gate_policy: {},
    ui: {},
  };
  const insertAt = Math.max(0, (spec.nodes || []).findIndex((item) => item.id === "evidence_audit"));
  spec.nodes.splice(insertAt >= 0 ? insertAt : spec.nodes.length, 0, node);
  if (kind === "domain_review") {
    spec.edges = (spec.edges || []).filter((edge) => !(edge.from === "reason" && edge.to === "evidence_audit"));
    spec.edges.push({
      id: "reason_to_domain_review",
      from: "reason",
      to: id,
      type: "flow",
      condition: "draft ready",
      handoff_contract: { payload: "answer_draft + evidence_context" },
      gate_policy: {},
      planner_contract: { required_output: "domain_review_notes" },
      reviewer_required: true,
    });
    spec.edges.push({
      id: "domain_review_to_evidence_audit",
      from: id,
      to: "evidence_audit",
      type: "flow",
      condition: "review complete",
      handoff_contract: { payload: "domain_review_notes" },
      gate_policy: {},
      planner_contract: {},
      reviewer_required: false,
    });
  }
  state.selectedWorkflowElement = { kind: "node", id };
  setWorkflowDraft(spec);
}

function addWorkflowEdge() {
  const spec = editableWorkflowSpec();
  const nodes = spec.nodes || [];
  if (nodes.length < 2) return;
  const from = state.selectedWorkflowElement?.kind === "node" ? state.selectedWorkflowElement.id : nodes[0].id;
  const to = nodes.find((node) => node.id !== from && node.id === "respond")?.id || nodes[nodes.length - 1].id;
  const id = nextWorkflowEdgeId(spec, `${from}_to_${to}`);
  const edge = {
    id,
    from,
    to,
    type: "flow",
    condition: "manual handoff",
    handoff_contract: {},
    gate_policy: {},
    planner_contract: {},
    reviewer_required: false,
  };
  spec.edges.push(edge);
  state.selectedWorkflowElement = { kind: "edge", id, from, to, type: "flow" };
  setWorkflowDraft(spec);
}

function deleteSelectedWorkflowObject() {
  const selection = state.selectedWorkflowElement;
  if (!selection) return;
  const spec = editableWorkflowSpec();
  if (selection.kind === "edge") {
    spec.edges = (spec.edges || []).filter(
      (edge) =>
        !(
          edge.id === selection.id ||
          (edge.from === selection.from && edge.to === selection.to && (edge.type || "flow") === selection.type)
        ),
    );
    state.selectedWorkflowElement = null;
    setWorkflowDraft(spec);
    return;
  }
  if ((spec.protected_nodes || []).includes(selection.id)) {
    appendMessage("agent", `受保护节点 ${selection.id} 不能删除。`);
    return;
  }
  spec.nodes = (spec.nodes || []).filter((node) => node.id !== selection.id);
  spec.edges = (spec.edges || []).filter((edge) => edge.from !== selection.id && edge.to !== selection.id);
  state.selectedWorkflowElement = null;
  setWorkflowDraft(spec);
}

function nextWorkflowNodeId(spec, base) {
  const ids = new Set((spec.nodes || []).map((node) => node.id));
  if (!ids.has(base)) return base;
  let index = 2;
  while (ids.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

function nextWorkflowEdgeId(spec, base) {
  const ids = new Set((spec.edges || []).map((edge) => edge.id));
  if (!ids.has(base)) return base;
  let index = 2;
  while (ids.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

async function saveWorkflowDraft() {
  const spec = updateSelectedWorkflowObject(editableWorkflowSpec());
  const payload = await api("/api/workflow/draft", { method: "POST", body: JSON.stringify({ spec }) });
  state.workflowSpecPayload = { ...(state.workflowSpecPayload || {}), draft: payload.spec, validation: payload.validation };
  renderStatus(state.payload || {});
}

async function createWorkflowProposal() {
  await saveWorkflowDraft();
  state.workflowProposal = await api("/api/workflow/proposal", { method: "POST", body: JSON.stringify({}) });
  renderStatus(state.payload || {});
}

async function applyWorkflowProposal() {
  if (!state.workflowProposal?.proposal_id) return;
  const result = await api("/api/workflow/apply", {
    method: "POST",
    body: JSON.stringify({
      proposal_id: state.workflowProposal.proposal_id,
      approved: true,
      approved_by: "web-debugger",
    }),
  });
  state.workflowProposal = null;
  state.workflowSpecPayload = result.workflow;
  state.workflowEditMode = false;
  await refreshStatus();
}

async function handleWorkflowEditorAction(action) {
  try {
    if (action === "load-spec") return loadWorkflowSpec({ render: true });
    if (action === "toggle-edit") {
      state.workflowEditMode = !state.workflowEditMode;
      if (!state.workflowSpecPayload) await loadWorkflowSpec();
      renderStatus(state.payload || {});
      return;
    }
    if (action === "add-node") return addWorkflowNode();
    if (action === "add-domain-review") return addWorkflowNode("domain_review");
    if (action === "add-edge") return addWorkflowEdge();
    if (action === "delete-selected") return deleteSelectedWorkflowObject();
    if (action === "save-node" || action === "save-edge") {
      setWorkflowDraft(updateSelectedWorkflowObject(editableWorkflowSpec()));
      return;
    }
    if (action === "save-draft") return saveWorkflowDraft();
    if (action === "create-proposal") return createWorkflowProposal();
    if (action === "apply-proposal") return applyWorkflowProposal();
  } catch (error) {
    appendMessage("agent", `工作流编辑错误: ${error.message}`);
  }
}

function setChatMode(mode) {
  state.chatMode = mode === "builder" ? "builder" : "chat";
  document.querySelectorAll("[data-chat-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.chatMode === state.chatMode);
  });
  const hint = $("conversationModeHint");
  const input = $("messageInput");
  if (state.chatMode === "builder") {
    if (hint) hint.textContent = "搭建助手模式。直接描述你想要的 Agent、工作流、门禁和交付机制；生成结果会进入右侧人工审查面板。";
    if (input) input.placeholder = "例如：我要一个材料学研究多 Agent，在推理后加入领域审查，并要求 reviewer 审查门禁...";
  } else {
    if (hint) hint.textContent = "普通对话。输入 /配置、/搭建、#配置 或 #搭建 可以临时启用搭建助手。";
    if (input) input.placeholder = "输入消息，回答将由 DeepSeek API 生成...";
  }
}

function isConfiguratorMessage(message) {
  const text = message.trim();
  return /^(?:(?:\/配置|\/搭建|#配置|#搭建|\/config|\/build)(?:\s|$)|配置助手[:：]?)/i.test(text);
}

function chooseGeneratedAgent(spec) {
  const agents = spec?.agents || [];
  if (!agents.length) return state.selectedAgentName;
  const protectedAgents = new Set(spec.protected_agents || []);
  const generated = agents.filter((agent) => !protectedAgents.has(agent.id));
  return (generated.at(-1) || agents.at(-1))?.id || state.selectedAgentName;
}

function chooseGeneratedWorkflowNode(spec) {
  const nodes = spec?.nodes || [];
  if (!nodes.length) return state.selectedWorkflowElement;
  const protectedNodes = new Set(spec.protected_nodes || []);
  const generated = nodes.filter((node) => !protectedNodes.has(node.id));
  return { kind: "node", id: (generated.at(-1) || nodes.at(-1)).id };
}

async function sendConfiguratorMessage(message) {
  $("agentStatus").textContent = "运行中";
  $("runId").textContent = "搭建助手正在生成草稿...";
  $("workingAgent").textContent = "Agent: configurator";
  $("workingStage").textContent = "节点: workflow/agents draft";
  $("workingHint").textContent = "顶层配置助手正在把自然语言需求转换为 Agent 与工作流草稿，右侧面板只负责人工审查和应用。";
  const payload = await api("/api/configurator/compose", {
    method: "POST",
    body: JSON.stringify({ prompt: message, target: "auto" }),
  });
  if (payload.agents) {
    state.agentSpecPayload = {
      ...(state.agentSpecPayload || {}),
      draft: payload.agents.spec,
      validation: payload.agents.validation,
      paths: payload.agents.paths,
    };
    state.agentEditMode = true;
    state.agentProposal = null;
    state.selectedAgentName = chooseGeneratedAgent(payload.agents.spec);
  }
  if (payload.workflow) {
    state.workflowSpecPayload = {
      ...(state.workflowSpecPayload || {}),
      draft: payload.workflow.spec,
      validation: payload.workflow.validation,
      paths: payload.workflow.paths,
    };
    state.workflowEditMode = true;
    state.workflowProposal = null;
    state.selectedWorkflowElement = chooseGeneratedWorkflowNode(payload.workflow.spec);
  }
  await refreshStatus();
  $("agentStatus").textContent = "已完成";
  $("workingAgent").textContent = "Agent: configurator";
  $("workingStage").textContent = `目标: ${(payload.targets || []).join(" + ") || "auto"}`;
  const plan = payload.builder_plan;
  const planHint = plan
    ? `规模判断 ${plan.scale}，推荐 ${plan.agent_count} 个设计 Agent / ${plan.node_count} 个设计节点。`
    : "";
  $("workingHint").textContent = `${planHint}草稿已生成，但尚未应用。请在多 Agent / 工作流面板人工检查，再保存草稿、生成提案并批准应用。`;
  return payload;
}

async function submitRagQuery(event) {
  event.preventDefault();
  const form = event.target;
  const query = form.query.value.trim();
  const methods = Array.from(form.querySelectorAll('input[name="ragMethod"]:checked')).map((item) => item.value);
  state.ragLab = { query, methods, result: null, error: "" };
  if (!query) {
    state.ragLab.error = "请输入查询。";
    renderTab();
    return;
  }
  if (!methods.length) {
    state.ragLab.error = "至少选择一种检索方法。";
    renderTab();
    return;
  }
  try {
    const payload = await api("/api/rag/query", {
      method: "POST",
      body: JSON.stringify({ query, methods, top_k: 5 }),
    });
    state.ragLab = { query, methods, result: payload, error: "" };
  } catch (error) {
    state.ragLab = { query, methods, result: null, error: error.message };
  }
  renderTab();
}

function appendMessage(role, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = `<strong>${role === "user" ? "用户" : "Agent"}</strong><p>${escapeHtml(text).replaceAll("\n", "<br>")}</p>`;
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function refreshStatus() {
  const [payload] = await Promise.all([api("/api/status"), loadWorkflowSpec(), loadAgentSpec(), loadTemplates()]);
  renderStatus(payload);
}

async function waitForRunCompletion(question) {
  while (true) {
    await sleep(550);
    const payload = await api("/api/status");
    renderStatus(payload);
    const isSameQuestion = !payload.question || payload.question === question;
    if (isSameQuestion && (payload.status === "completed" || payload.status === "failed")) {
      return payload;
    }
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $("messageInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  state.pendingQuestion = message;
  appendMessage("user", message);
  const useConfigurator = state.chatMode === "builder" || isConfiguratorMessage(message);
  $("sendButton").disabled = true;
  $("agentStatus").textContent = "运行中";
  $("runId").textContent = "后台运行中...";
  $("workingAgent").textContent = useConfigurator ? "Agent: configurator" : "Agent: coordinator";
  $("workingStage").textContent = useConfigurator ? "节点: configure" : "节点: intake";
  $("workingHint").textContent = useConfigurator ? "请求已交给顶层搭建助手。" : "请求已提交，等待 Coordinator 启动。";
  try {
    if (useConfigurator) {
      const payload = await sendConfiguratorMessage(message);
      appendMessage("agent", payload.answer || "配置助手已生成草稿，请到右侧面板人工检查。");
      return;
    }
    await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, thread_id: state.threadId, async: true }),
    });
    const payload = await waitForRunCompletion(message);
    renderStatus(payload);
    if (payload.status === "failed") {
      appendMessage("agent", `错误: ${payload.error || "运行失败"}`);
    } else {
      appendMessage("agent", payload.answer || "本轮运行已完成，但没有返回回答。");
    }
  } catch (error) {
    $("agentStatus").textContent = "错误";
    appendMessage("agent", `错误: ${error.message}`);
  } finally {
    state.pendingQuestion = null;
    $("sendButton").disabled = false;
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tab);
    renderTab();
  });
});

$("chatForm").addEventListener("submit", sendMessage);
$("refreshButton").addEventListener("click", refreshStatus);
document.addEventListener("submit", (event) => {
  if (event.target?.id === "ragQueryForm") {
    submitRagQuery(event);
  }
});
document.addEventListener("change", (event) => {
  if (event.target?.id === "templateSelect") {
    state.selectedTemplateId = event.target.value;
    if (state.payload) renderStatus(state.payload);
  }
});
document.querySelectorAll("[data-chat-mode]").forEach((button) => {
  button.addEventListener("click", () => setChatMode(button.dataset.chatMode));
});
document.addEventListener("click", (event) => {
  const templateButton = event.target.closest("[data-template-action]");
  if (templateButton) {
    event.preventDefault();
    handleTemplateAction(templateButton.dataset.templateAction);
    return;
  }
  const agentSelect = event.target.closest("[data-agent-select]");
  if (agentSelect) {
    state.selectedAgentName = agentSelect.dataset.agentSelect;
    if (state.agentEditMode) renderStatus(state.payload || {});
    return;
  }
  const agentButton = event.target.closest("[data-agent-action]");
  if (agentButton) {
    event.preventDefault();
    handleAgentEditorAction(agentButton.dataset.agentAction);
    return;
  }
  const button = event.target.closest("[data-workflow-action]");
  if (!button) return;
  event.preventDefault();
  handleWorkflowEditorAction(button.dataset.workflowAction);
});

setChatMode("chat");
refreshStatus().catch((error) => appendMessage("agent", `状态加载失败: ${error.message}`));

