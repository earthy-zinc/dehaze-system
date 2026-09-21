import {
  AiAgentAPI,
  AgentConfig,
  AgentDetail,
  AgentListItem,
  AgentPageQuery,
  AgentSubAgentItem,
  AgentVersionDetail,
  AgentVersionResult,
  EndpointCreateForm,
  EndpointResult,
  EndpointUpdateForm,
  EvalDatasetCreateForm,
  EvalDatasetResult,
  EvalDatasetUpdateForm,
  EvalRunGateResult,
  EvalRunResult,
  EvalRunTaskResult,
  EvalSampleCreateForm,
  EvalSampleResult,
  EvalSampleUpdateForm,
  EnabledStatus,
  ReasoningMode,
  VersionResult,
} from "dehaze-sdk-js";

/** Agent 配置表单提交载荷：主表字段 + 覆盖式关联（Skills/MCP/子 Agent） */
export interface AgentFormPayload {
  name: string;
  agentCode: string;
  description: string;
  /** 分类标签（逗号分隔输入，提交前拆分为数组） */
  tags: string[];
  systemPrompt: string | null;
  modelId: string;
  reasoningMode: ReasoningMode;
  config: AgentConfig;
  isSubagent: boolean;
  isTeam: boolean;
  isExposed: boolean;
  permissions: Array<Record<string, unknown>>;
  sortOrder: number;
  /** 仅创建时生效 */
  status?: EnabledStatus;
  skills: string[];
  mcpNamespaces: string[];
  subagents: AgentSubAgentItem[];
}

// 管理端智能体 Store：列表/类型筛选/配置表单（草稿）/版本发布/评测/A2A 端点
export const useAdminAgentStore = defineStore("adminAgent", () => {
  // ==================== 列表 ====================
  const agents = ref<AgentListItem[]>([]);
  const total = ref(0);
  const loading = ref(false);
  const query = reactive<AgentPageQuery>({ pageNum: 1, pageSize: 10 });
  const agentTypeFilter = ref<"all" | "agent" | "subagent" | "team">("all");

  async function fetchAgents() {
    loading.value = true;
    try {
      const page = await AiAgentAPI.list({
        ...query,
        type:
          agentTypeFilter.value === "all" ? undefined : agentTypeFilter.value,
      });
      agents.value = page.list ?? [];
      total.value = page.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  // ==================== 配置表单（草稿快照） ====================
  const agentForm = reactive<{ visible: boolean; agentId: number | null }>({
    visible: false,
    agentId: null,
  });
  const detail = ref<AgentDetail | null>(null);
  const detailLoading = ref(false);

  async function fetchAgentDetail(agentId: number) {
    detailLoading.value = true;
    try {
      detail.value = await AiAgentAPI.detail(agentId);
      return detail.value;
    } finally {
      detailLoading.value = false;
    }
  }

  /**
   * 保存 Agent 配置：更新生成草稿快照，发布后对新会话生效。
   * Skills/MCP 命名空间/子 Agent 为覆盖式关联端点，与主表更新一并提交。
   */
  async function saveAgent(form: AgentFormPayload, agentId: number | null) {
    const {
      agentCode,
      status,
      skills,
      mcpNamespaces,
      subagents,
      ...updateForm
    } = form;
    if (agentId == null) {
      const created = await AiAgentAPI.create({
        ...updateForm,
        agentCode,
        status,
      });
      if (skills.length) {
        await AiAgentAPI.setSkills(created.id, { skills });
      }
      if (mcpNamespaces.length) {
        await AiAgentAPI.setMcps(created.id, { mcpNamespaces });
      }
      if (subagents.length) {
        await AiAgentAPI.setSubAgents(created.id, { subagents });
      }
    } else {
      await AiAgentAPI.update(agentId, updateForm);
      await AiAgentAPI.setSkills(agentId, { skills });
      await AiAgentAPI.setMcps(agentId, { mcpNamespaces });
      await AiAgentAPI.setSubAgents(agentId, { subagents });
    }
    await fetchAgents();
  }

  async function copyAgent(agentId: number, agentCode: string) {
    await AiAgentAPI.copy(agentId, { agentCode });
    await fetchAgents();
  }

  async function switchAgentStatus(agentId: number, status: EnabledStatus) {
    await AiAgentAPI.setStatus(agentId, { status });
    await fetchAgents();
  }

  async function deleteAgent(agentId: number) {
    await AiAgentAPI.delete(agentId);
    await fetchAgents();
  }

  async function switchAgentExposed(agentId: number, isExposed: boolean) {
    await AiAgentAPI.update(agentId, { isExposed });
    await fetchAgentDetail(agentId);
  }

  // ==================== 版本管理 ====================
  const versions = ref<AgentVersionResult[]>([]);
  const versionsTotal = ref(0);
  const versionsLoading = ref(false);
  const versionsQuery = reactive({ pageNum: 1, pageSize: 10 });
  const versionDiff = ref<Array<Record<string, unknown>>>([]);
  const diffLoading = ref(false);
  const versionDetail = ref<AgentVersionDetail | null>(null);

  async function fetchVersions(agentId: number) {
    versionsLoading.value = true;
    try {
      const page = await AiAgentAPI.versions(agentId, { ...versionsQuery });
      versions.value = page.list ?? [];
      versionsTotal.value = page.total ?? 0;
    } finally {
      versionsLoading.value = false;
    }
  }

  async function fetchVersionDetail(agentId: number, versionNo: number) {
    versionDetail.value = await AiAgentAPI.versionDetail(agentId, versionNo);
  }

  async function compareVersions(
    agentId: number,
    base: number,
    target: number
  ) {
    diffLoading.value = true;
    try {
      versionDiff.value = (await AiAgentAPI.versionDiff(
        agentId,
        base,
        target
      )) as Array<Record<string, unknown>>;
    } finally {
      diffLoading.value = false;
    }
  }

  /**
   * 发布 Agent。force=true 为判分漂移豁免：门禁因判分模型漂移暂停时，
   * 管理员确认风险后强制发布，豁免原因随 changeNote 一并记录。
   */
  async function publishAgent(
    agentId: number,
    changeNote: string,
    force = false
  ) {
    const result: VersionResult = await AiAgentAPI.publish(agentId, {
      changeNote,
      force,
    });
    await fetchVersions(agentId);
    return result;
  }

  async function rollbackVersion(agentId: number, versionNo: number) {
    await AiAgentAPI.rollback(agentId, versionNo);
    await fetchVersions(agentId);
  }

  // ==================== 评测（发布门禁） ====================
  /** 评测任务轮询间隔与上限（单次回归评测最长约 10 分钟） */
  const EVAL_TASK_POLL_INTERVAL_MS = 1500;
  const EVAL_TASK_MAX_POLLS = 400;

  const evalDatasets = ref<EvalDatasetResult[]>([]);
  const evalSamples = ref<EvalSampleResult[]>([]);
  const evalSamplesLoading = ref(false);
  const evalRuns = ref<EvalRunResult[]>([]);
  const evalRunsTotal = ref(0);
  const evalLoading = ref(false);
  const evalRunsQuery = reactive({ pageNum: 1, pageSize: 10 });

  async function fetchEvalDatasets(agentId: number) {
    evalDatasets.value = (await AiAgentAPI.listEvalDatasets(agentId)) ?? [];
  }

  async function createEvalDataset(
    agentId: number,
    form: EvalDatasetCreateForm
  ) {
    await AiAgentAPI.createEvalDataset(agentId, form);
    await fetchEvalDatasets(agentId);
  }

  async function updateEvalDataset(
    agentId: number,
    datasetId: number,
    form: EvalDatasetUpdateForm
  ) {
    await AiAgentAPI.updateEvalDataset(agentId, datasetId, form);
    await fetchEvalDatasets(agentId);
  }

  async function deleteEvalDataset(agentId: number, datasetId: number) {
    await AiAgentAPI.deleteEvalDataset(agentId, datasetId);
    await fetchEvalDatasets(agentId);
  }

  async function fetchEvalSamples(agentId: number, datasetId: number) {
    evalSamplesLoading.value = true;
    try {
      evalSamples.value =
        (await AiAgentAPI.listEvalSamples(agentId, datasetId)) ?? [];
    } finally {
      evalSamplesLoading.value = false;
    }
  }

  async function createEvalSample(
    agentId: number,
    datasetId: number,
    form: EvalSampleCreateForm
  ) {
    await AiAgentAPI.createEvalSample(agentId, datasetId, form);
    await fetchEvalSamples(agentId, datasetId);
  }

  async function updateEvalSample(
    agentId: number,
    datasetId: number,
    sampleId: number,
    form: EvalSampleUpdateForm
  ) {
    await AiAgentAPI.updateEvalSample(agentId, sampleId, form);
    await fetchEvalSamples(agentId, datasetId);
  }

  async function deleteEvalSample(
    agentId: number,
    datasetId: number,
    sampleId: number
  ) {
    await AiAgentAPI.deleteEvalSample(agentId, sampleId);
    await fetchEvalSamples(agentId, datasetId);
  }

  async function fetchEvalRuns(agentId: number) {
    evalLoading.value = true;
    try {
      const page = await AiAgentAPI.listEvalRuns(agentId, {
        ...evalRunsQuery,
      });
      evalRuns.value = page.list ?? [];
      evalRunsTotal.value = page.total ?? 0;
    } finally {
      evalLoading.value = false;
    }
  }

  /** 评测异步任务（进度与结果由 runEval 轮询写入） */
  const evalTask = ref<EvalRunTaskResult | null>(null);
  const evalProgress = computed(() => {
    const progress = evalTask.value?.progress;
    return progress?.total
      ? Math.round((progress.done / progress.total) * 100)
      : 0;
  });

  /**
   * 手动触发回归评测：提交后轮询任务进度至 succeeded/failed，返回门禁判定结果
   *（任务失败时返回 null，失败原因读 evalTask.error）。
   */
  async function runEval(agentId: number): Promise<EvalRunGateResult | null> {
    evalTask.value = null;
    const { taskId } = await AiAgentAPI.runEvalAsync(agentId);
    for (let poll = 0; poll < EVAL_TASK_MAX_POLLS; poll++) {
      const task = await AiAgentAPI.getEvalTask(agentId, taskId);
      evalTask.value = task;
      if (task.status === "succeeded" || task.status === "failed") {
        await fetchEvalRuns(agentId);
        return task.result ?? null;
      }
      await new Promise((resolve) =>
        setTimeout(resolve, EVAL_TASK_POLL_INTERVAL_MS)
      );
    }
    throw new Error("评测执行超时，请稍后在评测执行记录中查看结果");
  }

  // ==================== 测试（即时预览，不入库） ====================
  async function testAgent(agentId: number, message: string) {
    return AiAgentAPI.test(agentId, { message });
  }

  // ==================== A2A 端点 ====================
  const a2aEndpoints = ref<EndpointResult[]>([]);
  const a2aTotal = ref(0);
  const a2aLoading = ref(false);
  const a2aQuery = reactive({ pageNum: 1, pageSize: 10 });

  async function fetchA2aEndpoints() {
    a2aLoading.value = true;
    try {
      const page = await AiAgentAPI.listEndpoints({ ...a2aQuery });
      a2aEndpoints.value = page.list ?? [];
      a2aTotal.value = page.total ?? 0;
    } finally {
      a2aLoading.value = false;
    }
  }

  async function manageA2aEndpoints(
    action: "create" | "update" | "delete" | "refresh",
    payload: { id?: number; form?: EndpointCreateForm | EndpointUpdateForm }
  ) {
    if (action === "create") {
      await AiAgentAPI.createEndpoint(payload.form as EndpointCreateForm);
    } else if (action === "update") {
      // update 分支由 A2aPanel 构造 EndpointUpdateForm（无 baseUrl）
      await AiAgentAPI.updateEndpoint(
        payload.id!,
        payload.form as EndpointUpdateForm
      );
    } else if (action === "delete") {
      await AiAgentAPI.deleteEndpoint(payload.id!);
    } else {
      await AiAgentAPI.refreshEndpointCard(payload.id!);
    }
    await fetchA2aEndpoints();
  }

  return {
    agents,
    total,
    loading,
    query,
    agentTypeFilter,
    agentForm,
    detail,
    detailLoading,
    versions,
    versionsTotal,
    versionsLoading,
    versionsQuery,
    versionDiff,
    diffLoading,
    versionDetail,
    evalDatasets,
    evalSamples,
    evalSamplesLoading,
    evalTask,
    evalProgress,
    evalRuns,
    evalRunsTotal,
    evalLoading,
    evalRunsQuery,
    a2aEndpoints,
    a2aTotal,
    a2aLoading,
    a2aQuery,
    fetchAgents,
    fetchAgentDetail,
    saveAgent,
    copyAgent,
    switchAgentStatus,
    deleteAgent,
    switchAgentExposed,
    fetchVersions,
    fetchVersionDetail,
    compareVersions,
    publishAgent,
    rollbackVersion,
    fetchEvalDatasets,
    createEvalDataset,
    updateEvalDataset,
    deleteEvalDataset,
    fetchEvalSamples,
    createEvalSample,
    updateEvalSample,
    deleteEvalSample,
    fetchEvalRuns,
    runEval,
    testAgent,
    fetchA2aEndpoints,
    manageA2aEndpoints,
  };
});
