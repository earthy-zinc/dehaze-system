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
  EvalDatasetResult,
  EvalRunResult,
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

  async function publishAgent(agentId: number, changeNote: string) {
    const result: VersionResult = await AiAgentAPI.publish(agentId, {
      changeNote,
    });
    await fetchVersions(agentId);
    return result;
  }

  async function rollbackVersion(agentId: number, versionNo: number) {
    await AiAgentAPI.rollback(agentId, versionNo);
    await fetchVersions(agentId);
  }

  // ==================== 评测（发布门禁） ====================
  const evalDatasets = ref<EvalDatasetResult[]>([]);
  const evalRuns = ref<EvalRunResult[]>([]);
  const evalRunsTotal = ref(0);
  const evalLoading = ref(false);
  const evalRunsQuery = reactive({ pageNum: 1, pageSize: 10 });

  async function fetchEvalDatasets(agentId: number) {
    evalDatasets.value = (await AiAgentAPI.listEvalDatasets(agentId)) ?? [];
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

  /** 手动触发回归评测，返回门禁判定（runId/passed/scoreSummary/failedSamples） */
  async function runEval(agentId: number) {
    const result = await AiAgentAPI.runEval(agentId);
    await fetchEvalRuns(agentId);
    return result;
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
    fetchEvalRuns,
    runEval,
    testAgent,
    fetchA2aEndpoints,
    manageA2aEndpoints,
  };
});
