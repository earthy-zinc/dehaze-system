import {
  AiMCPAPI,
  McpCallQuery,
  McpCallVO,
  McpCredentialForm,
  McpHealthVO,
  McpMarketPresetVO,
  McpNamespaceVO,
  McpServerForm,
  McpServerQuery,
  McpServerVO,
  McpToolTestResult,
  McpToolVO,
} from "dehaze-sdk-js";

/** Server 抽屉 Tab：配置（注册/编辑）/工具与命名空间/凭据 */
export type McpServerDrawerTab = "config" | "tools" | "credentials";

/** 传输协议展示名 */
export const MCP_PROTOCOL_LABELS: Record<string, string> = {
  stdio: "stdio（本地进程）",
  "streamable-http": "streamable-http",
  sse: "sse（传统 SSE）",
};

/** 鉴权方式展示名 */
export const MCP_AUTH_LABELS: Record<string, string> = {
  none: "无鉴权",
  api_key: "API Key",
  oauth2: "OAuth2",
};

// 管理端 MCP Store：市场目录、Server 注册表、工具与命名空间、凭据、健康探测、调用审计
export const useAdminMcpStore = defineStore("adminMcp", () => {
  const activeTab = ref<"market" | "servers" | "calls">("servers");

  // ==================== 市场 ====================
  const marketPresets = ref<McpMarketPresetVO[]>([]);
  const marketLoading = ref(false);
  const installingPresetId = ref("");

  // ==================== Server 注册表 ====================
  const servers = ref<McpServerVO[]>([]);
  const serverTotal = ref(0);
  const serverLoading = ref(false);
  const serverQuery = reactive<McpServerQuery>({ pageNum: 1, pageSize: 10 });
  const serverForm = reactive<{
    visible: boolean;
    mode: "create" | "edit";
    server: McpServerVO | null;
    tab: McpServerDrawerTab;
  }>({ visible: false, mode: "create", server: null, tab: "config" });

  // ==================== 工具与命名空间 ====================
  const tools = ref<McpToolVO[]>([]);
  const toolsLoading = ref(false);
  const namespaces = ref<McpNamespaceVO[]>([]);
  const namespacesLoading = ref(false);

  // ==================== 凭据 ====================
  // 后端不返回凭据明细也不回显明文，"已配置" 仅在本次会话内成功保存后置位
  const credentialsConfigured = ref<Record<number, boolean>>({});

  // ==================== 健康探测 ====================
  const health = ref<Record<number, McpHealthVO>>({});
  const healthLoading = ref(false);
  const healthServerId = ref<number | null>(null);

  // ==================== 调用审计 ====================
  const mcpCalls = ref<McpCallVO[]>([]);
  const mcpCallTotal = ref(0);
  const mcpCallLoading = ref(false);
  const mcpCallQuery = reactive<McpCallQuery>({ pageNum: 1, pageSize: 10 });

  /** 切换页面主 Tab，各区数据按需加载 */
  async function switchTab(tab: "market" | "servers" | "calls") {
    activeTab.value = tab;
    if (tab === "market" && marketPresets.value.length === 0) {
      await fetchMarketPresets();
    }
    if (tab === "servers" && servers.value.length === 0) {
      await fetchServers();
    }
    if (tab === "calls" && mcpCalls.value.length === 0) {
      await fetchMcpCalls();
    }
  }

  async function fetchMarketPresets() {
    marketLoading.value = true;
    try {
      marketPresets.value = (await AiMCPAPI.getMarket()) ?? [];
    } finally {
      marketLoading.value = false;
    }
  }

  /** 市场一键接入：注册后自动拉取工具清单，管理员预览工具后再启用 */
  async function installPreset(preset: McpMarketPresetVO) {
    installingPresetId.value = preset.presetId;
    try {
      const server = await AiMCPAPI.installPreset(preset.presetId);
      ElMessage.success(`「${preset.name}」已接入，请预览工具清单后再启用`);
      await fetchMarketPresets();
      await fetchServers();
      await openServerDrawer(server, "tools");
    } finally {
      installingPresetId.value = "";
    }
  }

  async function fetchServers() {
    serverLoading.value = true;
    try {
      const page = await AiMCPAPI.listServers({ ...serverQuery });
      servers.value = page.list ?? [];
      serverTotal.value = page.total ?? 0;
    } finally {
      serverLoading.value = false;
    }
  }

  function openCreateDrawer() {
    serverForm.mode = "create";
    serverForm.server = null;
    serverForm.tab = "config";
    serverForm.visible = true;
  }

  async function openServerDrawer(
    server: McpServerVO,
    tab: McpServerDrawerTab = "config"
  ) {
    serverForm.mode = "edit";
    serverForm.server = server;
    serverForm.tab = tab;
    serverForm.visible = true;
    if (tab === "tools") {
      await fetchServerResources(server.id);
    }
  }

  /** 切换抽屉 Tab，工具与命名空间按需加载 */
  async function switchDrawerTab(tab: McpServerDrawerTab) {
    serverForm.tab = tab;
    if (tab === "tools" && serverForm.server) {
      await fetchServerResources(serverForm.server.id);
    }
  }

  async function fetchServerResources(serverId: number) {
    await Promise.all([fetchTools(serverId), fetchNamespaces(serverId)]);
  }

  /** 注册/更新 Server；注册成功后直接进入工具 Tab（接入即发现） */
  async function registerServer(form: McpServerForm) {
    if (serverForm.server) {
      serverForm.server = await AiMCPAPI.updateServer(
        serverForm.server.id,
        form
      );
      ElMessage.success("Server 配置已更新");
    } else {
      serverForm.server = await AiMCPAPI.createServer(form);
      serverForm.mode = "edit";
      serverForm.tab = "tools";
      ElMessage.success("Server 已注册，请预览工具清单后再启用");
    }
    await fetchServers();
    if (serverForm.tab === "tools" && serverForm.server) {
      await fetchServerResources(serverForm.server.id);
    }
  }

  async function switchServerStatus(server: McpServerVO, status: 0 | 1) {
    const updated = await AiMCPAPI.switchServerStatus(server.id, status);
    await fetchServers();
    // 列表整表刷新后抽屉持有的是旧对象，同步回写以刷新抽屉内启用状态
    if (serverForm.server?.id === server.id) {
      serverForm.server = updated;
    }
  }

  async function deleteServer(server: McpServerVO) {
    await AiMCPAPI.deleteServer(server.id);
    await fetchServers();
    if (serverForm.server?.id === server.id) {
      serverForm.visible = false;
    }
  }

  async function configureCredentials(
    serverId: number,
    form: McpCredentialForm
  ) {
    await AiMCPAPI.updateCredentials(serverId, form);
    ElMessage.success("凭据已加密保存");
    // 凭据不回显，"已配置"状态以后端 credentialConfigured 为准，保存后刷新
    await fetchServers();
    if (serverForm.server?.id === serverId) {
      serverForm.server =
        servers.value.find((s) => s.id === serverId) ?? serverForm.server;
    }
  }

  async function fetchTools(serverId: number) {
    toolsLoading.value = true;
    try {
      tools.value = (await AiMCPAPI.getTools(serverId)) ?? [];
    } finally {
      toolsLoading.value = false;
    }
  }

  async function fetchNamespaces(serverId: number) {
    namespacesLoading.value = true;
    try {
      namespaces.value = (await AiMCPAPI.getNamespaces(serverId)) ?? [];
    } finally {
      namespacesLoading.value = false;
    }
  }

  async function configureNamespaces(serverId: number, list: McpNamespaceVO[]) {
    namespaces.value = (await AiMCPAPI.updateNamespaces(serverId, list)) ?? [];
    ElMessage.success("命名空间已保存（覆盖式更新）");
  }

  /** 试调用 MCP 工具（管理员验证连通性与参数，不走 LLM） */
  async function testTool(
    serverId: number,
    toolName: string,
    arguments_: Record<string, unknown>
  ): Promise<McpToolTestResult> {
    return AiMCPAPI.testTool(serverId, { toolName, arguments: arguments_ });
  }

  /** 健康探测：结果同步回列表行，异常 Server 在列表显著标注 */
  async function probeHealth(server: McpServerVO) {
    healthServerId.value = server.id;
    healthLoading.value = true;
    try {
      const result = await AiMCPAPI.probeHealth(server.id);
      health.value[server.id] = result;
      server.health = result.status;
    } finally {
      healthLoading.value = false;
    }
  }

  async function fetchMcpCalls() {
    mcpCallLoading.value = true;
    try {
      const page = await AiMCPAPI.listCalls({ ...mcpCallQuery });
      mcpCalls.value = page.list ?? [];
      mcpCallTotal.value = page.total ?? 0;
    } finally {
      mcpCallLoading.value = false;
    }
  }

  return {
    activeTab,
    marketPresets,
    marketLoading,
    installingPresetId,
    servers,
    serverTotal,
    serverLoading,
    serverQuery,
    serverForm,
    tools,
    toolsLoading,
    namespaces,
    namespacesLoading,
    health,
    healthLoading,
    healthServerId,
    mcpCalls,
    mcpCallTotal,
    mcpCallLoading,
    mcpCallQuery,
    switchTab,
    fetchMarketPresets,
    installPreset,
    fetchServers,
    openCreateDrawer,
    openServerDrawer,
    switchDrawerTab,
    registerServer,
    switchServerStatus,
    deleteServer,
    configureCredentials,
    fetchTools,
    fetchNamespaces,
    configureNamespaces,
    testTool,
    probeHealth,
    fetchMcpCalls,
  };
});
