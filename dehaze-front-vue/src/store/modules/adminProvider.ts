import {
  AiProviderAPI,
  ConnectionTestResult,
  ProviderCreateForm,
  ProviderHealth,
  ProviderHealthStatVO,
  ProviderKeyVO,
  ProviderPageQuery,
  ProviderUpdateForm,
  ProviderVO,
} from "dehaze-sdk-js";

// 管理端供应商 Store：列表/配置抽屉/API Key/连通性测试/熔断处置
export const useAdminProviderStore = defineStore("adminProvider", () => {
  const providers = ref<ProviderVO[]>([]);
  const total = ref(0);
  const loading = ref(false);
  const query = reactive<ProviderPageQuery>({ pageNum: 1, pageSize: 10 });

  /** 配置抽屉：编辑中的供应商（null 表示新增）+ 连通性测试结果 */
  const drawer = reactive<{
    visible: boolean;
    provider: ProviderVO | null;
    testResult: ConnectionTestResult | null;
  }>({
    visible: false,
    provider: null,
    testResult: null,
  });

  const keys = ref<ProviderKeyVO[]>([]);
  const keysLoading = ref(false);

  const healthBoard = ref<ProviderHealthStatVO[]>([]);
  const healthLoading = ref(false);

  async function fetchProviders() {
    loading.value = true;
    try {
      const page = await AiProviderAPI.listProviders(query);
      providers.value = page.list ?? [];
      total.value = page.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  async function fetchHealthBoard() {
    healthLoading.value = true;
    try {
      const stats = await AiProviderAPI.getUsageStats();
      healthBoard.value = stats.providerHealth ?? [];
    } finally {
      healthLoading.value = false;
    }
  }

  /** 保存供应商，保存成功后异步触发连通性测试（仅提示不阻断） */
  async function saveProvider(form: ProviderCreateForm, id?: number) {
    const saved = id
      ? await AiProviderAPI.updateProvider(id, form as ProviderUpdateForm)
      : await AiProviderAPI.createProvider(form);
    testConnectionSilently(saved.id);
    await fetchProviders();
    await fetchHealthBoard();
  }

  /** 连通性测试不参与保存事务，结果仅作提示，失败静默（抽屉内可见手动测试结果） */
  async function testConnectionSilently(providerId: number) {
    try {
      const result = await AiProviderAPI.testConnection(providerId);
      ElMessage.success("供应商已保存，连通性测试通过");
      drawer.testResult = result;
    } catch {
      ElMessage.warning("供应商已保存，但连通性测试未通过，请稍后手动重试");
    }
  }

  async function deleteProvider(id: number) {
    await AiProviderAPI.deleteProvider(id);
    await fetchProviders();
    await fetchHealthBoard();
  }

  async function openDrawer(provider: ProviderVO | null) {
    drawer.provider = provider;
    drawer.testResult = null;
    drawer.visible = true;
    keys.value = [];
    if (provider) {
      await fetchKeys(provider.id);
    }
  }

  function closeDrawer() {
    drawer.visible = false;
    drawer.provider = null;
    drawer.testResult = null;
  }

  async function fetchKeys(providerId: number) {
    keysLoading.value = true;
    try {
      keys.value = (await AiProviderAPI.listKeys(providerId)) ?? [];
    } finally {
      keysLoading.value = false;
    }
  }

  async function createKey(providerId: number, form: Record<string, unknown>) {
    await AiProviderAPI.createKey(providerId, form as never);
    await fetchKeys(providerId);
  }

  /** 编辑 Key 优先级/权重/限额等非敏感字段 */
  async function updateKey(
    providerId: number,
    keyId: number,
    form: Record<string, unknown>
  ) {
    await AiProviderAPI.updateKey(providerId, keyId, form as never);
    await fetchKeys(providerId);
  }

  /** 删除 Key 由后端校验至少保留一个启用 Key，此处透传业务错误提示 */
  async function deleteKey(providerId: number, keyId: number) {
    await AiProviderAPI.deleteKey(providerId, keyId);
    await fetchKeys(providerId);
  }

  async function testConnection(providerId: number) {
    drawer.testResult = await AiProviderAPI.testConnection(providerId);
    return drawer.testResult;
  }

  /** 解除熔断后刷新列表与健康看板（熔断状态由后端实时判定） */
  async function closeCircuit(providerId: number) {
    await AiProviderAPI.closeCircuit(providerId);
    await Promise.all([fetchProviders(), fetchHealthBoard()]);
  }

  /** 健康状态标签配置：open 熔断醒目 */
  const healthTagMap: Record<
    ProviderHealth,
    { label: string; type: "success" | "warning" | "danger" }
  > = {
    healthy: { label: "健康", type: "success" },
    suspicious: { label: "可疑", type: "warning" },
    open: { label: "熔断", type: "danger" },
  };

  return {
    providers,
    total,
    loading,
    query,
    drawer,
    keys,
    keysLoading,
    healthBoard,
    healthLoading,
    healthTagMap,
    fetchProviders,
    fetchHealthBoard,
    saveProvider,
    deleteProvider,
    openDrawer,
    closeDrawer,
    fetchKeys,
    createKey,
    updateKey,
    deleteKey,
    testConnection,
    closeCircuit,
  };
});
