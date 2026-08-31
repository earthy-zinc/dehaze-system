import {
  AiModelAPI,
  AiModelForm,
  AiModelQuery,
  AiModelType,
  AiModelVO,
  AiProviderAPI,
  ModelPriceDetailForm,
  ModelPriceQuery,
  ModelPriceVO,
  ProviderVO,
  UsageStatsVO,
} from "dehaze-sdk-js";

// 管理端模型 Store：模型列表/类型筛选/类型化表单/价格版本/运营视图
export const useAdminModelStore = defineStore("adminModel", () => {
  const models = ref<AiModelVO[]>([]);
  const total = ref(0);
  const loading = ref(false);
  const query = reactive<AiModelQuery>({ pageNum: 1, pageSize: 10 });
  const modelTypeFilter = ref<AiModelType | "all">("all");

  /** 供应商字典：表单下拉 + 列表供应商名映射 */
  const providers = ref<ProviderVO[]>([]);
  const providerNameMap = computed(() => {
    const map = new Map<number, string>();
    providers.value.forEach((p) => map.set(p.id, p.displayName));
    return map;
  });

  const formDialog = reactive<{ visible: boolean; model: AiModelVO | null }>({
    visible: false,
    model: null,
  });
  /** chat 模型的降级目标候选（启用 chat 模型，排除自身） */
  const fallbackOptions = ref<AiModelVO[]>([]);

  const priceDialog = reactive<{ visible: boolean; model: AiModelVO | null }>({
    visible: false,
    model: null,
  });
  /** 价格档位编辑行：保存即生成新价格版本 */
  const priceRows = reactive<ModelPriceDetailForm[]>([]);
  const priceSubmitting = ref(false);
  const priceHistory = ref<ModelPriceVO[]>([]);
  const priceTotal = ref(0);
  const priceLoading = ref(false);
  const priceQuery = reactive<ModelPriceQuery>({ page: 1, size: 5 });

  const statsTab = ref<"health" | "usage" | "degrade">("health");
  const operation = ref<UsageStatsVO | null>(null);
  const operationLoading = ref(false);

  async function fetchProviders() {
    const page = await AiProviderAPI.listProviders({
      pageNum: 1,
      pageSize: 100,
    });
    providers.value = page.list ?? [];
  }

  async function fetchModels() {
    loading.value = true;
    try {
      const params: AiModelQuery = { ...query };
      if (modelTypeFilter.value !== "all") {
        params.modelType = modelTypeFilter.value;
      } else {
        delete params.modelType;
      }
      const page = await AiModelAPI.listModels(params);
      models.value = page.list ?? [];
      total.value = page.total ?? 0;
    } finally {
      loading.value = false;
    }
  }

  async function openFormDialog(model: AiModelVO | null) {
    formDialog.model = model;
    formDialog.visible = true;
    if (providers.value.length === 0) {
      await fetchProviders();
    }
    // 降级目标候选仅在 chat 模型表单使用，排除自身避免自引用
    if (!model || model.modelType === "chat") {
      const enabled = await AiModelAPI.listEnabledModels("chat");
      fallbackOptions.value = (enabled ?? []).filter(
        (m) => !model || m.modelId !== model.modelId
      );
    }
  }

  async function saveModel(form: AiModelForm) {
    if (formDialog.model) {
      // modelType/dimension 创建后不可改，后端更新接口收到即拒绝，编辑提交时剔除
      const { modelType: _m, dimension: _d, ...updateForm } = form;
      await AiModelAPI.updateModel(formDialog.model.modelId, updateForm);
    } else {
      await AiModelAPI.createModel(form);
    }
    await fetchModels();
  }

  async function toggleStatus(model: AiModelVO, status: 0 | 1) {
    await AiModelAPI.updateModel(model.modelId, { status });
    await fetchModels();
  }

  async function deleteModel(model: AiModelVO) {
    await AiModelAPI.deleteModel(model.modelId);
    await fetchModels();
  }

  /** 打开价格弹窗：初始化编辑行为最新版本档位 */
  async function openPriceDialog(model: AiModelVO) {
    priceDialog.model = model;
    priceDialog.visible = true;
    priceQuery.page = 1;
    priceRows.splice(0, priceRows.length);
    const history = await fetchPriceHistory(model.modelId);
    // 保存即生成新版本，编辑行从最新版本档位复制
    const latest = history[0];
    if (latest) {
      priceRows.push(
        ...latest.details.map((d) => ({
          tokenType: d.tokenType,
          timeSlot: d.timeSlot,
          minTokens: d.minTokens,
          maxTokens: d.maxTokens ?? null,
          unitPrice: Number(d.unitPrice),
        }))
      );
    }
  }

  async function fetchPriceHistory(modelId: string) {
    priceLoading.value = true;
    try {
      const page = await AiModelAPI.listPrices(modelId, priceQuery);
      priceHistory.value = page.list ?? [];
      priceTotal.value = page.total ?? 0;
      return priceHistory.value;
    } finally {
      priceLoading.value = false;
    }
  }

  async function savePrice() {
    const model = priceDialog.model;
    if (!model) return;
    priceSubmitting.value = true;
    try {
      await AiModelAPI.createPrice(model.modelId, {
        modelId: model.modelId,
        providerId: model.providerId,
        details: priceRows.map((row) => ({ ...row })),
      });
      ElMessage.success("价格新版本已保存");
      priceDialog.visible = false;
      await fetchPriceHistory(model.modelId);
    } finally {
      priceSubmitting.value = false;
    }
  }

  async function fetchOperation() {
    operationLoading.value = true;
    try {
      operation.value = await AiProviderAPI.getUsageStats();
    } finally {
      operationLoading.value = false;
    }
  }

  return {
    models,
    total,
    loading,
    query,
    modelTypeFilter,
    providers,
    providerNameMap,
    formDialog,
    fallbackOptions,
    priceDialog,
    priceRows,
    priceSubmitting,
    priceHistory,
    priceTotal,
    priceLoading,
    priceQuery,
    statsTab,
    operation,
    operationLoading,
    fetchModels,
    fetchProviders,
    openFormDialog,
    saveModel,
    toggleStatus,
    deleteModel,
    openPriceDialog,
    fetchPriceHistory,
    savePrice,
    fetchOperation,
  };
});
