import {
  AiKnowledgeBaseAPI,
  IndexStatsVO,
  KnowledgeBaseVO,
  LowQualityChunkQuery,
  LowQualityChunkVO,
  RecallTestResultVO,
  TestSetCreateForm,
  TestSetVO,
} from "dehaze-sdk-js";

// 管理端知识库 Store：私有库监控、索引状态、召回测试、低质量片段
export const useAdminKbStore = defineStore("adminKb", () => {
  const adminTab = ref<"public" | "private">("public");
  const privateKbs = ref<KnowledgeBaseVO[]>([]);
  const privateLoading = ref(false);
  const indexStats = ref<IndexStatsVO | null>(null);
  const qualityTab = ref<"retrieve" | "recall" | "low-quality">("retrieve");
  const recallSets = ref<TestSetVO[]>([]);
  const recallSetsTotal = ref(0);
  const recallCompare = ref<RecallTestResultVO | null>(null);
  const lowQualityChunks = ref<LowQualityChunkVO[]>([]);
  const lowQualityTotal = ref(0);
  const lowQualityQuery = reactive<LowQualityChunkQuery>({
    pageNum: 1,
    pageSize: 10,
  });

  /** 切换列表区 Tab（公共知识库/私有库监控），私有库按需加载 */
  async function switchAdminTab(tab: "public" | "private") {
    adminTab.value = tab;
    if (tab === "private") {
      await fetchPrivateKbs();
    }
  }

  /** 拉取私有库监控列表（view=admin 返回全部库，按可见性过滤私有库） */
  async function fetchPrivateKbs() {
    privateLoading.value = true;
    try {
      const result = await AiKnowledgeBaseAPI.getList({
        view: "admin",
        pageNum: 1,
        pageSize: 100,
      });
      privateKbs.value =
        result.list?.filter((kb) => kb.visibility === "private") ?? [];
    } finally {
      privateLoading.value = false;
    }
  }

  async function fetchIndexStats(kbId: number) {
    indexStats.value = await AiKnowledgeBaseAPI.getIndexStats(kbId);
  }

  async function fetchRecallSets(kbId: number) {
    const result = await AiKnowledgeBaseAPI.getTestSets(kbId, {
      pageNum: 1,
      pageSize: 50,
    });
    recallSets.value = result.list ?? [];
    recallSetsTotal.value = result.total ?? 0;
  }

  async function createTestSet(kbId: number, form: TestSetCreateForm) {
    await AiKnowledgeBaseAPI.createTestSet(kbId, form);
    await fetchRecallSets(kbId);
  }

  async function runRecallSet(kbId: number, testSetId: number) {
    recallCompare.value = await AiKnowledgeBaseAPI.runTestSet(kbId, testSetId);
    return recallCompare.value;
  }

  async function fetchLowQuality(kbId: number) {
    const result = await AiKnowledgeBaseAPI.getLowQualityChunks(
      kbId,
      lowQualityQuery
    );
    lowQualityChunks.value = result.list ?? [];
    lowQualityTotal.value = result.total ?? 0;
  }

  /**
   * 处置低质量片段（清理=删除来源文档，重新分块=重跑文档流水线）。
   * 专有片段级处置接口后端规划中，暂以文档级操作兜底。
   */
  async function disposeLowQuality(
    kbId: number,
    chunk: LowQualityChunkVO,
    mode: "clean" | "rechunk"
  ) {
    if (mode === "clean") {
      await AiKnowledgeBaseAPI.deleteDocument(chunk.documentId);
    } else {
      await AiKnowledgeBaseAPI.reprocessDocument(chunk.documentId);
    }
    await fetchLowQuality(kbId);
  }

  return {
    adminTab,
    privateKbs,
    privateLoading,
    indexStats,
    qualityTab,
    recallSets,
    recallSetsTotal,
    recallCompare,
    lowQualityChunks,
    lowQualityTotal,
    lowQualityQuery,
    switchAdminTab,
    fetchPrivateKbs,
    fetchIndexStats,
    fetchRecallSets,
    createTestSet,
    runRecallSet,
    fetchLowQuality,
    disposeLowQuality,
  };
});
