import {
  AiKnowledgeBaseAPI,
  IndexStatsVO,
  KnowledgeBaseUpdateForm,
  KnowledgeBaseVO,
  LowQualityChunkQuery,
  LowQualityChunkVO,
  RecallTestResultVO,
  TestSetCreateForm,
  TestSetVO,
} from "dehaze-sdk-js";
import { useKbDataStore } from "@/store/modules/kbData";

// 管理端知识库 Store：私有库监控、索引状态、embedding 迁移、召回测试、低质量片段
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
  const migrateDialog = reactive({ visible: false });

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

  /**
   * 提交 embedding 迁移。
   * 专有迁移接口后端规划中，先以更新知识库配置的方式替换 embedding 模型字段，
   * 迁移本身需要后台批量重新向量化并重建索引，前端仅提示。
   */
  async function submitEmbeddingMigrate(kbId: number, targetModelId: string) {
    const kbDataStore = useKbDataStore();
    const kb = kbDataStore.kbDetail;
    if (!kb) {
      ElMessage.error("知识库详情未加载，无法提交迁移");
      return;
    }
    const form: KnowledgeBaseUpdateForm & { embeddingModel?: string } = {
      name: kb.name,
      description: kb.description,
      searchStrategy: kb.searchStrategy,
      hybridWeight: kb.hybridWeight,
      topK: kb.topK,
      scoreThreshold: kb.scoreThreshold,
      enableRerank: kb.enableRerank === 1,
      rerankModel: kb.rerankModel,
      embeddingModel: targetModelId,
    };
    await AiKnowledgeBaseAPI.update(kbId, form);
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
    migrateDialog,
    switchAdminTab,
    fetchPrivateKbs,
    fetchIndexStats,
    submitEmbeddingMigrate,
    fetchRecallSets,
    createTestSet,
    runRecallSet,
    fetchLowQuality,
    disposeLowQuality,
  };
});
