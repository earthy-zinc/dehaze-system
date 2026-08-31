import {
  AiAgentAPI,
  AiEvalAPI,
  AiEvalAgentOverviewItem,
  AiEvalJudgeStatus,
  AiEvalReviewQueueResult,
  AiEvalReviewStatus,
  AiEvalRunCompareResult,
  AiEvalTrendItem,
  EvalDatasetResult,
  EvalRunResult,
} from "dehaze-sdk-js";

// 管理端评测中心 Store：总览 / 执行记录 / 详情与版本对比 / 趋势 / 判分状态 / 人工复核
export const useAdminEvalStore = defineStore("adminEval", () => {
  // ==================== 筛选条件 ====================
  const evalFilter = reactive({
    agentId: undefined as number | undefined,
    datasetId: undefined as number | undefined,
    startTime: undefined as string | undefined,
    endTime: undefined as string | undefined,
    pageNum: 1,
    pageSize: 10,
  });

  // ==================== 评测总览 ====================
  const evalOverview = ref<AiEvalAgentOverviewItem[]>([]);
  const overviewLoading = ref(false);

  async function fetchOverview() {
    overviewLoading.value = true;
    try {
      evalOverview.value = await AiEvalAPI.getOverview();
    } finally {
      overviewLoading.value = false;
    }
  }

  // 评测集下拉：执行记录按 Agent 维度拉取，切换 Agent 时重新加载
  const datasets = ref<EvalDatasetResult[]>([]);

  async function fetchDatasets(agentId: number) {
    datasets.value = await AiAgentAPI.listEvalDatasets(agentId);
  }

  // ==================== 评测执行记录 ====================
  const evalRuns = ref<EvalRunResult[]>([]);
  const evalRunsTotal = ref(0);
  const runsLoading = ref(false);

  async function fetchRuns() {
    const agentId = evalFilter.agentId;
    if (agentId == null) {
      evalRuns.value = [];
      evalRunsTotal.value = 0;
      return;
    }
    runsLoading.value = true;
    try {
      const page = await AiAgentAPI.listEvalRuns(agentId, {
        pageNum: evalFilter.pageNum,
        pageSize: evalFilter.pageSize,
        datasetId: evalFilter.datasetId,
      });
      evalRuns.value = page.list ?? [];
      evalRunsTotal.value = page.total ?? 0;
    } finally {
      runsLoading.value = false;
    }
  }

  /** 切换 Agent：重置分页与评测集，联动执行记录与趋势 */
  async function selectAgent(agentId: number | undefined) {
    evalFilter.agentId = agentId;
    evalFilter.datasetId = undefined;
    evalFilter.pageNum = 1;
    datasets.value = [];
    if (agentId != null) {
      await fetchDatasets(agentId);
    }
    await Promise.all([fetchRuns(), fetchTrends()]);
  }

  // ==================== 评测详情与版本对比 ====================
  const detailVisible = ref(false);
  const evalDetail = ref<EvalRunResult | null>(null);
  /** 基准 run 候选（同一 Agent 的已完成评测，从趋势端点获取） */
  const baseRunOptions = ref<AiEvalTrendItem[]>([]);
  const evalCompare = ref<AiEvalRunCompareResult | null>(null);
  const compareLoading = ref(false);

  async function openDetail(run: EvalRunResult) {
    evalDetail.value = run;
    evalCompare.value = null;
    detailVisible.value = true;
    const trends = await AiEvalAPI.getTrends({
      agentId: run.agentId,
      limit: 50,
    });
    baseRunOptions.value = trends.filter((item) => item.runId !== run.id);
  }

  function closeDetail() {
    detailVisible.value = false;
  }

  async function fetchCompare(runId: number, baseRunId: number) {
    compareLoading.value = true;
    try {
      evalCompare.value = await AiEvalAPI.compareRuns(runId, baseRunId);
    } finally {
      compareLoading.value = false;
    }
  }

  // ==================== 评测历史趋势 ====================
  const trendData = ref<AiEvalTrendItem[]>([]);
  const trendLoading = ref(false);

  async function fetchTrends() {
    trendLoading.value = true;
    try {
      trendData.value = await AiEvalAPI.getTrends({
        agentId: evalFilter.agentId,
        startTime: evalFilter.startTime,
        endTime: evalFilter.endTime,
        limit: 100,
      });
    } finally {
      trendLoading.value = false;
    }
  }

  // ==================== 判分模型状态 ====================
  const judgeStatus = ref<AiEvalJudgeStatus | null>(null);
  const judgeLoading = ref(false);

  async function fetchJudgeStatus() {
    judgeLoading.value = true;
    try {
      judgeStatus.value = await AiEvalAPI.getJudgeStatus();
    } finally {
      judgeLoading.value = false;
    }
  }

  // ==================== 人工复核 ====================
  const reviewQueue = ref<AiEvalReviewQueueResult | null>(null);
  const reviewStatus = ref<AiEvalReviewStatus | "all">("all");
  const reviewLoading = ref(false);
  const reviewSubmitting = ref(false);

  async function fetchReviews() {
    reviewLoading.value = true;
    try {
      reviewQueue.value = await AiEvalAPI.getReviews(
        reviewStatus.value === "all" ? {} : { status: reviewStatus.value }
      );
    } finally {
      reviewLoading.value = false;
    }
  }

  /** 复核回填（status 1→2 不可逆），回填后一致率变化联动判分状态 */
  async function submitReview(id: number, agree: boolean, remark?: string) {
    reviewSubmitting.value = true;
    try {
      await AiEvalAPI.submitReview(id, { agree, remark });
      await Promise.all([fetchReviews(), fetchJudgeStatus()]);
    } finally {
      reviewSubmitting.value = false;
    }
  }

  /** 页面挂载：总览 / 趋势 / 判分状态 / 复核队列，执行记录待选定 Agent 后拉取 */
  function refreshAll() {
    fetchOverview();
    fetchTrends();
    fetchJudgeStatus();
    fetchReviews();
  }

  return {
    evalFilter,
    // 总览
    evalOverview,
    overviewLoading,
    fetchOverview,
    // 评测集
    datasets,
    fetchDatasets,
    // 执行记录
    evalRuns,
    evalRunsTotal,
    runsLoading,
    fetchRuns,
    selectAgent,
    // 详情与对比
    detailVisible,
    evalDetail,
    baseRunOptions,
    evalCompare,
    compareLoading,
    openDetail,
    fetchCompare,
    // 趋势
    trendData,
    trendLoading,
    fetchTrends,
    // 判分状态
    judgeStatus,
    judgeLoading,
    fetchJudgeStatus,
    // 人工复核
    reviewQueue,
    reviewStatus,
    reviewLoading,
    reviewSubmitting,
    fetchReviews,
    submitReview,
    // 初始化
    refreshAll,
  };
});
