// 管理端 AI 可观测中心 Store：异常总览 / 性能趋势 / 资源消耗 / 审计检索 / 过程链下钻
import {
  AiObservabilityAPI,
  type AiObservabilityCostDimension,
  type AiObservabilityCostItem,
  type AiObservabilityCostTrendItem,
  type AiObservabilityStatus,
  type AiObservabilitySummary,
  type AiObservabilityTraceDetail,
  type AiObservabilityTraceItem,
  type AiObservabilityTraceQuery,
  type AiObservabilityTrendDimension,
  type AiObservabilityTrendItem,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { downloadBlob } from "@/composables/useImportExport";

/** 趋势/资源面板时间范围（天） */
export type MetricRange = 7 | 30 | 90;

function fmtDate(date: Date) {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** 近 N 天的查询时间范围（含首尾整天） */
function resolveRange(days: MetricRange) {
  const start = new Date();
  start.setDate(start.getDate() - days + 1);
  return {
    startTime: `${fmtDate(start)} 00:00:00`,
    endTime: `${fmtDate(new Date())} 23:59:59`,
  };
}

interface AuditFilter {
  userId?: number;
  conversationId?: number;
  status?: AiObservabilityStatus;
  agentCode?: string;
  model?: string;
  /** [开始日期, 结束日期]（YYYY-MM-DD），无筛选为 null */
  timeRange: [string, string] | null;
}

export const useAdminObservabilityStore = defineStore(
  "adminObservability",
  () => {
    // ==================== 异常总览 ====================
    const summary = ref<AiObservabilitySummary | null>(null);
    const summaryLoading = ref(false);

    async function fetchAnomalySummary() {
      summaryLoading.value = true;
      try {
        summary.value = await AiObservabilityAPI.getSummary();
      } finally {
        summaryLoading.value = false;
      }
    }

    // ==================== 性能趋势 ====================
    const trendDimension = ref<AiObservabilityTrendDimension>("model");
    const trendRange = ref<MetricRange>(7);
    const trendData = ref<AiObservabilityTrendItem[]>([]);
    const trendLoading = ref(false);

    async function fetchTrends() {
      trendLoading.value = true;
      try {
        trendData.value = await AiObservabilityAPI.getTrends({
          dimension: trendDimension.value,
          ...resolveRange(trendRange.value),
        });
      } finally {
        trendLoading.value = false;
      }
    }

    // ==================== 资源消耗 ====================
    const costDimension = ref<AiObservabilityCostDimension>("model");
    const costRange = ref<MetricRange>(30);
    const costItems = ref<AiObservabilityCostItem[]>([]);
    const costTrend = ref<AiObservabilityCostTrendItem[]>([]);
    const costTotal = ref(0);
    const costPageNum = ref(1);
    const costPageSize = ref(10);
    const costLoading = ref(false);

    async function fetchCosts() {
      costLoading.value = true;
      try {
        const result = await AiObservabilityAPI.getCosts({
          dimension: costDimension.value,
          pageNum: costPageNum.value,
          pageSize: costPageSize.value,
          ...resolveRange(costRange.value),
        });
        costItems.value = result.items ?? [];
        costTrend.value = result.trend ?? [];
        costTotal.value = result.total ?? 0;
      } finally {
        costLoading.value = false;
      }
    }

    function setCostDimension(dimension: AiObservabilityCostDimension) {
      costDimension.value = dimension;
      costPageNum.value = 1;
      fetchCosts();
    }

    function setCostRange(range: MetricRange) {
      costRange.value = range;
      costPageNum.value = 1;
      fetchCosts();
    }

    // ==================== 审计检索 ====================
    const auditFilter = reactive<AuditFilter>({ timeRange: null });
    const traceList = ref<AiObservabilityTraceItem[]>([]);
    const traceTotal = ref(0);
    const auditPageNum = ref(1);
    const auditPageSize = ref(10);
    const tracesLoading = ref(false);

    function buildTraceQuery(): AiObservabilityTraceQuery {
      const [start, end] = auditFilter.timeRange ?? [];
      return {
        pageNum: auditPageNum.value,
        pageSize: auditPageSize.value,
        userId: auditFilter.userId || undefined,
        conversationId: auditFilter.conversationId || undefined,
        status: auditFilter.status,
        agentCode: auditFilter.agentCode?.trim() || undefined,
        model: auditFilter.model?.trim() || undefined,
        startTime: start ? `${start} 00:00:00` : undefined,
        endTime: end ? `${end} 23:59:59` : undefined,
      };
    }

    async function fetchTraces() {
      tracesLoading.value = true;
      try {
        const page = await AiObservabilityAPI.getTraces(buildTraceQuery());
        traceList.value = page.list ?? [];
        traceTotal.value = page.total ?? 0;
      } finally {
        tracesLoading.value = false;
      }
    }

    function searchTraces() {
      auditPageNum.value = 1;
      fetchTraces();
    }

    function resetAuditFilter() {
      auditFilter.userId = undefined;
      auditFilter.conversationId = undefined;
      auditFilter.status = undefined;
      auditFilter.agentCode = undefined;
      auditFilter.model = undefined;
      auditFilter.timeRange = null;
      searchTraces();
    }

    /** 异常总览卡片跳审计检索：按状态收敛筛选 */
    function filterTracesByStatus(status?: AiObservabilityStatus) {
      auditFilter.status = status;
      searchTraces();
    }

    // ==================== 导出 ====================
    const exportLoading = ref(false);

    async function exportTraces() {
      exportLoading.value = true;
      try {
        const { pageNum, pageSize, ...filters } = buildTraceQuery();
        const blob = await AiObservabilityAPI.exportTraces(filters);
        downloadBlob(blob, `ai-trace-export_${Date.now()}.csv`);
        ElMessage.success("导出成功，已开始下载");
      } finally {
        exportLoading.value = false;
      }
    }

    // ==================== 过程链下钻 ====================
    const detailVisible = ref(false);
    const detailLoading = ref(false);
    const detailNotFound = ref(false);
    const traceDetail = ref<AiObservabilityTraceDetail | null>(null);

    /** 详情 404（A0401）：跨会话/不存在均统一返回，不暴露存在性 */
    function isNotFound(err: unknown) {
      const bizCode = (err as { response?: { data?: { code?: string } } })
        ?.response?.data?.code;
      return bizCode === "A0401";
    }

    async function fetchTraceDetail(traceId: string) {
      detailVisible.value = true;
      detailLoading.value = true;
      detailNotFound.value = false;
      traceDetail.value = null;
      try {
        traceDetail.value = await AiObservabilityAPI.getTraceDetail(traceId);
      } catch (err) {
        if (isNotFound(err)) {
          detailNotFound.value = true;
        }
        // 其余错误拦截器已 toast，抽屉保持空态
      } finally {
        detailLoading.value = false;
      }
    }

    return {
      // 异常总览
      summary,
      summaryLoading,
      fetchAnomalySummary,
      // 趋势
      trendDimension,
      trendRange,
      trendData,
      trendLoading,
      fetchTrends,
      // 资源消耗
      costDimension,
      costRange,
      costItems,
      costTrend,
      costTotal,
      costPageNum,
      costPageSize,
      costLoading,
      fetchCosts,
      // 审计检索
      auditFilter,
      traceList,
      traceTotal,
      auditPageNum,
      auditPageSize,
      tracesLoading,
      fetchTraces,
      searchTraces,
      resetAuditFilter,
      filterTracesByStatus,
      // 导出
      exportLoading,
      exportTraces,
      // 过程链下钻
      detailVisible,
      detailLoading,
      detailNotFound,
      traceDetail,
      fetchTraceDetail,
    };
  }
);
