import {
  AiBillingAPI,
  AnomalyRecordQuery,
  AnomalyRecordVO,
  BillingRefundQuery,
  BillingRefundVO,
  BillingStatGroupBy,
  BillingStatQuery,
  BillingStatVO,
  CostStatQuery,
  CostStatVO,
  CreditAdjustForm,
  ModelCostForm,
  ModelCostQuery,
  ModelCostVO,
} from "dehaze-sdk-js";
import { useBillingDataStore } from "@/store/modules/billingData";

export type OverviewPeriod = "month" | "quarter" | "custom";
export type DrilldownDimension = "user" | "model" | "provider" | "day";

/** 调价模拟结果（前端估算口径） */
export interface PriceImpactResult {
  currentCost: number;
  simulatedCost: number;
  currentProfit: number;
  simulatedProfit: number;
}

// 管理端计费 Store：概览双口径指标 / 下钻聚合 / 成本单价版本 / 对账 / 异常 / 积分调整
export const useAdminBillingStore = defineStore("adminBilling", () => {
  // ==================== 概览看板 ====================
  const overview = ref<CostStatVO[]>([]);
  const overviewPeriod = ref<OverviewPeriod>("month");
  const periodRange = ref<[string, string]>(["", ""]);
  const overviewLoading = ref(false);

  const overallStat = computed(
    () => overview.value.find((row) => row.metric === "overall") ?? null
  );
  const aiStat = computed(
    () => overview.value.find((row) => row.metric === "ai") ?? null
  );

  /** 周期口径：月=当月、季=当季、自定义=periodRange，返回 yyyy-MM-dd */
  function resolvePeriodRange(): { startTime: string; endTime: string } {
    const now = new Date();
    const fmt = (d: Date) => d.toISOString().slice(0, 10);
    if (overviewPeriod.value === "month") {
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      return { startTime: fmt(start), endTime: fmt(now) };
    }
    if (overviewPeriod.value === "quarter") {
      const quarterStartMonth = Math.floor(now.getMonth() / 3) * 3;
      const start = new Date(now.getFullYear(), quarterStartMonth, 1);
      return { startTime: fmt(start), endTime: fmt(now) };
    }
    return { startTime: periodRange.value[0], endTime: periodRange.value[1] };
  }

  async function fetchOverview() {
    overviewLoading.value = true;
    try {
      const { startTime, endTime } = resolvePeriodRange();
      overview.value = await AiBillingAPI.getCostStats({ startTime, endTime });
    } finally {
      overviewLoading.value = false;
    }
  }

  // ==================== 消耗与成本下钻 ====================
  const drilldownDimension = ref<DrilldownDimension>("user");
  const drilldownData = ref<BillingStatVO[]>([]);
  /** 供应商维度成本统计（BillingStatVO 无收入/成本字段，单独承载） */
  const providerStats = ref<CostStatVO[]>([]);
  const drilldownLoading = ref(false);
  const selectedUserId = ref<number | null>(null);

  async function fetchDrilldown() {
    drilldownLoading.value = true;
    selectedUserId.value = null;
    const { startTime, endTime } = resolvePeriodRange();
    try {
      if (drilldownDimension.value === "provider") {
        // BillingStatGroupBy 无 provider，供应商维度走成本统计接口
        const query: CostStatQuery = { startTime, endTime };
        providerStats.value = await AiBillingAPI.getCostStats(query);
        drilldownData.value = [];
        return;
      }
      const groupBy = drilldownDimension.value as BillingStatGroupBy;
      const query: BillingStatQuery = {
        groupBy,
        dateStart: startTime,
        dateEnd: endTime,
      };
      drilldownData.value = await AiBillingAPI.getStats(query);
      providerStats.value = [];
    } finally {
      drilldownLoading.value = false;
    }
  }

  // ==================== 成本单价 ====================
  const costVersions = ref<ModelCostVO[]>([]);
  const costTotal = ref(0);
  const costQuery = reactive<ModelCostQuery>({ pageNum: 1, pageSize: 10 });
  const costLoading = ref(false);

  async function fetchCostVersions() {
    costLoading.value = true;
    try {
      const page = await AiBillingAPI.getCosts(costQuery);
      costVersions.value = page.list ?? [];
      costTotal.value = page.total ?? 0;
    } finally {
      costLoading.value = false;
    }
  }

  /** 新增（无 id）或更新（有 id）成本价格版本 */
  async function saveCostVersion(form: ModelCostForm, id?: number) {
    if (id) {
      await AiBillingAPI.updateCost(id, form);
    } else {
      await AiBillingAPI.createCost(form);
    }
    await fetchCostVersions();
  }

  async function deleteCostVersion(id: number) {
    await AiBillingAPI.deleteCost(id);
    await fetchCostVersions();
  }

  // ==================== 账单对账 ====================
  const reconcileImporting = ref(false);

  async function importReconcileBill(
    content: string,
    startTime: string,
    endTime: string
  ) {
    reconcileImporting.value = true;
    try {
      const result = await AiBillingAPI.importReconcile({
        content,
        startTime,
        endTime,
      });
      return result.imported;
    } finally {
      reconcileImporting.value = false;
    }
  }

  // ==================== 调价影响测算（前端演示口径） ====================
  /**
   * 按新档位均价与现有成本价估算重算：模拟成本 = 当前成本 × (新均价/现有均价)，
   * 仅反映"调价前后单位成本比例"，用于毛利影响演示；精确测算以后端实现为准。
   */
  async function simulatePriceImpact(
    form: ModelCostForm
  ): Promise<PriceImpactResult | null> {
    const { startTime, endTime } = resolvePeriodRange();
    const [stats, versionsPage] = await Promise.all([
      AiBillingAPI.getCostStats({ startTime, endTime }),
      AiBillingAPI.getCosts({
        pageNum: 1,
        pageSize: 1,
        modelId: form.modelId,
        providerId: form.providerId,
      }),
    ]);
    const current = stats.find((row) => row.metric === "overall");
    if (!current) return null;

    const latest = (versionsPage.list ?? [])[0];
    const oldPrice = latest?.details?.length
      ? latest.details.reduce((sum, d) => sum + d.unitPrice, 0) /
        latest.details.length
      : 0;
    const newPrice = form.details?.length
      ? form.details.reduce((sum, d) => sum + d.unitPrice, 0) /
        form.details.length
      : 0;
    const ratio = oldPrice > 0 ? newPrice / oldPrice : 1;

    const simulatedCost = current.cost * ratio;
    return {
      currentCost: current.cost,
      simulatedCost,
      currentProfit: current.profit,
      simulatedProfit: current.revenue - simulatedCost,
    };
  }

  // ==================== 异常监控 ====================
  const anomalies = ref<AnomalyRecordVO[]>([]);
  const anomalyTotal = ref(0);
  const anomalyFilter = reactive<{
    anomalyType?: string;
    dateStart?: string;
    dateEnd?: string;
  }>({});
  const anomalyPageNum = ref(1);
  const anomalyPageSize = ref(10);
  const anomalyLoading = ref(false);

  async function fetchAnomalies() {
    anomalyLoading.value = true;
    try {
      const query: AnomalyRecordQuery = {
        pageNum: anomalyPageNum.value,
        pageSize: anomalyPageSize.value,
        ...anomalyFilter,
      };
      const page = await AiBillingAPI.getAnomalies(query);
      anomalies.value = page.list ?? [];
      anomalyTotal.value = page.total ?? 0;
    } finally {
      anomalyLoading.value = false;
    }
  }

  // ==================== 积分调整 ====================
  const adjustDialog = reactive<{ userId?: number; visible: boolean }>({
    visible: false,
  });

  async function submitCreditAdjust(form: CreditAdjustForm) {
    await AiBillingAPI.adjustCredits(form);
    adjustDialog.visible = false;
  }

  // ==================== 退款审核 ====================
  const refunds = ref<BillingRefundVO[]>([]);
  const refundTotal = ref(0);
  const refundFilter = reactive<{ status?: number; userId?: number }>({});
  const refundPageNum = ref(1);
  const refundPageSize = ref(10);
  const refundLoading = ref(false);

  async function fetchRefunds() {
    refundLoading.value = true;
    try {
      const query: BillingRefundQuery = {
        pageNum: refundPageNum.value,
        pageSize: refundPageSize.value,
        ...refundFilter,
      };
      const page = await AiBillingAPI.getRefunds(query);
      refunds.value = page.list ?? [];
      refundTotal.value = page.total ?? 0;
    } finally {
      refundLoading.value = false;
    }
  }

  async function auditRefund(
    refundId: number,
    approved: boolean,
    auditRemark?: string
  ) {
    const refund = await AiBillingAPI.auditRefund(refundId, {
      approved,
      auditRemark,
    });
    // 审核结果本地置位刷新目标用户明细行状态，下次拉取以服务端为准
    const billingDataStore = useBillingDataStore();
    billingDataStore.refreshRecordStatus(refund.billingId, approved ? 2 : 3);
    await fetchRefunds();
    return refund;
  }

  return {
    // 概览
    overview,
    overviewPeriod,
    periodRange,
    overviewLoading,
    overallStat,
    aiStat,
    fetchOverview,
    resolvePeriodRange,
    // 下钻
    drilldownDimension,
    drilldownData,
    providerStats,
    drilldownLoading,
    selectedUserId,
    fetchDrilldown,
    // 成本单价
    costVersions,
    costTotal,
    costQuery,
    costLoading,
    fetchCostVersions,
    saveCostVersion,
    deleteCostVersion,
    // 对账
    reconcileImporting,
    importReconcileBill,
    // 调价测算
    simulatePriceImpact,
    // 异常
    anomalies,
    anomalyTotal,
    anomalyFilter,
    anomalyPageNum,
    anomalyPageSize,
    anomalyLoading,
    fetchAnomalies,
    // 积分调整
    adjustDialog,
    submitCreditAdjust,
    // 退款审核
    refunds,
    refundTotal,
    refundFilter,
    refundPageNum,
    refundPageSize,
    refundLoading,
    fetchRefunds,
    auditRefund,
  };
});
