// 计费数据 Store：用户端（scope=self）与管理端（scope=userId）共用的数据视图层
import {
  AiBillingAPI,
  BalanceVO,
  BillingRecordQuery,
  BillingRecordVO,
  CreditLogQuery,
  CreditLogVO,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { computed, reactive, ref } from "vue";

export type BillingDataScope = "self" | number;

/** 误扣申诉状态：0-无，1-待审核，2-已通过，3-已驳回 */
export type RefundStatus = NonNullable<BillingRecordVO["refundStatus"]>;

const emptyRecordQuery = (): BillingRecordQuery => ({
  pageNum: 1,
  pageSize: 10,
  billType: undefined,
  dateStart: undefined,
  dateEnd: undefined,
});

const emptyCreditLogQuery = (): CreditLogQuery => ({
  pageNum: 1,
  pageSize: 10,
  source: undefined,
});

export const useBillingDataStore = defineStore("billingData", () => {
  const scope = ref<BillingDataScope>("self");

  const balance = ref<BalanceVO | null>(null);
  const balanceLoading = ref(false);

  const records = ref<BillingRecordVO[]>([]);
  const recordsTotal = ref(0);
  const recordsLoading = ref(false);

  const creditLogs = ref<CreditLogVO[]>([]);
  const creditLogsTotal = ref(0);
  const creditLogsLoading = ref(false);

  const recordQuery = reactive<BillingRecordQuery>(emptyRecordQuery());
  const creditLogQuery = reactive<CreditLogQuery>(emptyCreditLogQuery());

  /** scope=self 时不传 userId，后端按登录态取本人数据 */
  const userId = computed(() =>
    typeof scope.value === "number" ? scope.value : undefined
  );

  /** 注入数据范围，由宿主页面在挂载时调用 */
  function initScope(next: BillingDataScope) {
    scope.value = next;
  }

  async function fetchBalance() {
    balanceLoading.value = true;
    try {
      balance.value = await AiBillingAPI.getBalance(userId.value);
    } finally {
      balanceLoading.value = false;
    }
  }

  async function fetchRecords() {
    recordsLoading.value = true;
    try {
      const result = await AiBillingAPI.getRecords({
        ...recordQuery,
        userId: userId.value,
      });
      records.value = result.list ?? [];
      recordsTotal.value = result.total ?? 0;
    } finally {
      recordsLoading.value = false;
    }
  }

  async function fetchCreditLogs() {
    creditLogsLoading.value = true;
    try {
      const result = await AiBillingAPI.getCreditLogs({
        ...creditLogQuery,
        userId: userId.value,
      });
      creditLogs.value = result.list ?? [];
      creditLogsTotal.value = result.total ?? 0;
    } finally {
      creditLogsLoading.value = false;
    }
  }

  /** 申诉提交后本地置位避免整页刷新，下次拉取以服务端为准 */
  function refreshRecordStatus(billingId: number, refundStatus: RefundStatus) {
    const record = records.value.find((r) => r.id === billingId);
    if (record) {
      record.refundStatus = refundStatus;
    }
  }

  function resetScope() {
    scope.value = "self";
    balance.value = null;
    records.value = [];
    recordsTotal.value = 0;
    creditLogs.value = [];
    creditLogsTotal.value = 0;
    Object.assign(recordQuery, emptyRecordQuery());
    Object.assign(creditLogQuery, emptyCreditLogQuery());
  }

  return {
    scope,
    balance,
    balanceLoading,
    records,
    recordsTotal,
    recordsLoading,
    creditLogs,
    creditLogsTotal,
    creditLogsLoading,
    recordQuery,
    creditLogQuery,
    userId,
    initScope,
    fetchBalance,
    fetchRecords,
    fetchCreditLogs,
    refreshRecordStatus,
    resetScope,
  };
});
