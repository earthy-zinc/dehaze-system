// 用户端计费页 Store：消耗汇总、月结账单、误扣申诉、加购引导
import {
  AiBillingAPI,
  BillVO,
  BillingRecordVO,
  BillingRefundApplyForm,
  BillingSummaryVO,
} from "dehaze-sdk-js";
import { ElMessage } from "element-plus";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { downloadBlob } from "@/composables/useImportExport";
import { useBillingDataStore } from "./billingData";

export type SummaryDimension = "day" | "month";

/** 加购引导触发原因：欠费 / 配额达限 / 余额偏低 */
export type RechargeAlertReason = "arrears" | "quota" | "low";

export interface RechargeAlert {
  reason: RechargeAlertReason;
  /** 建议补足的积分缺口，配额达限时为 0 */
  gap: number;
}

// 余额低于该阈值即提示加购，避免余额归零后才被动中断服务
const LOW_BALANCE_CREDITS = 100;

function formatMonth(date: Date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

export const useBillingStore = defineStore("billing", () => {
  const billingDataStore = useBillingDataStore();

  const consumptionSummary = ref<BillingSummaryVO | null>(null);
  const summaryDimension = ref<SummaryDimension>("day");
  const currentBill = ref<BillVO | null>(null);
  const currentMonth = ref(formatMonth(new Date()));
  const refundDialog = ref({ visible: false, billingId: 0 });
  // 申诉目标记录快照：退款申请需携带该条实扣积分
  const refundTarget = ref<BillingRecordVO | null>(null);
  const rechargeGuideVisible = ref(false);
  const loading = ref(false);
  // 账单查询与消耗汇总互不阻塞，加载态分开以免联动闪烁
  const billLoading = ref(false);

  const rechargeAlert = computed<RechargeAlert | null>(() => {
    const balance = billingDataStore.balance;
    if (!balance) return null;
    if (balance.arrearsStatus) {
      return { reason: "arrears", gap: Math.max(0, -balance.creditsBalance) };
    }
    const quotaReached =
      (balance.dailyLimit > 0 && balance.dailyUsed >= balance.dailyLimit) ||
      (balance.monthlyLimit > 0 && balance.monthlyUsed >= balance.monthlyLimit);
    if (quotaReached) {
      return { reason: "quota", gap: 0 };
    }
    if (balance.creditsBalance < LOW_BALANCE_CREDITS) {
      return {
        reason: "low",
        gap: LOW_BALANCE_CREDITS - balance.creditsBalance,
      };
    }
    return null;
  });

  async function setDimension(dimension: SummaryDimension) {
    summaryDimension.value = dimension;
    await fetchSummary();
  }

  async function fetchSummary() {
    loading.value = true;
    try {
      consumptionSummary.value = await AiBillingAPI.getSummary(
        summaryDimension.value
      );
    } finally {
      loading.value = false;
    }
  }

  async function fetchBill(month: string) {
    billLoading.value = true;
    try {
      currentMonth.value = month;
      currentBill.value = await AiBillingAPI.getBill(month);
    } finally {
      billLoading.value = false;
    }
  }

  async function downloadBill(month: string) {
    const bill = await AiBillingAPI.downloadBill(month);
    const content = JSON.stringify(bill, null, 2);
    downloadBlob(
      new Blob([content], { type: "application/json;charset=utf-8" }),
      `billing-${month}.json`
    );
    ElMessage.success("账单已下载");
  }

  function openRefund(record: BillingRecordVO) {
    refundTarget.value = record;
    refundDialog.value = { visible: true, billingId: record.id };
  }

  function closeRefund() {
    refundDialog.value.visible = false;
  }

  async function submitRefund(
    billingId: number,
    form: Omit<BillingRefundApplyForm, "billingId">
  ) {
    await AiBillingAPI.applyRefund({ billingId, ...form });
    billingDataStore.refreshRecordStatus(billingId, 1);
    closeRefund();
  }

  function syncRechargeGuide() {
    rechargeGuideVisible.value = rechargeAlert.value !== null;
  }

  return {
    consumptionSummary,
    summaryDimension,
    currentBill,
    currentMonth,
    refundDialog,
    refundTarget,
    rechargeGuideVisible,
    rechargeAlert,
    loading,
    billLoading,
    setDimension,
    fetchSummary,
    fetchBill,
    downloadBill,
    openRefund,
    closeRefund,
    submitRefund,
    syncRechargeGuide,
  };
});
