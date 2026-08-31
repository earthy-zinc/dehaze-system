<!-- 账单与流水区：积分流水 / 月结账单查询与下载 -->
<script lang="ts" setup>
import { BillingType } from "dehaze-sdk-js";
import { Download } from "@element-plus/icons-vue";
import { computed, ref, watch } from "vue";
import { useBillingStore } from "@/store/modules/billing";

defineOptions({ name: "BillCreditTabs" });

const BILL_TYPE_LABELS: Record<BillingType, string> = {
  chat: "AI 对话",
  tool_llm: "工具推理",
  kb_inject: "知识库注入",
  asr: "语音识别",
  tts: "语音合成",
};

const billingStore = useBillingStore();

const activeTab = ref("credit");

const billItems = computed(() =>
  Object.entries(billingStore.currentBill?.itemSummary ?? {}).map(
    ([type, credits]) => ({
      label: BILL_TYPE_LABELS[type as BillingType] ?? type,
      credits,
    })
  )
);

// 账单 Tab 首次切入时才拉数据，避免与流水表同时抢占首屏请求
watch(activeTab, (tab) => {
  if (tab === "bill" && !billingStore.currentBill) {
    billingStore.fetchBill(billingStore.currentMonth);
  }
});
</script>

<template>
  <el-card shadow="never" class="bill-card">
    <template #header>
      <span class="card-title">账单与流水</span>
    </template>

    <el-tabs v-model="activeTab">
      <el-tab-pane label="积分流水" name="credit">
        <CreditLogTable scope="self" />
      </el-tab-pane>

      <el-tab-pane label="月结账单" name="bill">
        <div class="bill-toolbar">
          <el-date-picker
            v-model="billingStore.currentMonth"
            type="month"
            value-format="YYYY-MM"
            placeholder="选择账单月份"
            :clearable="false"
            @change="billingStore.fetchBill"
          />
          <el-button
            type="primary"
            :disabled="!billingStore.currentBill"
            @click="billingStore.downloadBill(billingStore.currentMonth)"
          >
            <el-icon class="mr-1"><Download /></el-icon>
            下载账单
          </el-button>
        </div>

        <div v-loading="billingStore.billLoading">
          <template v-if="billingStore.currentBill">
            <el-descriptions :column="3" border class="bill-summary">
              <el-descriptions-item label="账单月份">
                {{ billingStore.currentBill.month }}
              </el-descriptions-item>
              <el-descriptions-item label="总消耗">
                {{ billingStore.currentBill.totalConsume }} 积分
              </el-descriptions-item>
              <el-descriptions-item label="总充值">
                {{ billingStore.currentBill.totalRecharge }} 积分
              </el-descriptions-item>
              <el-descriptions-item label="总退款">
                {{ billingStore.currentBill.totalRefund }} 积分
              </el-descriptions-item>
              <el-descriptions-item label="月初余额">
                {{ billingStore.currentBill.balanceStart }} 积分
              </el-descriptions-item>
              <el-descriptions-item label="月末余额">
                {{ billingStore.currentBill.balanceEnd }} 积分
              </el-descriptions-item>
            </el-descriptions>

            <el-table :data="billItems" size="small" class="bill-items">
              <el-table-column prop="label" label="计费类型" min-width="140" />
              <el-table-column label="消耗积分" min-width="120">
                <template #default="{ row }">{{ row.credits }}</template>
              </el-table-column>
            </el-table>
          </template>

          <el-empty v-else description="该月份暂无账单数据" :image-size="90" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </el-card>
</template>

<style lang="scss" scoped>
.bill-card {
  border-radius: 12px;

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .bill-toolbar {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
  }

  .bill-summary {
    margin-bottom: 16px;
  }

  .bill-items {
    border-radius: 8px;
  }
}
</style>
