<!-- 计费明细区：公共计费明细表 + 端内误扣申诉弹窗 -->
<script lang="ts" setup>
import { useBillingStore } from "@/store/modules/billing";

defineOptions({ name: "BillingRecordList" });

const billingStore = useBillingStore();
</script>

<template>
  <el-card shadow="never" class="record-card">
    <template #header>
      <span class="card-title">计费明细</span>
    </template>

    <BillingRecordTable
      scope="self"
      :show-refund-action="true"
      @refund="billingStore.openRefund"
    />

    <RefundApplyDialog :record="billingStore.refundTarget" />
  </el-card>
</template>

<style lang="scss" scoped>
.record-card {
  border-radius: 12px;

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }
}
</style>
