<!-- 用户端计费页：余额状态区 → 消耗总览区 → 计费明细区 → 账单与流水区 -->
<script lang="ts" setup>
import {
  computed,
  onActivated,
  onDeactivated,
  onMounted,
  ref,
  watch,
} from "vue";
import { useBillingDataStore } from "@/store/modules/billingData";
import { useBillingStore } from "@/store/modules/billing";

defineOptions({ name: "Billing", inheritAttrs: false });

const billingDataStore = useBillingDataStore();
const billingStore = useBillingStore();

const rechargeAlert = computed(() =>
  billingStore.rechargeGuideVisible ? billingStore.rechargeAlert : null
);

// 首次进入的余额由 BalanceQuotaCard 自拉，加购返回（keep-alive 重新激活）才需补拉，避免重复请求
const reentered = ref(false);

async function loadPageData() {
  billingDataStore.initScope("self");
  if (reentered.value) {
    await billingDataStore.fetchBalance();
  }
  await billingStore.fetchSummary();
}

// 余额由 BalanceQuotaCard 自行拉取，加购引导跟随余额到位后再判定
watch(
  () => billingDataStore.balance,
  () => billingStore.syncRechargeGuide(),
  { immediate: true }
);

onMounted(loadPageData);
onActivated(loadPageData);
onDeactivated(() => {
  reentered.value = true;
});
</script>

<template>
  <div class="billing-page">
    <!-- ① 余额状态区 -->
    <BalanceQuotaCard scope="self" />
    <RechargeGuide v-if="rechargeAlert" :alert="rechargeAlert" />

    <!-- ② 消耗总览区 -->
    <ConsumptionOverview />

    <!-- ③ 计费明细区 -->
    <BillingRecordList />

    <!-- ④ 账单与流水区 -->
    <BillCreditTabs />
  </div>
</template>

<style lang="scss" scoped>
.billing-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  padding: 20px;
  margin: 0 auto;
}
</style>
