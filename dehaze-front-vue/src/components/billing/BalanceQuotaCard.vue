<!-- 余额与配额卡：用户端(scope=self)与管理端(scope=userId)共用 -->
<script lang="ts" setup>
import { computed, watch } from "vue";
import type { BillingDataScope } from "@/store/modules/billingData";
import { useBillingDataStore } from "@/store/modules/billingData";

defineOptions({ name: "BalanceQuotaCard" });

const props = withDefaults(
  defineProps<{
    scope?: BillingDataScope;
  }>(),
  { scope: "self" }
);

const store = useBillingDataStore();

const balance = computed(() => store.balance);

/** 限额仅作防滥用阈值不计余额，达限标红、≥80% 黄色预警 */
function quotaColor(used: number, limit: number) {
  if (limit <= 0) return "#409eff";
  if (used >= limit) return "#f56c6c";
  if (used / limit >= 0.8) return "#e6a23c";
  return "#409eff";
}

function quotaPercent(used: number, limit: number) {
  if (limit <= 0) return 0;
  return Math.min(100, Math.round((used / limit) * 100));
}

watch(
  () => props.scope,
  (next) => {
    store.initScope(next);
    store.fetchBalance();
  },
  { immediate: true }
);
</script>

<template>
  <el-card v-loading="store.balanceLoading" shadow="never">
    <div class="flex flex-wrap items-center gap-10">
      <div>
        <div class="text-sm text-gray-400">当前积分余额</div>
        <div class="text-[32px] leading-10 font-bold">
          {{ balance?.creditsBalance ?? 0 }}
        </div>
        <el-tag v-if="balance?.arrearsStatus" type="danger" size="small">
          已欠费
        </el-tag>
      </div>

      <div class="min-w-[220px] flex-1">
        <div class="mb-1 text-sm">
          日限额：{{ balance?.dailyUsed ?? 0 }} / {{ balance?.dailyLimit ?? 0 }}
        </div>
        <el-progress
          :percentage="
            quotaPercent(balance?.dailyUsed ?? 0, balance?.dailyLimit ?? 0)
          "
          :color="quotaColor(balance?.dailyUsed ?? 0, balance?.dailyLimit ?? 0)"
        />
        <div class="mt-3 mb-1 text-sm">
          月限额：{{ balance?.monthlyUsed ?? 0 }} /
          {{ balance?.monthlyLimit ?? 0 }}
        </div>
        <el-progress
          :percentage="
            quotaPercent(balance?.monthlyUsed ?? 0, balance?.monthlyLimit ?? 0)
          "
          :color="
            quotaColor(balance?.monthlyUsed ?? 0, balance?.monthlyLimit ?? 0)
          "
        />
      </div>
    </div>
  </el-card>
</template>
