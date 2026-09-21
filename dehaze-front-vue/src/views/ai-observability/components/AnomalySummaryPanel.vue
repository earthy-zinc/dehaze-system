<!-- 异常会话概览：失败/配额拒绝/中断取消计数（可观测性 summary 口径），点击进入对应异常筛选 -->
<script lang="ts" setup>
import { ArrowRight } from "@element-plus/icons-vue";
import { computed } from "vue";
import { storeToRefs } from "pinia";
import {
  useAdminAuditStore,
  type AuditAnomalyType,
} from "@/store/modules/adminAudit";

defineOptions({ name: "AnomalySummaryPanel" });

interface AnomalyEntry {
  label: string;
  count: number;
  /** 可作为会话异常类型筛选（failed/quota/canceled）；过程链口径指标仅展示 */
  anomalyType?: AuditAnomalyType;
}

const adminAuditStore = useAdminAuditStore();
const { anomalySummary, summaryLoading } = storeToRefs(adminAuditStore);

const entries = computed<AnomalyEntry[]>(() => {
  const summary = anomalySummary.value;
  return [
    {
      label: "失败会话",
      count: summary?.failedCount ?? 0,
      anomalyType: "failed",
    },
    {
      label: "配额拒绝",
      count: summary?.quotaRejected ?? 0,
      anomalyType: "quota",
    },
    {
      label: "中断/取消",
      count: summary?.interruptedCount ?? 0,
      anomalyType: "canceled",
    },
    { label: "超时调用", count: summary?.timeoutCount ?? 0 },
    { label: "高风险调用", count: summary?.highRiskCalls ?? 0 },
  ];
});

function handleClick(entry: AnomalyEntry) {
  if (entry.anomalyType) {
    adminAuditStore.applyAuditFilter({ anomalyType: entry.anomalyType });
  }
}
</script>

<template>
  <el-card shadow="never" class="!border-none" v-loading="summaryLoading">
    <template #header>
      <div class="flex items-center justify-between">
        <span class="font-bold">异常概览</span>
        <el-button
          link
          type="primary"
          @click="adminAuditStore.fetchAnomalySummary()"
        >
          刷新
        </el-button>
      </div>
    </template>
    <div class="anomaly-summary">
      <div
        v-for="entry in entries"
        :key="entry.label"
        class="anomaly-summary__item"
        :class="{ 'is-clickable': !!entry.anomalyType }"
        @click="handleClick(entry)"
      >
        <div class="anomaly-summary__count">{{ entry.count }}</div>
        <div class="anomaly-summary__label">
          {{ entry.label }}
          <el-icon v-if="entry.anomalyType" class="anomaly-summary__arrow">
            <ArrowRight />
          </el-icon>
        </div>
      </div>
    </div>
  </el-card>
</template>

<style scoped lang="scss">
.anomaly-summary {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;

  &__item {
    padding: 12px;
    text-align: center;
    cursor: default;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;

    &.is-clickable {
      cursor: pointer;

      &:hover {
        background-color: var(--el-color-primary-light-9);
      }
    }
  }

  &__count {
    font-size: 24px;
    font-weight: 600;
    color: var(--el-color-danger);
  }

  &__label {
    display: flex;
    gap: 2px;
    align-items: center;
    justify-content: center;
    margin-top: 4px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__arrow {
    font-size: 12px;
  }
}
</style>
