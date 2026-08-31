<!-- 异常总览：失败/中断/超时点击跳审计检索收敛状态；配额拒绝与高风险调用为统计口径展示 -->
<template>
  <el-card v-loading="store.summaryLoading" shadow="never">
    <template #header>
      <div class="flex justify-between items-center">
        <span>异常总览</span>
        <el-button
          :loading="store.summaryLoading"
          @click="store.fetchAnomalySummary()"
        >
          <el-icon><Refresh /></el-icon>刷新
        </el-button>
      </div>
    </template>

    <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-[12px]">
      <div
        v-for="tile in tiles"
        :key="tile.label"
        class="tile"
        :class="{ clickable: tile.status != null }"
        @click="tile.status != null && emit('select', tile.status)"
      >
        <div
          class="tile-value"
          :class="tile.danger && tile.value > 0 ? 'text-red-500' : ''"
        >
          {{ tile.value }}
        </div>
        <div class="tile-label">
          {{ tile.label }}
          <el-icon v-if="tile.status != null" class="tile-arrow"
            ><Right
          /></el-icon>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script lang="ts" setup>
import { Refresh, Right } from "@element-plus/icons-vue";
import type { AiObservabilityStatus } from "dehaze-sdk-js";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "AnomalyOverview" });

const emit = defineEmits<{
  (e: "select", status: AiObservabilityStatus): void;
}>();

const store = useAdminObservabilityStore();

// 配额拒绝/高风险调用无独立检索参数，仅作统计展示，不提供跳转
const tiles = computed(() => {
  const summary = store.summary;
  return [
    {
      label: "总调用",
      value: summary?.total ?? 0,
      status: undefined as AiObservabilityStatus | undefined,
      danger: false,
    },
    {
      label: "成功",
      value: summary?.successCount ?? 0,
      status: 1 as const,
      danger: false,
    },
    {
      label: "失败",
      value: summary?.failedCount ?? 0,
      status: 2 as const,
      danger: true,
    },
    {
      label: "中断",
      value: summary?.interruptedCount ?? 0,
      status: 3 as const,
      danger: true,
    },
    {
      label: "超时",
      value: summary?.timeoutCount ?? 0,
      status: 4 as const,
      danger: true,
    },
    {
      label: "配额拒绝",
      value: summary?.quotaRejected ?? 0,
      status: undefined,
      danger: true,
    },
    {
      label: "高风险调用",
      value: summary?.highRiskCalls ?? 0,
      status: undefined,
      danger: true,
    },
  ];
});
</script>

<style lang="scss" scoped>
.tile {
  padding: 12px;
  text-align: center;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  transition: box-shadow 0.2s;

  &.clickable {
    cursor: pointer;

    &:hover {
      box-shadow: var(--el-box-shadow-light);
    }
  }

  .tile-value {
    font-size: 24px;
    font-weight: 600;
  }

  .tile-label {
    display: flex;
    gap: 2px;
    align-items: center;
    justify-content: center;
    margin-top: 4px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  .tile-arrow {
    color: var(--el-color-primary);
  }
}
</style>
