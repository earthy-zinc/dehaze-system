<!-- 过程摘要：步数/耗时/消耗，异常时显著标注失败环节（纯展示，props 进） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatTraceSummaryVM } from "../types";

defineOptions({ name: "TraceSummary" });

const props = defineProps<{
  summary: ChatTraceSummaryVM;
}>();

const hasFailed = computed(() => (props.summary.failedCount ?? 0) > 0);

const durationText = computed(() => {
  const ms = props.summary.durationMs;
  if (ms == null) return null;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} 秒` : `${ms} ms`;
});
</script>

<template>
  <div class="trace-summary" :class="{ 'trace-summary--failed': hasFailed }">
    <span class="trace-summary__item">共 {{ summary.stepCount }} 步</span>
    <span v-if="durationText" class="trace-summary__item">
      耗时 {{ durationText }}
    </span>
    <span v-if="summary.credits != null" class="trace-summary__item">
      消耗 {{ summary.credits }} 积分
    </span>
    <el-tag v-if="hasFailed" type="danger" size="small">
      {{ summary.failedCount }} 步失败
    </el-tag>
  </div>
</template>

<style scoped lang="scss">
.trace-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  font-size: 12px;
  color: var(--el-text-color-secondary);

  &--failed {
    color: var(--el-color-danger);
  }

  &__item {
    white-space: nowrap;
  }
}
</style>
