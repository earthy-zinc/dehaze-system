<!-- 子智能体并行消耗面板：各子智能体累计 Token 与积分 -->
<script lang="ts" setup>
import type { ThoughtEvent, TokenUsage } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "SubAgentPanel" });

const props = defineProps<{
  steps: ThoughtEvent[];
  usage: TokenUsage | Record<string, unknown> | null;
}>();

const stepCount = computed(() => props.steps.length);

const usageEntries = computed(() => {
  if (!props.usage || typeof props.usage !== "object") return [];
  return Object.entries(props.usage).map(([key, value]) => ({ key, value }));
});
</script>

<template>
  <div class="sub-agent-panel">
    <div class="sub-agent-panel__title">
      子智能体并行执行（{{ stepCount }} 步）
    </div>
    <div v-if="usageEntries.length > 0" class="sub-agent-panel__usage">
      <span
        v-for="entry in usageEntries"
        :key="entry.key"
        class="sub-agent-panel__item"
      >
        {{ entry.key }}: {{ entry.value }}
      </span>
    </div>
  </div>
</template>

<style scoped lang="scss">
.sub-agent-panel {
  padding: 8px 10px;
  margin-bottom: 8px;
  background-color: var(--el-color-primary-light-9);
  border-radius: 6px;

  &__title {
    font-size: 13px;
    font-weight: 600;
  }

  &__usage {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 4px;
  }

  &__item {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
