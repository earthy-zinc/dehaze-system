<!-- 单个推理步骤：序号/耗时/thought/工具调用/observation/status -->
<script lang="ts" setup>
import type { ThoughtEvent } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "ThoughtStep" });

const props = defineProps<{
  step: ThoughtEvent;
}>();

const statusMeta = computed(() => {
  switch (props.step.status) {
    case 1:
      return { label: "成功", type: "success" as const };
    case 2:
      return { label: "失败", type: "danger" as const };
    case 3:
      return { label: "跳过", type: "info" as const };
    default:
      return { label: "未知", type: "info" as const };
  }
});

function formatValue(value: unknown) {
  if (value == null) return "";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
</script>

<template>
  <div class="thought-step">
    <div class="thought-step__header">
      <span class="thought-step__index">#{{ step.position }}</span>
      <el-tag v-if="step.tool" size="small" type="warning">{{
        step.tool
      }}</el-tag>
      <el-tag size="small" :type="statusMeta.type">{{
        statusMeta.label
      }}</el-tag>
      <span v-if="step.latencyMs != null" class="thought-step__latency">
        {{ step.latencyMs }}ms
      </span>
    </div>
    <div v-if="step.thought" class="thought-step__text">{{ step.thought }}</div>
    <pre v-if="step.toolInput != null" class="thought-step__payload">{{
      formatValue(step.toolInput)
    }}</pre>
    <div v-if="step.observation" class="thought-step__observation">
      {{ step.observation }}
    </div>
    <el-alert
      v-if="step.status === 2 && step.error"
      type="error"
      :closable="false"
      :title="step.error"
    />
  </div>
</template>

<style scoped lang="scss">
.thought-step {
  padding: 8px 0;

  & + & {
    border-top: 1px solid var(--el-border-color-lighter);
  }

  &__header {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  &__index {
    font-size: 12px;
    font-weight: 600;
    color: var(--el-text-color-secondary);
  }

  &__latency {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__text {
    margin-top: 4px;
    font-size: 13px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }

  &__payload,
  &__observation {
    max-height: 160px;
    padding: 6px 8px;
    margin: 4px 0 0;
    overflow: auto;
    font-size: 12px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }
}
</style>
