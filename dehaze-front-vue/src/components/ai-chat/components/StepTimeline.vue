<!-- 推理步骤时间线：按 kind 分色节点，失败/中断步骤显著标注；不透出 raw 报文（纯展示） -->
<script lang="ts" setup>
import type { ChatTraceStepVM } from "../types";

defineOptions({ name: "StepTimeline" });

defineProps<{
  steps: ChatTraceStepVM[];
}>();

const emit = defineEmits<{
  select: [step: ChatTraceStepVM];
}>();

const kindLabels: Record<ChatTraceStepVM["kind"], string> = {
  input: "用户输入",
  context: "上下文组装",
  llm_call: "LLM 调用",
  tool_exec: "工具执行",
  system_event: "系统事件",
  billing: "计费",
};

const statusLabels: Record<ChatTraceStepVM["status"], string> = {
  ok: "正常",
  failed: "失败",
  running: "进行中",
};

function formatLatency(ms?: number) {
  if (ms == null) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}
</script>

<template>
  <ol class="step-timeline">
    <li
      v-for="step in steps"
      :key="step.id"
      class="step-timeline__item"
      :class="[`step-timeline__item--${step.kind}`, `is-${step.status}`]"
      @click="emit('select', step)"
    >
      <span class="step-timeline__dot" />
      <div class="step-timeline__header">
        <el-tag size="small" effect="plain">{{ kindLabels[step.kind] }}</el-tag>
        <span class="step-timeline__label">{{ step.label }}</span>
        <el-tag
          v-if="step.status !== 'ok'"
          size="small"
          :type="step.status === 'failed' ? 'danger' : 'warning'"
        >
          {{ statusLabels[step.status] }}
        </el-tag>
        <span v-if="step.latencyMs != null" class="step-timeline__latency">
          {{ formatLatency(step.latencyMs) }}
        </span>
      </div>
      <div v-if="step.detail" class="step-timeline__detail">
        {{ step.detail }}
      </div>
      <div v-if="step.error" class="step-timeline__error">{{ step.error }}</div>
    </li>
  </ol>
</template>

<style scoped lang="scss">
.step-timeline {
  padding: 0;
  margin: 0;
  list-style: none;

  &__item {
    position: relative;
    padding: 6px 0 6px 18px;
    font-size: 13px;

    &::before {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 5px;
      width: 1px;
      content: "";
      background-color: var(--el-border-color-lighter);
    }

    &:first-child::before {
      top: 12px;
    }

    &:last-child::before {
      bottom: calc(100% - 14px);
    }

    &.is-failed {
      color: var(--el-color-danger);

      .step-timeline__label {
        font-weight: 600;
      }
    }
  }

  &__dot {
    position: absolute;
    top: 12px;
    left: 1px;
    width: 9px;
    height: 9px;
    background-color: var(--el-color-primary);
    border-radius: 50%;
  }

  &__item--llm_call &__dot {
    background-color: var(--el-color-success);
  }

  &__item--tool_exec &__dot {
    background-color: var(--el-color-warning);
  }

  &__item--billing &__dot {
    background-color: var(--el-color-info);
  }

  &__item.is-failed &__dot {
    background-color: var(--el-color-danger);
  }

  &__header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }

  &__label {
    color: var(--el-text-color-primary);
  }

  &__latency {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__detail {
    margin-top: 2px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
    overflow-wrap: anywhere;
  }

  &__error {
    margin-top: 2px;
    font-size: 12px;
    color: var(--el-color-danger);
    overflow-wrap: anywhere;
  }
}
</style>
