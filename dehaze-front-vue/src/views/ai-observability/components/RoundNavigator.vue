<!-- 轮次导航：左栏轮次列表（用户消息摘要/状态徽标/耗时/token/调用次数），异常轮显著标注 -->
<script lang="ts" setup>
import { computed } from "vue";
import type {
  AiObservabilityStatus,
  AiObservabilityTimelineRound,
} from "dehaze-sdk-js";
import type { TagType } from "@/enums/TagType";
import { fmtTokens, TRACE_STATUS_META } from "../format";

defineOptions({ name: "RoundNavigator" });

const props = defineProps<{
  rounds: AiObservabilityTimelineRound[];
  activeIndex: number;
}>();

const emit = defineEmits<{
  select: [index: number];
}>();

const SUMMARY_MAX_LEN = 30;

function summary(text?: string) {
  const content = (text ?? "").replace(/\s+/g, " ").trim();
  if (!content) return "（空）";
  return content.length > SUMMARY_MAX_LEN
    ? `${content.slice(0, SUMMARY_MAX_LEN)}…`
    : content;
}

interface RoundMeta {
  status: AiObservabilityStatus;
  totalDurationMs: number;
  tokens: number;
  llmCallCount: number;
  hasAnomaly: boolean;
}

function roundMeta(round: AiObservabilityTimelineRound): RoundMeta {
  const statuses = round.traces.map((trace) => trace.status);
  // 轮次状态取最差 trace 状态：失败 > 中断 > 超时 > 成功
  let status: AiObservabilityStatus = 1;
  for (const traceStatus of statuses) {
    if (traceStatus === 2) {
      status = 2;
      break;
    }
    if (traceStatus === 3 || traceStatus === 4) status = traceStatus;
  }
  let totalDurationMs = 0;
  let llmCallCount = 0;
  let hasAnomaly = false;
  for (const trace of round.traces) {
    if (trace.errorDetail) hasAnomaly = true;
    for (const event of trace.events) {
      if (event.kind === "llm_call") {
        llmCallCount += 1;
        totalDurationMs += event.durationMs ?? 0;
        if (event.errorType || event.status === 2) hasAnomaly = true;
      }
    }
  }
  const tokens =
    (round.assistantMessage?.inputTokens ?? 0) +
    (round.assistantMessage?.outputTokens ?? 0);
  return { status, totalDurationMs, tokens, llmCallCount, hasAnomaly };
}

const metas = computed(() => props.rounds.map(roundMeta));

function statusMeta(status: AiObservabilityStatus) {
  return TRACE_STATUS_META[status] ?? { label: "未知", tag: "info" as TagType };
}
</script>

<template>
  <div class="round-navigator">
    <div
      v-for="(round, index) in rounds"
      :key="round.userMessage?.id ?? index"
      class="round-navigator__item"
      :class="{ 'is-active': index === activeIndex }"
      @click="emit('select', index)"
    >
      <div class="round-navigator__header">
        <span class="round-navigator__index">轮次 {{ index + 1 }}</span>
        <el-tag
          :type="metas[index].status === 1 ? 'success' : 'danger'"
          size="small"
        >
          {{ statusMeta(metas[index].status).label }}
        </el-tag>
        <el-tag
          v-if="metas[index].hasAnomaly"
          type="danger"
          size="small"
          effect="plain"
        >
          异常
        </el-tag>
      </div>
      <div class="round-navigator__summary">
        {{ summary(round.userMessage?.content) }}
      </div>
      <div class="round-navigator__stats">
        <span>{{ (metas[index].totalDurationMs / 1000).toFixed(1) }}s</span>
        <span>{{ fmtTokens(metas[index].tokens) }} tok</span>
        <span>{{ metas[index].llmCallCount }} 次调用</span>
      </div>
    </div>
    <el-empty v-if="!rounds.length" description="无轮次数据" :image-size="60" />
  </div>
</template>

<style scoped lang="scss">
.round-navigator {
  display: flex;
  flex-direction: column;
  gap: 8px;

  &__item {
    padding: 8px 10px;
    cursor: pointer;
    border: 1px solid var(--el-border-color-lighter);
    border-radius: 6px;
    transition: background-color 0.2s;

    &:hover {
      background-color: var(--el-fill-color-light);
    }

    &.is-active {
      background-color: var(--el-color-primary-light-9);
      border-color: var(--el-color-primary-light-5);
    }
  }

  &__header {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  &__index {
    font-size: 13px;
    font-weight: 600;
  }

  &__summary {
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-regular);
  }

  &__stats {
    display: flex;
    gap: 10px;
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
