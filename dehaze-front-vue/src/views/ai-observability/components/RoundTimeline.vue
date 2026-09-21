<!-- 轮次时序回放时间轴：轮内事件按 ts 交织排序，按 kind 分色节点渲染；旁路 trace 挂尾部次要样式 -->
<script lang="ts" setup>
import { computed } from "vue";
import type {
  AiObservabilityContextEvent,
  AiObservabilityContextItem,
  AiObservabilityTimelineEvent,
  AiObservabilityTimelineMessage,
  AiObservabilityTimelineRound,
  AiObservabilityTimelineTrace,
} from "dehaze-sdk-js";
import {
  CONTEXT_ITEM_META,
  TRACE_STATUS_META,
  fmtDuration,
  fmtTokens,
  traceTypeMeta,
} from "../format";
import LlmCallNode from "./LlmCallNode.vue";
import JsonViewer from "@/components/JsonViewer.vue";

defineOptions({ name: "RoundTimeline" });

const props = defineProps<{
  round: AiObservabilityTimelineRound;
}>();

const EVENT_KIND_META: Record<string, { label: string; color: string }> = {
  input: { label: "用户输入", color: "var(--el-color-primary)" },
  context: { label: "上下文组装", color: "#9254de" },
  llm_call: { label: "LLM 调用", color: "var(--el-color-success)" },
  tool_exec: { label: "工具执行", color: "var(--el-color-warning)" },
  system_event: { label: "系统事件", color: "var(--el-color-danger)" },
  billing: { label: "计费", color: "var(--el-text-color-secondary)" },
};

function kindMeta(kind: string) {
  return (
    EVENT_KIND_META[kind] ?? {
      label: kind,
      color: "var(--el-text-color-secondary)",
    }
  );
}

/** 主对话 trace（traceType=conversation 或缺省）；其余为旁路（摘要/记忆提取等）挂尾部 */
const mainEvents = computed(() => {
  const mainTraces = props.round.traces.filter(
    (trace) => !trace.traceType || trace.traceType === "conversation"
  );
  const events = mainTraces.flatMap((trace) => trace.events);
  // 按 ts 稳定排序：无 ts（后端无原始时序）沉底；同时刻保持业务序（seq/position）
  return events
    .map((event, index) => ({ event, index }))
    .sort((a, b) => {
      const aRank = a.event.ts ? 0 : 1;
      const bRank = b.event.ts ? 0 : 1;
      if (aRank !== bRank) return aRank - bRank;
      if (aRank === 0 && a.event.ts !== b.event.ts) {
        return a.event.ts! < b.event.ts! ? -1 : 1;
      }
      return a.index - b.index;
    })
    .map(({ event }) => event);
});

const bypassTraces = computed(() =>
  props.round.traces.filter(
    (trace) => trace.traceType && trace.traceType !== "conversation"
  )
);

function traceErrorMeta(trace: AiObservabilityTimelineTrace) {
  return TRACE_STATUS_META[trace.status];
}

// ==================== 上下文组装节点 ====================

function contextItems(event: AiObservabilityTimelineEvent) {
  return [...(event.snapshot?.items ?? [])].sort((a, b) => b.tokens - a.tokens);
}

function contextTotalTokens(event: AiObservabilityTimelineEvent) {
  return (event.snapshot?.items ?? []).reduce(
    (sum, item) => sum + item.tokens,
    0
  );
}

function itemMeta(type: string) {
  return CONTEXT_ITEM_META[type] ?? { label: type, color: "#c0c4cc" };
}

function ctxPercent(event: AiObservabilityTimelineEvent, tokens: number) {
  const total = contextTotalTokens(event);
  if (!total) return "0%";
  return `${Math.max(2, Math.round((tokens / total) * 100))}%`;
}

function ctxItemDesc(item: AiObservabilityContextItem) {
  const parts = [`${fmtTokens(item.tokens)} tokens`];
  if (item.count != null) parts.push(`${item.count} 条`);
  if (item.counts) {
    const counts = Object.entries(item.counts)
      .filter(([, count]) => count != null)
      .map(([role, count]) => `${role}:${count}`)
      .join(" / ");
    if (counts) parts.push(counts);
  }
  if (item.source) parts.push(item.source === "summarized" ? "已压缩" : "原文");
  return parts.join(" · ");
}

// ==================== 系统事件节点（护栏/计划/中断恢复） ====================

function systemEventLabel(event: AiObservabilityTimelineEvent) {
  switch (event.event) {
    case "guardrail":
      return "护栏拦截";
    case "plan":
      return "计划快照";
    case "resume":
      return "中断恢复";
    case "error":
      return "执行错误";
    default:
      return event.event ?? "系统事件";
  }
}

function ctxEventDesc(event: AiObservabilityContextEvent) {
  let label: string;
  switch (event.event) {
    case "summarize":
      label = "上下文压缩";
      break;
    case "truncate":
      label = "历史截断";
      break;
    default:
      return systemEventLabel({ kind: "system_event", event: event.event });
  }
  if (event.before_tokens == null && event.after_tokens == null) return label;
  return `${label}：${fmtTokens(event.before_tokens)} → ${fmtTokens(event.after_tokens)} tokens`;
}

function formatTs(ts?: string | null) {
  if (!ts) return "";
  return ts.length >= 19 ? ts.slice(11, 23) : ts;
}

function msgTokens(message?: AiObservabilityTimelineMessage | null) {
  if (!message) return "";
  const total = (message.inputTokens ?? 0) + (message.outputTokens ?? 0);
  return total > 0 ? fmtTokens(total) : "";
}
</script>

<template>
  <div class="round-timeline">
    <!-- 主对话事件流 -->
    <el-timeline class="round-timeline__main">
      <el-timeline-item
        v-for="(event, index) in mainEvents"
        :key="index"
        :timestamp="formatTs(event.ts)"
        placement="top"
      >
        <template #dot>
          <span
            class="round-timeline__dot"
            :style="{ background: kindMeta(event.kind).color }"
          />
        </template>

        <div class="round-timeline__kind">{{ kindMeta(event.kind).label }}</div>

        <!-- 用户输入全文 -->
        <template v-if="event.kind === 'input'">
          <pre class="round-timeline__pre">{{
            event.message?.content ?? "-"
          }}</pre>
        </template>

        <!-- 上下文组装：构成项占比 + 压缩事件 + 系统提示/摘要/记忆全文 -->
        <template v-else-if="event.kind === 'context'">
          <div
            v-for="item in contextItems(event)"
            :key="item.type"
            class="ctx-item"
          >
            <div class="ctx-item__label">
              <span>{{ itemMeta(item.type).label }}</span>
              <span class="ctx-item__desc">{{ ctxItemDesc(item) }}</span>
            </div>
            <div class="ctx-item__bar">
              <div
                class="ctx-item__bar-fill"
                :style="{
                  width: ctxPercent(event, item.tokens),
                  background: itemMeta(item.type).color,
                }"
              />
            </div>
            <el-collapse v-if="item.content" class="round-timeline__collapse">
              <el-collapse-item
                :title="
                  item.type === 'system'
                    ? '系统提示正文'
                    : item.type === 'summary'
                      ? '会话摘要原文'
                      : '正文'
                "
                name="content"
              >
                <pre class="round-timeline__pre">{{ item.content }}</pre>
              </el-collapse-item>
            </el-collapse>
            <el-collapse
              v-if="item.items?.length"
              class="round-timeline__collapse"
            >
              <el-collapse-item title="注入记忆原文" name="memory">
                <div
                  v-for="(mem, memIndex) in item.items"
                  :key="memIndex"
                  class="round-timeline__memory"
                >
                  <el-tag v-if="mem.memory_type" size="small">{{
                    mem.memory_type
                  }}</el-tag>
                  <el-tag v-if="mem.source" size="small" type="info">{{
                    mem.source
                  }}</el-tag>
                  <pre class="round-timeline__pre">{{ mem.content }}</pre>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
          <div
            v-for="(ctxEvent, eventIndex) in event.snapshot?.events ?? []"
            :key="eventIndex"
            class="ctx-event"
          >
            <el-tag type="warning" size="small">{{
              ctxEventDesc(ctxEvent)
            }}</el-tag>
            <el-collapse
              v-if="
                ctxEvent.event === 'guardrail' &&
                (ctxEvent.rule || ctxEvent.detail)
              "
              class="round-timeline__collapse"
            >
              <el-collapse-item :title="ctxEvent.rule ?? '护栏详情'" name="d">
                <pre class="round-timeline__pre">{{
                  ctxEvent.detail ?? ctxEvent.rule
                }}</pre>
              </el-collapse-item>
            </el-collapse>
            <pre
              v-else-if="
                ctxEvent.event === 'plan' &&
                (ctxEvent.phase || ctxEvent.plan_summary)
              "
              class="round-timeline__pre"
              >{{
                [
                  ctxEvent.phase ? `阶段：${ctxEvent.phase}` : "",
                  ctxEvent.plan_summary ?? "",
                ]
                  .filter(Boolean)
                  .join("\n")
              }}</pre>
            <div
              v-else-if="ctxEvent.event === 'resume'"
              class="ctx-event__meta"
            >
              <template v-if="ctxEvent.interrupt_type"
                >类型：{{ ctxEvent.interrupt_type }}</template
              >
              <template v-if="ctxEvent.decision"
                >· 决策：{{ ctxEvent.decision }}</template
              >
              <template v-if="ctxEvent.from_trace_id"
                >· 原链路：{{ ctxEvent.from_trace_id }}</template
              >
            </div>
          </div>
        </template>

        <!-- LLM 调用：摘要卡 + raw 报文回放 -->
        <template v-else-if="event.kind === 'llm_call'">
          <LlmCallNode :event="event" />
        </template>

        <!-- 工具执行：思考/完整入参出参/耗时/子Agent 归属 -->
        <template v-else-if="event.kind === 'tool_exec'">
          <div class="round-timeline__tool-header">
            <el-tag size="small" type="warning">{{
              event.tool ?? "工具"
            }}</el-tag>
            <el-tag
              v-if="event.isSubagent || event.agentCode"
              size="small"
              type="info"
            >
              {{ event.agentCode ? `子Agent: ${event.agentCode}` : "子Agent" }}
            </el-tag>
            <span v-if="event.position != null" class="ctx-event__meta"
              >步骤 {{ event.position }}</span
            >
            <span class="ctx-event__meta"
              >耗时 {{ fmtDuration(event.latencyMs) }}</span
            >
          </div>
          <pre v-if="event.thought" class="round-timeline__pre">{{
            event.thought
          }}</pre>
          <div class="round-timeline__field">入参</div>
          <JsonViewer
            :data="event.toolInput"
            filename="tool-input.json"
            empty-text="无入参数据"
          />
          <div class="round-timeline__field">返回</div>
          <pre v-if="event.observation" class="round-timeline__pre">{{
            event.observation
          }}</pre>
          <div v-else class="ctx-event__meta">无返回数据</div>
        </template>

        <!-- 系统事件：护栏/计划/中断 -->
        <template v-else-if="event.kind === 'system_event'">
          <el-tag type="danger" size="small">{{
            systemEventLabel(event)
          }}</el-tag>
          <JsonViewer
            v-if="event.detail != null"
            :data="event.detail"
            filename="system-event.json"
            empty-text="无详情数据"
            class="mt-1"
          />
        </template>

        <!-- 计费 -->
        <template v-else-if="event.kind === 'billing'">
          <div class="round-timeline__tool-header">
            <el-tag size="small">{{ event.billType ?? "计费" }}</el-tag>
            <span class="ctx-event__meta">积分 {{ event.credits ?? "-" }}</span>
            <span v-if="event.tokens" class="ctx-event__meta">
              Token {{ fmtTokens(event.tokens.input) }}/{{
                fmtTokens(event.tokens.output)
              }}
              <template v-if="event.tokens.cached">
                （缓存
                {{ fmtTokens(event.tokens.cached) }}）</template
              >
            </span>
          </div>
        </template>
      </el-timeline-item>

      <!-- 助手输出（轮次收尾） -->
      <el-timeline-item
        :timestamp="formatTs(props.round.assistantMessage?.createTime)"
        placement="top"
      >
        <template #dot>
          <span
            class="round-timeline__dot"
            style="background: var(--el-color-primary)"
          />
        </template>
        <div class="round-timeline__kind">助手输出</div>
        <pre class="round-timeline__pre">{{
          props.round.assistantMessage?.content ?? "（无回复）"
        }}</pre>
      </el-timeline-item>
    </el-timeline>

    <!-- 旁路 trace：摘要压缩/记忆提取等，挂轮次尾部次要样式 -->
    <div v-if="bypassTraces.length" class="round-timeline__bypass">
      <div class="round-timeline__bypass-title">旁路调用</div>
      <el-collapse class="round-timeline__collapse">
        <el-collapse-item
          v-for="trace in bypassTraces"
          :key="trace.traceId"
          :name="trace.traceId"
        >
          <template #title>
            <el-tag
              :type="traceTypeMeta(trace.traceType)?.tag ?? 'info'"
              size="small"
            >
              {{ traceTypeMeta(trace.traceType)?.label ?? trace.traceType }}
            </el-tag>
            <el-tag :type="traceErrorMeta(trace).tag" size="small" class="ml-1">
              {{ traceErrorMeta(trace).label }}
            </el-tag>
            <span class="font-mono text-xs ml-1">{{ trace.traceId }}</span>
          </template>
          <div class="round-timeline__bypass-events">
            <div
              v-for="(event, index) in trace.events"
              :key="index"
              class="round-timeline__bypass-event"
            >
              <span class="ctx-event__meta"
                >{{ formatTs(event.ts) }} ·
                {{ kindMeta(event.kind).label }}</span
              >
              <LlmCallNode v-if="event.kind === 'llm_call'" :event="event" />
              <pre
                v-else-if="event.kind === 'input'"
                class="round-timeline__pre"
                >{{ event.message?.content ?? "-" }}</pre>
              <pre
                v-else-if="event.kind === 'billing'"
                class="round-timeline__pre"
                >{{
                  `计费 ${event.billType ?? "-"} · 积分 ${event.credits ?? "-"}`
                }}</pre>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>

    <el-empty
      v-if="!mainEvents.length && !bypassTraces.length"
      description="该轮次无事件数据"
      :image-size="60"
    />
  </div>
</template>

<style scoped lang="scss">
.round-timeline {
  &__dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  &__kind {
    margin-bottom: 4px;
    font-size: 12px;
    font-weight: 600;
    color: var(--el-text-color-secondary);
  }

  &__pre {
    max-height: 200px;
    padding: 6px 8px;
    margin: 4px 0;
    overflow: auto;
    font-size: 12px;
    line-height: 1.6;
    word-break: break-all;
    white-space: pre-wrap;
    background: var(--el-fill-color-lighter);
    border-radius: 6px;
  }

  &__field {
    margin: 8px 0 2px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__collapse {
    margin-top: 4px;
    border: none;

    :deep(.el-collapse-item__header) {
      height: 28px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }

    :deep(.el-collapse-item__wrap) {
      background: transparent;
      border-bottom: none;
    }

    :deep(.el-collapse-item__content) {
      padding-bottom: 4px;
    }
  }

  &__tool-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }

  &__memory {
    padding: 4px 0;

    & + & {
      border-top: 1px solid var(--el-border-color-lighter);
    }
  }

  &__bypass {
    padding-top: 8px;
    margin-top: 16px;
    border-top: 1px dashed var(--el-border-color);

    &-title {
      margin-bottom: 4px;
      font-size: 12px;
      font-weight: 600;
      color: var(--el-text-color-secondary);
    }

    &-event {
      padding: 4px 0;
    }
  }
}

.ctx-item {
  margin-bottom: 8px;

  &__label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
    font-size: 12px;
    color: var(--el-text-color-primary);
  }

  &__desc {
    color: var(--el-text-color-secondary);
  }

  &__bar {
    height: 8px;
    overflow: hidden;
    background: var(--el-fill-color-light);
    border-radius: 4px;

    .ctx-item__bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s;
    }
  }
}

.ctx-event {
  margin-top: 6px;

  &__meta {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
