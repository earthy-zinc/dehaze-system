<!-- 过程链详情抽屉：trace 汇总 + 计费明细 + 单轮时间线回放（复用 RoundTimeline，同一数据形状） -->
<script lang="ts" setup>
import { computed } from "vue";
import type {
  AiObservabilityTimelineEvent,
  AiObservabilityTimelineRound,
  AiObservabilityTraceDetail,
} from "dehaze-sdk-js";
import {
  TRACE_STATUS_META,
  fmtDuration,
  fmtTokens,
  traceTypeMeta,
} from "../format";
import RoundTimeline from "./RoundTimeline.vue";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "TraceDetailDrawer" });

const store = useAdminObservabilityStore();

const detail = computed(() => store.traceDetail);

/**
 * trace 详情 → 单轮时间线（RoundTimeline 数据形状）：
 * 事件按 ts 来源排序——llm_call 用 startTime、thought/billing 用 createTime、context 用 trace.createTime
 */
function buildRound(
  d: AiObservabilityTraceDetail
): AiObservabilityTimelineRound {
  const messages = d.messages ?? [];
  const assistantMessage =
    messages.find((message) => message.id === d.messageId) ??
    messages.find((message) => message.role === "assistant") ??
    null;
  const userMessage =
    (assistantMessage &&
      messages.find(
        (message) =>
          message.id === assistantMessage.parentMessageId &&
          message.role === "user"
      )) ??
    messages.find((message) => message.role === "user") ??
    null;

  const events: AiObservabilityTimelineEvent[] = [];
  if (userMessage) {
    events.push({
      kind: "input",
      ts: userMessage.createTime,
      message: {
        id: userMessage.id,
        role: userMessage.role,
        content: userMessage.content,
        createTime: userMessage.createTime,
      },
    });
  }
  if (d.contextSnapshot) {
    events.push({
      kind: "context",
      ts: d.createTime,
      snapshot: d.contextSnapshot,
    });
  }
  for (const call of d.llmCalls ?? []) {
    events.push({
      kind: "llm_call",
      ts: call.startTime ?? call.createTime,
      seq: call.seq,
      model: call.model,
      status: call.status,
      errorType: call.errorType,
      durationMs: call.durationMs,
      firstTokenMs: call.firstTokenMs,
      promptTokens: call.promptTokens,
      completionTokens: call.completionTokens,
      cachedTokens: call.cachedTokens,
      toolCall: call.toolCall,
      rawRequest: call.rawRequest ?? null,
      rawResponse: call.rawResponse ?? null,
      summary: {
        inputSnapshot: call.inputSnapshot ?? null,
        outputSnapshot: call.outputSnapshot ?? null,
      },
      attempts: call.attempts ?? null,
    });
  }
  for (const thought of d.thoughts ?? []) {
    events.push({
      kind: "tool_exec",
      ts: thought.createTime,
      position: thought.position,
      tool: thought.tool,
      thought: thought.thought,
      toolInput: thought.toolInput,
      observation: thought.observation,
      latencyMs: thought.latencyMs,
      agentCode: thought.agentCode,
      isSubagent: thought.isSubagent,
    });
  }
  for (const bill of d.billing ?? []) {
    events.push({
      kind: "billing",
      ts: bill.createTime,
      billType: bill.billType,
      credits: bill.credits,
      tokens: {
        input: bill.inputTokens,
        output: bill.outputTokens,
        cached: bill.cachedInputTokens,
      },
    });
  }
  return {
    userMessage,
    assistantMessage,
    traces: [
      {
        traceId: d.traceId,
        traceType: d.traceType,
        status: d.status,
        errorType: d.errorType,
        errorDetail: d.errorDetail,
        model: d.model,
        durationMs: d.durationMs,
        createTime: d.createTime,
        events,
      },
    ],
  };
}

const round = computed(() => (detail.value ? buildRound(detail.value) : null));
</script>

<template>
  <el-drawer v-model="store.detailVisible" title="过程链详情" size="820px">
    <div v-loading="store.detailLoading">
      <el-empty
        v-if="store.detailNotFound"
        description="轨迹不存在（已被清理或无权访问）"
      />
      <template v-else-if="detail">
        <!-- trace 汇总 -->
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="Trace ID">
            <span class="font-mono text-xs">{{ detail.traceId }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="TRACE_STATUS_META[detail.status].tag" size="small">
              {{ TRACE_STATUS_META[detail.status].label }}
            </el-tag>
            <el-tag
              v-if="traceTypeMeta(detail.traceType)"
              :type="traceTypeMeta(detail.traceType)?.tag"
              size="small"
              class="trace-type-tag"
            >
              {{ traceTypeMeta(detail.traceType)?.label }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="会话ID">{{
            detail.conversationId
          }}</el-descriptions-item>
          <el-descriptions-item label="消息ID">{{
            detail.messageId ?? "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="模型">{{
            detail.model ?? "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="智能体">{{
            detail.agentCode ?? "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="总耗时">{{
            fmtDuration(detail.durationMs)
          }}</el-descriptions-item>
          <el-descriptions-item label="首Token">{{
            fmtDuration(detail.firstTokenMs)
          }}</el-descriptions-item>
          <el-descriptions-item label="LLM调用"
            >{{ detail.llmCallCount }} 次</el-descriptions-item
          >
          <el-descriptions-item label="推理步数">{{
            detail.stepCount
          }}</el-descriptions-item>
          <el-descriptions-item label="Token" :span="2">
            总 {{ fmtTokens(detail.totalTokens) }}（输入
            {{ fmtTokens(detail.promptTokens) }} / 输出
            {{ fmtTokens(detail.completionTokens) }} / 缓存
            {{ fmtTokens(detail.cachedTokens) }}）
          </el-descriptions-item>
          <el-descriptions-item label="创建时间" :span="2">{{
            detail.createTime ?? "-"
          }}</el-descriptions-item>
          <el-descriptions-item
            v-if="detail.errorType"
            label="失败类型"
            :span="2"
          >
            <el-tag type="danger" size="small">{{ detail.errorType }}</el-tag>
            <el-collapse
              v-if="detail.errorDetail?.message || detail.errorDetail?.stack"
              class="error-detail-collapse"
            >
              <el-collapse-item title="异常详情" name="detail">
                <pre v-if="detail.errorDetail.message" class="call-output">{{
                  detail.errorDetail.message
                }}</pre>
                <pre
                  v-if="detail.errorDetail.stack"
                  class="call-output error-stack"
                  >{{ detail.errorDetail.stack }}</pre>
              </el-collapse-item>
            </el-collapse>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 单轮时间线回放：输入/上下文/LLM 调用（raw 报文）/工具/计费/助手输出 -->
        <h4 class="section-title">时间线回放</h4>
        <RoundTimeline v-if="round" :round="round" />

        <!-- 中间产物：推理过程中留存的可回溯产物 -->
        <template v-if="detail.artifacts?.length">
          <h4 class="section-title">
            中间产物（{{ detail.artifacts.length }} 项）
          </h4>
          <div class="msg-list">
            <div
              v-for="artifact in detail.artifacts"
              :key="artifact.id"
              class="msg-item"
            >
              <div class="msg-header">
                <el-tag size="small" type="info">{{
                  artifact.type ?? "未知类型"
                }}</el-tag>
                <span v-if="artifact.refType" class="msg-meta">
                  引用 {{ artifact.refType }}#{{ artifact.refId ?? "-" }}
                </span>
              </div>
              <pre v-if="artifact.summary" class="call-output">{{
                artifact.summary
              }}</pre>
            </div>
          </div>
        </template>
      </template>
    </div>
  </el-drawer>
</template>

<style lang="scss" scoped>
.section-title {
  margin: 20px 0 12px;
  font-size: 15px;
  font-weight: 600;
}

.trace-type-tag {
  margin-left: 6px;
}

.error-detail-collapse {
  display: inline-block;
  width: 100%;
  margin-top: 4px;
}

.error-stack {
  max-height: 240px;
}

.call-output {
  max-height: 160px;
  padding: 8px;
  margin: 6px 0 0;
  overflow: auto;
  font-size: 12px;
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
  background: var(--el-fill-color-lighter);
  border-radius: 6px;
}

.msg-list {
  .msg-item {
    padding: 8px 0;

    & + & {
      border-top: 1px solid var(--el-border-color-lighter);
    }
  }

  .msg-header {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .msg-meta {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}
</style>
