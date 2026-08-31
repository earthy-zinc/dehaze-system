<!-- 过程链详情抽屉：trace 汇总 + 消息记录 + 推理步骤 + 上下文快照构成 + LLM 调用按 seq 回放 -->
<template>
  <el-drawer v-model="store.detailVisible" title="过程链详情" size="760px">
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
              class="content-collapse error-detail-collapse"
            >
              <el-collapse-item title="异常详情" name="detail">
                <pre
                  v-if="detail.errorDetail.message"
                  class="call-output"
                  >{{ detail.errorDetail.message }}</pre
                >
                <pre
                  v-if="detail.errorDetail.stack"
                  class="call-output error-stack"
                  >{{ detail.errorDetail.stack }}</pre
                >
              </el-collapse-item>
            </el-collapse>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 计费明细：该 trace 关联的计费账单 -->
        <el-collapse
          v-if="detail.billing?.length"
          class="content-collapse billing-collapse"
        >
          <el-collapse-item
            :title="`计费明细（${detail.billing.length} 笔）`"
            name="billing"
          >
            <div class="msg-list">
              <div
                v-for="(bill, index) in detail.billing"
                :key="index"
                class="msg-item"
              >
                <div class="msg-header">
                  <el-tag size="small">{{ bill.billType ?? "计费" }}</el-tag>
                  <span class="font-mono text-xs">{{
                    bill.model ?? "-"
                  }}</span>
                  <span
                    v-if="bill.actualModel && bill.actualModel !== bill.model"
                    class="msg-meta"
                    >实际 {{ bill.actualModel }}</span
                  >
                  <span class="msg-meta"
                    >积分 {{ bill.credits ?? "-"
                    }}<template v-if="bill.creditsSaved"
                      >（省 {{ bill.creditsSaved }}）</template
                    ></span
                  >
                </div>
                <div class="bill-meta">
                  Token {{ fmtTokens(bill.inputTokens) }}/{{
                    fmtTokens(bill.outputTokens)
                  }}<template v-if="bill.cachedInputTokens"
                    >（缓存 {{ fmtTokens(bill.cachedInputTokens) }}）</template
                  >
                  · 耗时 {{ fmtDuration(bill.latencyMs) }}
                  <template v-if="bill.providerId != null">
                    · 供应商 {{ bill.providerId }}</template
                  >
                  <template v-if="bill.createTime"> · {{ bill.createTime }}</template>
                </div>
                <div v-if="bill.requestId || bill.errorCode" class="bill-meta">
                  <template v-if="bill.requestId">
                    请求 {{ bill.requestId }}</template
                  >
                  <template v-if="bill.errorCode">
                    · 错误码 {{ bill.errorCode }}</template
                  >
                </div>
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>

        <!-- 上下文快照：AI 当次回复"看到了什么" -->
        <h4 class="section-title">上下文快照</h4>
        <template v-if="contextItems.length">
          <div v-for="item in contextItems" :key="item.type" class="ctx-item">
            <div class="ctx-label">
              <span class="ctx-name">{{ itemMeta(item.type).label }}</span>
              <span class="ctx-desc">{{ ctxItemDesc(item) }}</span>
            </div>
            <div class="ctx-bar">
              <div
                class="ctx-bar-fill"
                :style="{
                  width: ctxPercent(item.tokens),
                  background: itemMeta(item.type).color,
                }"
              />
            </div>
            <el-collapse
              v-if="item.type === 'system' && item.content"
              class="content-collapse"
            >
              <el-collapse-item title="系统提示正文" name="content">
                <pre class="call-output">{{ item.content }}</pre>
              </el-collapse-item>
            </el-collapse>
            <el-collapse
              v-if="item.type === 'summary' && item.content"
              class="content-collapse"
            >
              <el-collapse-item title="会话摘要原文" name="content">
                <pre class="call-output">{{ item.content }}</pre>
              </el-collapse-item>
            </el-collapse>
            <el-collapse
              v-if="item.type === 'memory' && item.items?.length"
              class="content-collapse"
            >
              <el-collapse-item title="注入记忆原文" name="content">
                <div class="msg-list">
                  <div
                    v-for="(mem, index) in item.items"
                    :key="index"
                    class="msg-item"
                  >
                    <div class="msg-header">
                      <el-tag v-if="mem.memory_type" size="small">{{
                        mem.memory_type
                      }}</el-tag>
                      <el-tag v-if="mem.source" size="small" type="info">{{
                        mem.source
                      }}</el-tag>
                    </div>
                    <pre class="call-output">{{ mem.content }}</pre>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
          <div v-if="contextEvents.length" class="ctx-events">
            <div
              v-for="(event, index) in contextEvents"
              :key="index"
              class="ctx-event"
            >
              <el-tag type="warning" size="small">{{ eventDesc(event) }}</el-tag>
              <!-- 护栏命中：规则与拦截详情可折叠查看 -->
              <el-collapse
                v-if="
                  event.event === 'guardrail' && (event.rule || event.detail)
                "
                class="content-collapse"
              >
                <el-collapse-item :title="event.rule ?? '护栏详情'" name="d">
                  <pre class="call-output">{{ event.detail ?? event.rule }}</pre>
                </el-collapse-item>
              </el-collapse>
              <pre
                v-else-if="event.event === 'plan' && (event.phase || event.plan_summary)"
                class="call-output event-pre"
                >{{
                  [
                    event.phase ? `阶段：${event.phase}` : "",
                    event.plan_summary ?? "",
                  ]
                    .filter(Boolean)
                    .join("\n")
                }}</pre
              >
              <div
                v-else-if="event.event === 'resume'"
                class="event-meta"
              >
                <template v-if="event.interrupt_type"
                  >类型：{{ event.interrupt_type }}</template
                >
                <template v-if="event.decision">
                  · 决策：{{ event.decision }}</template
                >
                <template v-if="event.from_trace_id">
                  · 原链路：{{ event.from_trace_id }}</template
                >
              </div>
            </div>
          </div>
        </template>
        <el-empty v-else description="无上下文快照" :image-size="60" />

        <!-- 消息记录：该会话完整消息（系统提示/用户输入/助手输出/工具返回） -->
        <h4 class="section-title">
          消息记录（{{ messages.length }} 条）
        </h4>
        <el-empty
          v-if="!messages.length"
          description="无消息记录"
          :image-size="60"
        />
        <div v-else class="msg-list">
          <div v-for="(msg, index) in messages" :key="index" class="msg-item">
            <div class="msg-header">
              <el-tag size="small" :type="roleMeta(msg.role).tag">
                {{ roleMeta(msg.role).label }}
              </el-tag>
              <span v-if="msg.model" class="msg-meta">{{ msg.model }}</span>
              <span
                v-if="traceStatusLabel(msg.status)"
                class="msg-meta"
              >
                {{ traceStatusLabel(msg.status) }}
              </span>
            </div>
            <pre class="call-output">{{ msg.content }}</pre>
          </div>
        </div>

        <!-- 推理步骤：中间推理过程 -->
        <h4 class="section-title">
          推理步骤（{{ thoughts.length }} 步）
        </h4>
        <el-empty
          v-if="!thoughts.length"
          description="无推理步骤"
          :image-size="60"
        />
        <template v-else>
          <template v-for="step in thoughts" :key="step.position">
            <div v-if="step.isSubagent || step.agentCode" class="subagent-tag">
              <el-tag size="small" type="warning">
                {{ step.agentCode ? `子Agent: ${step.agentCode}` : "子Agent" }}
              </el-tag>
            </div>
            <ThoughtStep :step="step" />
          </template>
        </template>

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

        <!-- LLM 调用回放：AI 每一步做了什么 -->
        <h4 class="section-title">LLM 调用回放（{{ llmCalls.length }} 次）</h4>
        <el-empty
          v-if="!llmCalls.length"
          description="无 LLM 调用记录"
          :image-size="60"
        />
        <el-timeline v-else>
          <el-timeline-item
            v-for="call in llmCalls"
            :key="call.seq"
            :type="CALL_STATUS_META[call.status].tag"
          >
            <div class="call-header">
              <span class="call-seq">#{{ call.seq }}</span>
              <el-tag :type="CALL_STATUS_META[call.status].tag" size="small">
                {{ CALL_STATUS_META[call.status].label }}
              </el-tag>
              <span class="call-meta">{{ call.model ?? "-" }}</span>
              <span v-if="call.stepPosition != null" class="call-meta"
                >步骤 {{ call.stepPosition }}</span
              >
              <span class="call-meta"
                >耗时 {{ fmtDuration(call.durationMs) }}</span
              >
              <span v-if="call.firstTokenMs != null" class="call-meta">
                首Token {{ fmtDuration(call.firstTokenMs) }}
              </span>
              <span class="call-meta">
                Token {{ fmtTokens(call.promptTokens) }}/{{
                  fmtTokens(call.completionTokens)
                }}<template v-if="call.cachedTokens > 0"
                  >（缓存 {{ fmtTokens(call.cachedTokens) }}）</template
                >
              </span>
            </div>
            <div v-if="call.errorType" class="call-error">
              <el-tag type="danger" size="small">{{ call.errorType }}</el-tag>
            </div>
            <el-collapse
              v-if="call.attempts?.length"
              class="content-collapse"
            >
              <el-collapse-item
                :title="`物理调用尝试（${call.attempts.length} 次）`"
                name="attempts"
              >
                <div class="msg-list">
                  <div
                    v-for="(attempt, index) in call.attempts"
                    :key="index"
                    class="msg-item"
                  >
                    <div class="msg-header">
                      <el-tag
                        :type="attemptStatusMeta(attempt.status).tag"
                        size="small"
                      >
                        {{ attemptStatusMeta(attempt.status).label }}
                      </el-tag>
                      <span v-if="attempt.provider_id != null" class="msg-meta"
                        >供应商 {{ attempt.provider_id }}</span
                      >
                      <span v-if="attempt.key_id != null" class="msg-meta"
                        >Key {{ attempt.key_id }}</span
                      >
                      <span v-if="attempt.model" class="font-mono text-xs">{{
                        attempt.model
                      }}</span>
                      <span v-if="attempt.latency_ms != null" class="msg-meta"
                        >耗时 {{ fmtDuration(attempt.latency_ms) }}</span
                      >
                    </div>
                    <div v-if="attempt.error_code" class="msg-meta">
                      错误码 {{ attempt.error_code }}
                    </div>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
            <div v-if="inputDesc(call)" class="call-block">
              输入：{{ inputDesc(call) }}
            </div>
            <el-collapse
              v-if="call.inputSnapshot?.system_content"
              class="content-collapse"
            >
              <el-collapse-item title="系统提示" name="system">
                <pre class="call-output">{{
                  call.inputSnapshot.system_content
                }}</pre>
              </el-collapse-item>
            </el-collapse>
            <el-collapse
              v-if="call.inputSnapshot?.tools?.length"
              class="content-collapse"
            >
              <el-collapse-item title="工具定义" name="tools">
                <div class="msg-list">
                  <div
                    v-for="(tool, index) in call.inputSnapshot.tools"
                    :key="index"
                    class="msg-item"
                  >
                    <div class="msg-header">
                      <span class="font-mono text-xs">{{
                        tool.name
                      }}</span>
                    </div>
                    <pre class="call-output">{{ tool.description }}</pre>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
            <el-collapse
              v-if="call.inputSnapshot?.messages?.items?.length"
              class="content-collapse"
            >
              <el-collapse-item title="本轮输入消息" name="messages">
                <div class="msg-list">
                  <div
                    v-for="(msg, index) in call.inputSnapshot.messages.items"
                    :key="index"
                    class="msg-item"
                  >
                    <div class="msg-header">
                      <el-tag
                        v-if="msg.role"
                        size="small"
                        :type="roleMeta(msg.role).tag"
                      >
                        {{ roleMeta(msg.role).label }}
                      </el-tag>
                    </div>
                    <pre class="call-output">{{ msg.content }}</pre>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
            <div v-if="call.toolCall?.tools?.length" class="call-block">
              工具调用：{{
                call.toolCall.tools.map((tool) => tool.name).join("、")
              }}
            </div>
            <pre
              v-if="call.outputSnapshot?.tool_calls?.length"
              class="call-output"
              >{{
                call.outputSnapshot.tool_calls
                  .map((toolCall) => `${toolCall.name}(${toolCall.arguments})`)
                  .join("\n")
              }}</pre>
            <pre v-else-if="call.outputSnapshot?.text" class="call-output">{{
              call.outputSnapshot.text
            }}</pre>
          </el-timeline-item>
        </el-timeline>
      </template>
    </div>
  </el-drawer>
</template>

<script lang="ts" setup>
import type {
  AiObservabilityContextEvent,
  AiObservabilityContextItem,
  AiObservabilityLlmCall,
  AiObservabilityStatus,
  AiObservabilityThought,
  AiObservabilityTraceDetail,
  AiObservabilityTraceMessage,
} from "dehaze-sdk-js";
import {
  ATTEMPT_STATUS_META,
  CALL_STATUS_META,
  CONTEXT_ITEM_META,
  MESSAGE_ROLE_META,
  TRACE_STATUS_META,
  fmtDuration,
  fmtTokens,
  traceTypeMeta,
} from "../format";
import ThoughtStep from "@/components/chat/ThoughtStep.vue";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "TraceDetailDrawer" });

const store = useAdminObservabilityStore();

const detail = computed(() => store.traceDetail);
const llmCalls = computed(() => detail.value?.llmCalls ?? []);
const messages = computed(() => detail.value?.messages ?? []);
const thoughts = computed(() => detail.value?.thoughts ?? []);

function roleMeta(role: string) {
  return MESSAGE_ROLE_META[role] ?? { label: role, tag: "info" as const };
}

function attemptStatusMeta(status: number) {
  return ATTEMPT_STATUS_META[status] ?? { label: `未知(${status})`, tag: "info" as const };
}

function traceStatusLabel(status?: number) {
  if (status == null) return undefined;
  return TRACE_STATUS_META[status as AiObservabilityStatus]?.label;
}

// ==================== 上下文快照 ====================

const contextItems = computed(() =>
  [...(detail.value?.contextSnapshot?.items ?? [])].sort(
    (a, b) => b.tokens - a.tokens
  )
);

const contextTotalTokens = computed(() =>
  contextItems.value.reduce((sum, item) => sum + item.tokens, 0)
);

const contextEvents = computed(
  () => detail.value?.contextSnapshot?.events ?? []
);

function itemMeta(type: string) {
  return CONTEXT_ITEM_META[type] ?? { label: type, color: "#c0c4cc" };
}

function ctxPercent(tokens: number) {
  if (!contextTotalTokens.value) return "0%";
  return `${Math.max(2, Math.round((tokens / contextTotalTokens.value) * 100))}%`;
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

function eventDesc(event: AiObservabilityContextEvent) {
  let label: string;
  switch (event.event) {
    case "summarize":
      label = "上下文压缩";
      break;
    case "truncate":
      label = "历史截断";
      break;
    case "guardrail":
      label = "护栏拦截";
      break;
    case "plan":
      label = "计划快照";
      break;
    case "resume":
      label = "中断恢复";
      break;
    default:
      label = event.event;
  }
  // guardrail/plan/resume 的载荷在标签下方单独展示
  if (
    event.event === "guardrail" ||
    event.event === "plan" ||
    event.event === "resume"
  ) {
    return label;
  }
  if (event.before_tokens == null && event.after_tokens == null) return label;
  return `${label}：${fmtTokens(event.before_tokens)} → ${fmtTokens(event.after_tokens)} tokens`;
}

// ==================== LLM 调用回放 ====================

function inputDesc(call: AiObservabilityLlmCall) {
  const input = call.inputSnapshot;
  if (!input) return "";
  const counts = Object.entries(input.messages?.counts ?? {})
    .filter(([, count]) => count != null)
    .map(([role, count]) => `${role}:${count}`)
    .join(" / ");
  const parts = [
    `消息 ${counts || "0"}`,
    `${fmtTokens(input.messages?.tokens)} tokens`,
  ];
  if (input.tool_count) parts.push(`工具 ${input.tool_count} 个`);
  if (input.system_tokens)
    parts.push(`系统提示 ${fmtTokens(input.system_tokens)} tokens`);
  return parts.join(" · ");
}
</script>

<style lang="scss" scoped>
.section-title {
  margin: 20px 0 12px;
  font-size: 15px;
  font-weight: 600;
}

.ctx-item {
  margin-bottom: 8px;

  .ctx-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2px;
    font-size: 12px;

    .ctx-name {
      color: var(--el-text-color-primary);
    }

    .ctx-desc {
      color: var(--el-text-color-secondary);
    }
  }

  .ctx-bar {
    height: 8px;
    overflow: hidden;
    background: var(--el-fill-color-light);
    border-radius: 4px;

    .ctx-bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s;
    }
  }
}

.ctx-events {
  margin-top: 12px;

  .ctx-event + .ctx-event {
    margin-top: 6px;
  }

  .event-pre {
    margin-top: 4px;
  }

  .event-meta {
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.billing-collapse {
  margin-top: 12px;

  :deep(.el-collapse-item__header) {
    font-size: 13px;
    font-weight: 600;
  }
}

.error-detail-collapse {
  display: inline-block;
  width: 100%;
  margin-top: 4px;
}

.error-stack {
  max-height: 240px;
}

.subagent-tag {
  padding-top: 8px;
}

.call-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;

  .call-seq {
    font-weight: 600;
  }

  .call-meta {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.call-error {
  margin-top: 6px;
}

.trace-type-tag {
  margin-left: 6px;
}

.call-block {
  margin-top: 6px;
  font-size: 12px;
  color: var(--el-text-color-regular);
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

.content-collapse {
  margin-top: 6px;
  border: none;

  :deep(.el-collapse-item__header) {
    height: 28px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  :deep(.el-collapse-item__wrap) {
    border-bottom: none;
    background: transparent;
  }

  :deep(.el-collapse-item__content) {
    padding-bottom: 4px;
  }
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
