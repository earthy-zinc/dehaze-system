<!-- LLM 调用节点：默认摘要卡（模型/耗时/TTFT/token/工具调用/物理尝试），展开回放 wire 原始报文 -->
<script lang="ts" setup>
import { computed } from "vue";
import type {
  AiObservabilityCallStatus,
  AiObservabilityTimelineEvent,
} from "dehaze-sdk-js";
import { CALL_STATUS_META, fmtDuration, fmtTokens } from "../format";
import JsonViewer from "@/components/JsonViewer.vue";

defineOptions({ name: "LlmCallNode" });

const props = defineProps<{
  event: AiObservabilityTimelineEvent;
}>();

function statusMeta(status?: AiObservabilityCallStatus) {
  return (
    (status != null ? CALL_STATUS_META[status] : undefined) ?? {
      label: "未知",
      tag: "info" as const,
    }
  );
}

const inputDesc = computed(() => {
  const input = props.event.summary?.inputSnapshot;
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
  return parts.join(" · ");
});

const outputPreview = computed(() => {
  const output = props.event.summary?.outputSnapshot;
  if (output?.tool_calls?.length) {
    return output.tool_calls
      .map((toolCall) => `${toolCall.name}(${toolCall.arguments})`)
      .join("\n");
  }
  return output?.text ?? "";
});
</script>

<template>
  <div class="llm-call-node">
    <div class="llm-call-node__header">
      <span class="llm-call-node__seq">#{{ event.seq ?? "-" }}</span>
      <el-tag size="small" :type="statusMeta(event.status).tag">
        {{ statusMeta(event.status).label }}
      </el-tag>
      <span class="llm-call-node__meta">{{ event.model ?? "-" }}</span>
      <span class="llm-call-node__meta"
        >耗时 {{ fmtDuration(event.durationMs) }}</span
      >
      <span v-if="event.firstTokenMs != null" class="llm-call-node__meta">
        首Token {{ fmtDuration(event.firstTokenMs) }}
      </span>
      <span class="llm-call-node__meta">
        Token {{ fmtTokens(event.promptTokens) }}/{{
          fmtTokens(event.completionTokens)
        }}<template v-if="(event.cachedTokens ?? 0) > 0">
          （缓存 {{ fmtTokens(event.cachedTokens) }}）</template
        >
      </span>
    </div>

    <div v-if="event.errorType" class="llm-call-node__error">
      <el-tag type="danger" size="small">{{ event.errorType }}</el-tag>
    </div>

    <div v-if="event.toolCall?.tools?.length" class="llm-call-node__tool">
      工具调用：{{ event.toolCall.tools.map((tool) => tool.name).join("、") }}
    </div>
    <div v-if="inputDesc" class="llm-call-node__input">
      输入：{{ inputDesc }}
    </div>
    <pre v-if="outputPreview" class="llm-call-node__output">{{
      outputPreview
    }}</pre>

    <el-collapse v-if="event.attempts?.length" class="llm-call-node__collapse">
      <el-collapse-item
        :title="`物理调用尝试（${event.attempts.length} 次）`"
        name="attempts"
      >
        <div
          v-for="(attempt, index) in event.attempts"
          :key="index"
          class="llm-call-node__attempt"
        >
          <el-tag
            size="small"
            :type="attempt.status === 1 ? 'success' : 'danger'"
          >
            {{ attempt.status === 1 ? "成功" : "失败" }}
          </el-tag>
          <span v-if="attempt.provider_id != null" class="llm-call-node__meta">
            供应商 {{ attempt.provider_id }}
          </span>
          <span v-if="attempt.key_id != null" class="llm-call-node__meta">
            Key {{ attempt.key_id }}
          </span>
          <span v-if="attempt.model" class="llm-call-node__meta">{{
            attempt.model
          }}</span>
          <span v-if="attempt.latency_ms != null" class="llm-call-node__meta">
            耗时 {{ fmtDuration(attempt.latency_ms) }}
          </span>
          <span v-if="attempt.error_code" class="llm-call-node__meta">
            错误码 {{ attempt.error_code }}
          </span>
        </div>
      </el-collapse-item>
    </el-collapse>

    <!-- wire 原始报文回放：raw 为空时 JsonViewer 显示空态提示 -->
    <el-collapse class="llm-call-node__collapse">
      <el-collapse-item title="查看完整请求 JSON" name="request">
        <JsonViewer :data="event.rawRequest" filename="raw-request.json" />
      </el-collapse-item>
      <el-collapse-item title="查看完整响应 JSON" name="response">
        <JsonViewer :data="event.rawResponse" filename="raw-response.json" />
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<style scoped lang="scss">
.llm-call-node {
  &__header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;

    .llm-call-node__seq {
      font-weight: 600;
    }
  }

  &__meta {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__error,
  &__tool,
  &__input {
    margin-top: 6px;
    font-size: 12px;
    color: var(--el-text-color-regular);
  }

  &__output {
    max-height: 120px;
    padding: 6px 8px;
    margin: 6px 0 0;
    overflow: auto;
    font-size: 12px;
    line-height: 1.6;
    word-break: break-all;
    white-space: pre-wrap;
    background: var(--el-fill-color-lighter);
    border-radius: 6px;
  }

  &__collapse {
    margin-top: 6px;
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

  &__attempt {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 4px 0;
    font-size: 12px;

    & + & {
      border-top: 1px solid var(--el-border-color-lighter);
    }
  }
}
</style>
