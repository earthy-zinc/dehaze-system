<!-- 链路追踪面板：单条消息完整推理链（推理步骤/LLM 调用回放/中断与失败标注/最终回复） -->
<script lang="ts" setup>
import { computed } from "vue";
import { storeToRefs } from "pinia";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import ThoughtStep from "@/components/chat/ThoughtStep.vue";
import {
  useAdminAuditStore,
  type AuditLlmCall,
} from "@/store/modules/adminAudit";
import ToolCallEntry from "./ToolCallEntry.vue";

defineOptions({ name: "ChainTracePanel" });

const adminAuditStore = useAdminAuditStore();
const { traceMessage, traceChainData, traceLoading, traceError } =
  storeToRefs(adminAuditStore);

function callStatusMeta(status: number) {
  switch (status) {
    case 1:
      return { label: "成功", type: "success" as const };
    case 2:
      return { label: "失败", type: "danger" as const };
    case 3:
      return { label: "超时", type: "warning" as const };
    default:
      return { label: "未知", type: "info" as const };
  }
}

const totalDurationMs = computed(() =>
  (traceChainData.value?.llmCalls ?? []).reduce(
    (sum, call) => sum + call.durationMs,
    0
  )
);

// 无过程链数据（thoughts 与 llmCalls 均为空）展示空态
const noTraceData = computed(
  () =>
    !traceLoading.value &&
    !traceError.value &&
    !!traceChainData.value &&
    traceChainData.value.thoughts.length === 0 &&
    traceChainData.value.llmCalls.length === 0
);

function toolEntries(call: AuditLlmCall) {
  return (call.toolCall?.tools ?? []).map((tool) => {
    let args: unknown = tool.arguments;
    try {
      args = JSON.parse(tool.arguments);
    } catch {
      // 非法 JSON 保留原始字符串
    }
    return { name: tool.name, args };
  });
}
</script>

<template>
  <div class="chain-trace">
    <el-divider content-position="left">
      <span class="chain-trace__title">
        链路追踪<span v-if="traceMessage">（消息 #{{ traceMessage.id }}）</span>
      </span>
      <el-button link size="small" @click="adminAuditStore.closeChainTrace()">
        收起
      </el-button>
    </el-divider>

    <div v-loading="traceLoading">
      <el-alert
        v-if="traceError"
        type="error"
        :closable="false"
        :title="traceError"
      />
      <template v-else-if="traceMessage">
        <el-alert
          v-if="traceMessage.status === 3"
          type="error"
          :closable="false"
          :title="traceMessage.error || '该消息推理失败'"
          class="mb-2"
        />
        <el-alert
          v-else-if="traceMessage.status === 4"
          type="warning"
          :closable="false"
          title="该消息推理被中断/取消"
          class="mb-2"
        />

        <el-descriptions :column="4" size="small" border class="mb-3">
          <el-descriptions-item label="Trace ID">
            {{ traceChainData?.traceId ?? "-" }}
          </el-descriptions-item>
          <el-descriptions-item label="模型">
            {{ traceMessage.model ?? "-" }}
          </el-descriptions-item>
          <el-descriptions-item label="LLM 调用">
            {{ traceChainData?.llmCalls.length ?? 0 }} 次
          </el-descriptions-item>
          <el-descriptions-item label="调用总耗时">
            {{ totalDurationMs }}ms
          </el-descriptions-item>
        </el-descriptions>

        <div
          v-if="traceChainData?.thoughts.length"
          class="chain-trace__section"
        >
          <div class="chain-trace__section-title">推理步骤</div>
          <ThoughtStep
            v-for="step in traceChainData.thoughts"
            :key="step.position"
            :step="step"
          />
        </div>

        <div
          v-if="traceChainData?.llmCalls.length"
          class="chain-trace__section"
        >
          <div class="chain-trace__section-title">LLM 调用回放</div>
          <div
            v-for="call in traceChainData.llmCalls"
            :key="call.seq"
            class="chain-trace__call"
          >
            <div class="chain-trace__call-header">
              <span class="chain-trace__call-seq">#{{ call.seq }}</span>
              <el-tag v-if="call.stepPosition != null" size="small">
                步骤 {{ call.stepPosition }}
              </el-tag>
              <span>{{ call.model ?? "-" }}</span>
              <el-tag size="small" :type="callStatusMeta(call.status).type">
                {{ callStatusMeta(call.status).label }}
              </el-tag>
              <el-tag v-if="call.errorType" size="small" type="danger">
                {{ call.errorType }}
              </el-tag>
              <span class="chain-trace__call-meta">
                {{ call.promptTokens }}+{{ call.completionTokens }} tokens ·
                {{ call.durationMs }}ms
              </span>
            </div>
            <ToolCallEntry
              v-for="(tool, index) in toolEntries(call)"
              :key="index"
              :name="tool.name"
              :args="tool.args"
              :result="call.outputSnapshot?.text"
              :latency-ms="call.firstTokenMs ?? undefined"
            />
          </div>
        </div>

        <el-collapse v-if="traceChainData?.contextSnapshot">
          <el-collapse-item title="上下文快照" name="context">
            <pre class="chain-trace__json">{{
              JSON.stringify(traceChainData.contextSnapshot, null, 2)
            }}</pre>
          </el-collapse-item>
        </el-collapse>

        <el-empty
          v-if="noTraceData"
          description="该消息暂无过程链数据"
          :image-size="60"
        />

        <div v-if="traceMessage.content" class="chain-trace__section">
          <div class="chain-trace__section-title">最终回复</div>
          <MarkdownRenderer :content="traceMessage.content" />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped lang="scss">
.chain-trace {
  &__title {
    font-weight: 600;
  }

  &__section {
    margin-bottom: 16px;
  }

  &__section-title {
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--el-text-color-secondary);
  }

  &__call {
    padding: 8px 0;

    & + & {
      border-top: 1px solid var(--el-border-color-lighter);
    }
  }

  &__call-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 4px;
    font-size: 12px;
  }

  &__call-seq {
    font-weight: 600;
    color: var(--el-text-color-secondary);
  }

  &__call-meta {
    color: var(--el-text-color-secondary);
  }

  &__json {
    max-height: 240px;
    padding: 6px 8px;
    margin: 0;
    overflow: auto;
    font-size: 12px;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }
}
</style>
