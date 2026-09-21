<!-- 子智能体并行执行面板：并行子智能体步骤数 + 各子智能体 Token/积分消耗（纯展示，props 进）。
     子智能体粒度用量来自 message.end usage.subAgents，仅存在子智能体调用时由后端下发；
     无该数据时不渲染（避免把主 Agent 汇总用量误标为子智能体消耗，也不产生空壳）。 -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatSubAgentUsageVM, ChatThoughtStepVM } from "../types";

defineOptions({ name: "SubAgentPanel" });

const props = defineProps<{
  steps: ChatThoughtStepVM[];
  /** 子智能体粒度用量（后端不下发时缺省/为空，面板不渲染） */
  subAgents?: ChatSubAgentUsageVM[];
}>();

const stepCount = computed(() => props.steps.length);
const subAgents = computed(() => props.subAgents ?? []);
</script>

<template>
  <div v-if="subAgents.length > 0" class="sub-agent-panel">
    <div class="sub-agent-panel__title">
      子智能体并行执行（{{ stepCount }} 步）
    </div>
    <div class="sub-agent-panel__list">
      <div
        v-for="agent in subAgents"
        :key="agent.agentCode"
        class="sub-agent-panel__item"
      >
        <span class="sub-agent-panel__agent">{{ agent.agentCode }}</span>
        <span class="sub-agent-panel__metric">
          输入 {{ agent.inputTokens }} / 输出 {{ agent.outputTokens
          }}<template v-if="agent.cachedInputTokens">
            / 缓存 {{ agent.cachedInputTokens }}</template
          >
          · 积分 {{ agent.credits }}
        </span>
      </div>
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

  &__list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
  }

  &__item {
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__agent {
    font-weight: 600;
  }
}
</style>
