<!-- 推理过程折叠展示：按步骤显示，含子智能体并行执行用量（纯展示，props 进） -->
<script lang="ts" setup>
import { computed } from "vue";
import { sortStepsByPosition } from "../vm/steps";
import SubAgentPanel from "./SubAgentPanel.vue";
import ThoughtStep from "./ThoughtStep.vue";
import type { ChatSubAgentUsageVM, ChatThoughtStepVM } from "../types";

defineOptions({ name: "ThoughtChain" });

const props = defineProps<{
  thoughts: ChatThoughtStepVM[];
  /** 子智能体粒度用量（message.end usage.subAgents；缺省时子智能体面板不渲染） */
  subAgents?: ChatSubAgentUsageVM[];
}>();

const steps = computed(() => sortStepsByPosition(props.thoughts));
</script>

<template>
  <div class="thought-chain">
    <el-collapse>
      <el-collapse-item :name="'chain'">
        <template #title>
          <span class="thought-chain__title"
            >推理过程（{{ steps.length }} 步）</span
          >
        </template>
        <SubAgentPanel :steps="steps" :sub-agents="subAgents" />
        <ThoughtStep v-for="step in steps" :key="step.position" :step="step" />
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<style scoped lang="scss">
.thought-chain {
  margin-bottom: 8px;

  :deep(.el-collapse-item__header) {
    height: 32px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__title {
    font-size: 13px;
  }
}
</style>
