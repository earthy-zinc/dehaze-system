<!-- 推理过程折叠展示：按步骤显示，含子智能体并行执行用量（纯展示，props 进） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { SubAgentUsage } from "clarity-sdk-js";
import type { ChatThoughtStepVM } from "../types";
import SubAgentPanel from "./SubAgentPanel.vue";
import ThoughtStep from "./ThoughtStep.vue";

defineOptions({ name: "ThoughtChain" });

const props = defineProps<{
  /** 工具步骤（stepsOfBlocks 已按 seq 升序） */
  steps: ChatThoughtStepVM[];
  /** 子智能体粒度用量（message.end usage.subAgents；缺省时子智能体面板不渲染） */
  subAgents?: SubAgentUsage[];
}>();

const failedCount = computed(
  () => props.steps.filter((step) => step.status === "failed").length
);
</script>

<template>
  <div class="thought-chain">
    <el-collapse>
      <el-collapse-item :name="'chain'">
        <template #title>
          <span class="thought-chain__title"
            >查看过程（{{ steps.length }} 步）</span
          >
          <el-tag v-if="failedCount" type="danger" size="small" class="ml-2">
            {{ failedCount }} 步失败
          </el-tag>
        </template>
        <SubAgentPanel :steps="steps" :sub-agents="subAgents" />
        <ThoughtStep
          v-for="(step, index) in steps"
          :key="`${step.position}-${index}`"
          :step="step"
        />
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
