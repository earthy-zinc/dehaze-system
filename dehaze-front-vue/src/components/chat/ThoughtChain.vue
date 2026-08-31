<!-- 推理过程折叠展示：按步骤显示，含子智能体并行消耗 -->
<script lang="ts" setup>
import type { ThoughtEvent } from "dehaze-sdk-js";
import { computed } from "vue";
import SubAgentPanel from "./SubAgentPanel.vue";
import ThoughtStep from "./ThoughtStep.vue";

defineOptions({ name: "ThoughtChain" });

const props = defineProps<{
  thoughts: ThoughtEvent[];
}>();

const emit = defineEmits<{
  "step-click": [position: number];
}>();

// 推理过程默认折叠，避免长思考链压过最终回复
const activeNames = computed({
  get: () => [] as string[],
  set: () => {
    // 保持默认折叠语义，展开状态由渲染帧内组件自管理
  },
});

const hasSubAgent = computed(() =>
  props.thoughts.some((item) => item.tool === "task")
);
</script>

<template>
  <div class="thought-chain">
    <el-collapse>
      <el-collapse-item :name="'chain'">
        <template #title>
          <span class="thought-chain__title"
            >推理过程（{{ thoughts.length }} 步）</span
          >
        </template>
        <SubAgentPanel v-if="hasSubAgent" :steps="thoughts" :usage="null" />
        <ThoughtStep
          v-for="step in thoughts"
          :key="step.position"
          :step="step"
          @click="emit('step-click', step.position)"
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
