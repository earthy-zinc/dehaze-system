<template>
  <view class="workflow-section">
    <SectionHeader
      title="强大的功能生态"
      subtitle="从输入到输出，每一步都精心设计"
    />

    <view class="workflow-container">
      <ProcessStep
        v-for="step in workflowSteps"
        :key="step.id"
        :number="step.number"
        :title="step.title"
        :description="step.description"
        :icon="step.icon"
        @click="() => handleStepClick(step.target)"
      />

      <view v-for="index in 2" :key="`arrow-${index}`" class="workflow-arrow">
        <SvgIcon name="arrow-right" size="24" color="#d1d5db" />
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import SectionHeader from "@/components/common/SectionHeader.vue";
import ProcessStep from "@/components/common/ProcessStep.vue";
import { homeData } from "../data/homeData";

interface Emits {
  (e: "step-click", target: string): void;
}

const emit = defineEmits<Emits>();
const workflowSteps = ref(homeData.workflowSteps);

const handleStepClick = (target: string) => {
  emit("step-click", target);
};
</script>

<style lang="scss" scoped>
.workflow-section {
  padding: 80rpx 40rpx;
  background: #ffffff;
}

.workflow-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20rpx;
  margin-bottom: 80rpx;
  flex-wrap: wrap;
  max-width: 100%;
}

.workflow-arrow {
  color: #d1d5db;
  font-size: 24rpx;
  flex-shrink: 0;
}

@media screen and (max-width: 768rpx) {
  .workflow-container {
    flex-direction: column;
  }

  .workflow-arrow {
    transform: rotate(90deg);
  }
}
</style>
