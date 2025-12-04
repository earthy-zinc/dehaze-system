<template>
  <view class="home-page">
    <!-- Hero Section - 英雄区 -->
    <HeroSection
      @primary-click="handleStartClick"
      @secondary-click="handleDatasetClick"
    />

    <!-- 效果展示区 -->
    <ShowcaseSection />

    <!-- 核心功能区 - 工作流程 -->
    <WorkflowSection @step-click="handleStepClick" />

    <!-- 工具网格 -->
    <ToolsSection @tool-click="handleToolClick" />

    <!-- 算法优势区 -->
    <AlgorithmSection @learn-more="handleAlgorithmClick" />

    <!-- 技术特性区 -->
    <view class="tech-specs-section">
      <view class="specs-grid">
        <SpecCard
          v-for="spec in specData"
          :key="spec.title"
          :icon="spec.icon"
          :title="spec.title"
          :value="spec.value"
          :description="spec.description"
        />
      </view>
    </view>

    <!-- 最终CTA区域 -->
    <CTASection @start-click="handleStartClick" />
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import HeroSection from "./components/HeroSection.vue";
import ShowcaseSection from "./components/ShowcaseSection.vue";
import WorkflowSection from "./components/WorkflowSection.vue";
import ToolsSection from "./components/ToolsSection.vue";
import AlgorithmSection from "./components/AlgorithmSection.vue";
import CTASection from "./components/CTASection.vue";
import SpecCard from "@/components/business/SpecCard.vue";
import type { ToolItem } from "./data/homeData";
import { homeData } from "./data/homeData";

// 技术规格数据
const specData = ref(homeData.specs);

// 事件处理函数
const handleStartClick = () => {
  // 临时显示提示，待实现对应页面
  uni.showToast({
    title: "功能开发中",
    icon: "none",
  });
  // uni.navigateTo({ url: '/pages/image-input/index' })
};

const handleDatasetClick = () => {
  uni.showToast({
    title: "数据集功能开发中",
    icon: "none",
  });
  // uni.navigateTo({ url: '/pages/dataset/index' })
};

const handleStepClick = (target: string) => {
  // 根据目标跳转不同页面
  const routeMap: Record<string, string> = {
    "image-input": "图像输入功能开发中",
    "algorithm-select": "算法选择功能开发中",
    processing: "图像处理功能开发中",
  };

  const message = routeMap[target] || "功能开发中";
  uni.showToast({
    title: message,
    icon: "none",
  });

  // if (routeMap[target]) {
  //   uni.navigateTo({ url: routeMap[target] })
  // }
};

const handleToolClick = (tool: ToolItem) => {
  uni.showToast({
    title: `${tool.title}功能开发中`,
    icon: "none",
  });
  // 处理工具点击，可以跳转到对应功能页面
  console.log("Tool clicked:", tool);
};

const handleAlgorithmClick = () => {
  uni.showToast({
    title: "算法详情功能开发中",
    icon: "none",
  });
  // uni.navigateTo({ url: '/pages/algorithm/index' })
};
</script>

<style lang="scss" scoped>
.home-page {
  width: 100%;
  min-height: 100vh;
  background: #ffffff;
}

.tech-specs-section {
  padding: 80rpx 40rpx;
  background: #ffffff;
  width: 100%;

  .specs-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 32rpx;
    max-width: 100%;
    margin: 0 auto;
  }
}

@media screen and (max-width: 768rpx) {
  .tech-specs-section .specs-grid {
    grid-template-columns: 1fr;
    gap: 24rpx;
  }
}
</style>
