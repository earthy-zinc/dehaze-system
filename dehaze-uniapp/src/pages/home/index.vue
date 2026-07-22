<template>
  <PageLayout class="home-page">
    <view class="main-content">
      <!-- Hero Section - 英雄区 -->
      <HeroSection
        @primary-click="handleStartClick"
        @secondary-click="handleDatasetClick"
      />

      <!-- 效果展示区 -->
      <ShowcaseSection />

      <!-- 算法推荐区 -->
      <AlgorithmRecommendSection
        @select="handleRecommendSelect"
        @more="handleAlgorithmClick"
      />

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
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import HeroSection from "./components/HeroSection.vue";
import ShowcaseSection from "./components/ShowcaseSection.vue";
import WorkflowSection from "./components/WorkflowSection.vue";
import ToolsSection from "./components/ToolsSection.vue";
import AlgorithmSection from "./components/AlgorithmSection.vue";
import AlgorithmRecommendSection from "./components/AlgorithmRecommendSection.vue";
import CTASection from "./components/CTASection.vue";
import SpecCard from "@/components/business/SpecCard.vue";
import type { ToolItem } from "./data/homeData";
import { homeData } from "./data/homeData";

// 技术规格数据
const specData = ref(homeData.specs);

// 事件处理函数
const handleStartClick = () => {
  uni.navigateTo({ url: "/pages/image-input/index" });
};

const handleDatasetClick = () => {
  uni.navigateTo({ url: "/pages/dataset/index" });
};

const handleStepClick = (target: string) => {
  const routeMap: Record<string, string> = {
    "image-input": "/pages/image-input/index",
    "algorithm-select": "/pages/algorithm-select/index",
    // 处理步骤需先选图选算法，引导到图像输入
    processing: "/pages/image-input/index",
  };

  const url = routeMap[target];
  if (url) {
    uni.navigateTo({ url });
  } else {
    uni.showToast({ title: "页面开发中", icon: "none" });
  }
};

const handleToolClick = (tool: ToolItem) => {
  // 使用 tool.target 字段做路由映射
  const toolRoutes: Record<string, string> = {
    "side-by-side": "/pages/side-by-side/index",
    overlay: "/pages/overlay/index",
    magnifier: "/pages/magnifier/index",
    filter: "/pages/filter/index",
    metrics: "/pages/metrics/index",
    dataset: "/pages/dataset/index",
  };

  const url = toolRoutes[tool.target];
  if (url) {
    uni.navigateTo({ url });
  } else {
    uni.showToast({ title: `${tool.title}开发中`, icon: "none" });
  }
};

const handleAlgorithmClick = () => {
  uni.navigateTo({ url: "/pages/algorithm-select/index" });
};

/** 算法推荐：点击具体算法 → 跳转算法选择页 */
function handleRecommendSelect() {
  uni.navigateTo({ url: "/pages/algorithm-select/index" });
}
</script>

<style lang="scss" scoped>
.home-page {
  width: 100%;
  min-height: 100vh;
  background: #ffffff;
}

.main-content {
  // 为底部导航栏留出空间
  padding-bottom: calc(100rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(100rpx + env(safe-area-inset-bottom));
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
