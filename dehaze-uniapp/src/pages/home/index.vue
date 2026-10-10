<template>
  <PageLayout level="L1" title="首页" :is-home="true" class="home-page">
    <view class="main-content">
      <!-- Hero Section - 英雄区 -->
      <HeroSection
        :model-count="displayModelCount"
        @primary-click="handleStartClick"
        @secondary-click="handleDatasetClick"
      />

      <!-- 效果展示区 -->
      <ShowcaseSection />

      <!-- 模型推荐区 -->
      <ModelRecommendSection
        @select="handleRecommendSelect"
        @more="handleModelClick"
      />

      <!-- 核心功能区 - 工作流程 -->
      <WorkflowSection @step-click="handleStepClick" />

      <!-- 工具网格 -->
      <ToolsSection @tool-click="handleToolClick" />

      <!-- 模型优势区 -->
      <ModelSection @learn-more="handleModelClick" />

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
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import HeroSection from "./components/HeroSection.vue";
import ShowcaseSection from "./components/ShowcaseSection.vue";
import WorkflowSection from "./components/WorkflowSection.vue";
import ToolsSection from "./components/ToolsSection.vue";
import ModelSection from "./components/ModelSection.vue";
import ModelRecommendSection from "./components/ModelRecommendSection.vue";
import CTASection from "./components/CTASection.vue";
import SpecCard from "@/components/business/SpecCard.vue";
import type { ToolItem } from "./data/homeData";
import { homeData } from "./data/homeData";
import { AiModelAPI } from "dehaze-sdk-js";

// 技术规格数据
const specData = ref(homeData.specs);

// 模型数量（动态获取）
const modelCount = ref(0);
const modelLoading = ref(true);

/** Hero 区展示用模型数量：加载中显示 "..."，完成后显示实际数量 */
const displayModelCount = computed(() =>
  modelLoading.value ? "..." : modelCount.value
);

onMounted(async () => {
  try {
    // 普通用户可见范围是启用列表（listModels 为管理端接口）
    const list = await AiModelAPI.listEnabledModels("image_edit");
    modelCount.value = list.length;
    const modelSpec = specData.value.find((s) => s.title === "智能模型");
    if (modelSpec) modelSpec.value = `${modelCount.value}+`;
  } catch {
    // 获取失败时保持空值，不影响页面其他部分
  } finally {
    modelLoading.value = false;
  }
});

// 事件处理函数
/** CTA「开始去雾」→ 跳转去雾 Tab */
const handleStartClick = () => {
  uni.switchTab({ url: "/pages/dehaze/index" });
};

const handleDatasetClick = () => {
  uni.navigateTo({ url: "/pages/dataset/index" });
};

const handleStepClick = (target: string) => {
  const routeMap: Record<string, string> = {
    "image-input": "/pages/image-input/index",
    // 处理步骤需先选图，引导到图像输入
    processing: "/pages/image-input/index",
  };

  const url = routeMap[target];
  if (url) {
    uni.navigateTo({ url });
  } else {
    uni.showToast({ title: "页面开发中", icon: "none" });
  }
};

/** 工具网格跳转：指向 L2 功能页（image-input/dataset/metrics） */
const handleToolClick = (tool: ToolItem) => {
  const toolRoutes: Record<string, string> = {
    "image-input": "/pages/image-input/index",
    dataset: "/pages/dataset/index",
    metrics: "/pages/metrics/index",
  };

  const url = toolRoutes[tool.target];
  if (url) {
    uni.navigateTo({ url });
  } else {
    uni.showToast({ title: `${tool.title}开发中`, icon: "none" });
  }
};

/** 模型推荐/模型优势区入口 → 去雾 Tab（模型在选择页内联选择） */
const handleModelClick = () => {
  uni.switchTab({ url: "/pages/dehaze/index" });
};

function handleRecommendSelect() {
  uni.switchTab({ url: "/pages/dehaze/index" });
}
</script>

<style lang="scss" scoped>
.home-page {
  width: 100%;
  min-height: 100vh;
  background: $color-white;
}

.tech-specs-section {
  // 块级元素默认占满父宽，无需 width:100%；若同时写 width:100% + padding，
  // 在默认 content-box 下总宽会超出视口导致右侧被截断
  padding: 80rpx 40rpx;
  background: $color-white;

  .specs-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: $spacing-lg;
    max-width: 100%;
    margin: 0 auto;
  }
}

@media screen and (max-width: 768rpx) {
  .tech-specs-section .specs-grid {
    grid-template-columns: 1fr;
    gap: $spacing-md;
  }
}
</style>
