<template>
  <div class="home-page">
    <HeroSection />
    <FeatureSection />
    <ShowcaseSection />
    <CtaSection />
  </div>
</template>

<script lang="ts" setup>
import HeroSection from "./components/HeroSection.vue";
import FeatureSection from "./components/FeatureSection.vue";
import ShowcaseSection from "./components/ShowcaseSection.vue";
import CtaSection from "./components/CtaSection.vue";

defineOptions({
  name: "Home",
});
</script>

<style lang="scss">
/* 首页品牌主色（仅作用于 .home-page 局部，不污染全局）。
 * 蓝色系在亮/暗主题下均成立，故不随 html.dark 变化；
 * 页面背景、卡片、文字等语义色统一使用项目 Element Plus 主题变量（--el-*），
 * 由项目既有暗色主题自动适配，避免复刻设计稿中不完善的暗色模式。 */
.home-page {
  --home-primary-50: #eff6ff;
  --home-primary-100: #dbeafe;
  --home-primary-200: #bfdbfe;
  --home-primary-300: #93c5fd;
  --home-primary-400: #60a5fa;
  --home-primary-500: #3b82f6;
  --home-primary-600: #2563eb;
  --home-primary-700: #1d4ed8;

  color: var(--el-text-color-primary);
  background-color: var(--el-bg-color-page);
}
</style>
