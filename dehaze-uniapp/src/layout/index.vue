<template>
  <view class="page-layout">
    <!-- 顶部导航栏：L1 品牌+标题+搜索 / L2 返回+标题；L3 沉浸页由页面内工具栏替代 -->
    <AppNavbar
      v-if="level === 'L1' || level === 'L2'"
      :level="level"
      :title="title"
    />

    <!-- 主内容区（L1 预留原生 tabBar 高度） -->
    <view class="page-content" :class="{ 'with-tabbar': level === 'L1' }">
      <slot />
    </view>

    <!-- 底部导航采用原生 tabBar（pages.json tabBar 配置），由各端框架渲染，无需自绘 -->
  </view>
</template>

<script lang="ts" setup>
import AppNavbar from "./Navbar.vue";

interface Props {
  /**
   * 页面层级（决定导航形态）：
   * - L1：Tab 根页面（原生 TabBar + 顶部标题栏）
   * - L2：二级功能页（顶部导航栏：返回 + 标题，TabBar 隐藏）
   * - L3：深度沉浸页（无全局导航，页面内工具栏）
   */
  level?: "L1" | "L2" | "L3";
  /** 页面标题（L1 为 Tab 标题，L2 为页面功能名） */
  title?: string;
}

const props = withDefaults(defineProps<Props>(), {
  level: "L1",
  title: "",
});
</script>

<style lang="scss" scoped>
.page-layout {
  width: 100%;
  min-height: 100vh;
  background: #ffffff;
}

.page-content {
  width: 100%;

  &.with-tabbar {
    // 为底部导航栏留出空间
    padding-bottom: calc(100rpx + constant(safe-area-inset-bottom));
    padding-bottom: calc(100rpx + env(safe-area-inset-bottom));
  }
}
</style>
