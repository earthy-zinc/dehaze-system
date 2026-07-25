<template>
  <view class="page-header-card" :class="`variant-${variant}`">
    <view class="header-icon" :style="{ background: iconBg }">
      <u-icon :name="icon" size="28" :color="iconColor" />
    </view>
    <view class="header-text">
      <text class="header-title">{{ title }}</text>
      <text v-if="subtitle" class="header-subtitle">{{ subtitle }}</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
/**
 * 页面顶部标题卡片
 * 抽取自多个页面共用的 .page-header-card 模式，统一图标 / 标题 / 副标题结构
 */
withDefaults(
  defineProps<{
    /** u-icon 的图标名 */
    icon: string;
    /** u-icon 的颜色 */
    iconColor: string;
    /** 图标容器背景，支持单色或 CSS 渐变 */
    iconBg: string;
    /** 标题文本 */
    title: string;
    /** 副标题文本（可选） */
    subtitle?: string;
    /** 卡片视觉变体：default 适合浅色背景，dark 适合深色背景 */
    variant?: "default" | "dark";
  }>(),
  {
    subtitle: "",
    variant: "default",
  }
);
</script>

<style lang="scss" scoped>
.page-header-card {
  display: flex;
  align-items: center;
  gap: $spacing-md;
  padding: $spacing-lg;
  margin-bottom: $spacing-md;
  border-radius: $radius-xl;

  &.variant-default {
    background: $color-white;
    box-shadow: $shadow-md;
  }

  &.variant-dark {
    background: rgba(255, 255, 255, 0.95);
  }
}

.header-icon {
  flex-shrink: 0;
  width: 80rpx;
  height: 80rpx;
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.header-text {
  flex: 1;
  min-width: 0;
}

.header-title {
  display: block;
  font-size: $font-xl;
  font-weight: 700;
  color: $color-text-primary;
  margin-bottom: $spacing-xs;
}

.header-subtitle {
  display: block;
  font-size: $font-sm;
  color: $color-text-secondary;
}
</style>
