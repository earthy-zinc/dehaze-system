<template>
  <view class="section-header">
    <text class="section-title">{{ title }}</text>
    <text class="section-subtitle">{{ subtitle }}</text>
  </view>
</template>

<script lang="ts" setup>
interface Props {
  title: string;
  subtitle: string;
}

defineProps<Props>();
</script>

<style lang="scss" scoped>
.section-header {
  text-align: center;
  margin-bottom: 60rpx;
}

.section-title {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 16rpx;
  letter-spacing: -0.02em;
}

.section-subtitle {
  display: block;
  font-size: 28rpx;
  color: #6b7280;
  line-height: 1.6;
}
</style>
