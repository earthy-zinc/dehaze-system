<template>
  <view class="final-cta-section">
    <text class="cta-title">准备好体验专业级图像去雾了吗？</text>
    <text class="cta-subtitle">立即开始，让您的图像重获清晰</text>
    <button class="cta-btn" @click="handleStartClick">
      开始使用
      <SvgIcon name="arrow-right" size="18" color="#ffffff" :margin-left="12" />
    </button>
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Emits {
  (e: "start-click"): void;
}

const emit = defineEmits<Emits>();

const handleStartClick = () => {
  emit("start-click");
};
</script>

<style lang="scss" scoped>
.final-cta-section {
  padding: 100rpx 40rpx;
  text-align: center;
  background: linear-gradient(180deg, $color-white 0%, $color-bg-primary 100%);
}

.cta-btn {
  background: linear-gradient(
    135deg,
    $color-primary 0%,
    $color-primary-dark 100%
  );
  border: none;
  border-radius: 14rpx;
  padding: 20rpx 60rpx;
  font-size: 20rpx;
  font-weight: 700;
  box-shadow: 0 8rpx 32rpx rgba(59, 130, 246, 0.3);

  &::after {
    border: none;
  }
}

.cta-title {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  color: $color-text-primary;
  margin-bottom: 16rpx;
  letter-spacing: -0.02em;
}

.cta-subtitle {
  display: block;
  font-size: 32rpx;
  color: $color-text-secondary;
  margin-bottom: 48rpx;
}

@media screen and (max-width: 768rpx) {
  .cta-title {
    font-size: 36rpx;
  }

  .cta-subtitle {
    font-size: 28rpx;
  }
}
</style>
