<template>
  <view class="final-cta-section">
    <text class="cta-title">准备好体验专业级图像去雾了吗？</text>
    <text class="cta-subtitle">立即开始，让您的图像重获清晰</text>
    <up-button
      type="primary"
      size="large"
      @click="handleStartClick"
      :custom-style="ctaButtonStyle"
    >
      开始使用
      <SvgIcon name="arrow-right" size="18" color="#ffffff" :margin-left="12" />
    </up-button>
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Emits {
  (e: "start-click"): void;
}

const emit = defineEmits<Emits>();

const ctaButtonStyle = {
  background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
  border: "none",
  borderRadius: "14rpx",
  padding: "20rpx 60rpx",
  fontSize: "20rpx",
  fontWeight: "700",
  boxShadow: "0 8rpx 32rpx rgba(59, 130, 246, 0.3)",
};

const handleStartClick = () => {
  emit("start-click");
};
</script>

<style lang="scss" scoped>
.final-cta-section {
  padding: 100rpx 40rpx;
  text-align: center;
  background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
}

.cta-title {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 16rpx;
  letter-spacing: -0.02em;
}

.cta-subtitle {
  display: block;
  font-size: 32rpx;
  color: #6b7280;
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
