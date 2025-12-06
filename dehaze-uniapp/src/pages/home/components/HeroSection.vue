<template>
  <view class="hero-section">
    <view class="hero-content">
      <text class="hero-title">图像去雾</text>
      <text class="hero-subtitle">专业级图像处理系统</text>
      <text class="hero-description">
        采用先进的深度学习算法，一键还原清晰视界
        从图像输入到效果评估的完整闭环体验
      </text>
      <view class="hero-cta">
        <up-button
          type="primary"
          size="large"
          @click="handlePrimaryClick"
          :custom-style="primaryButtonStyle"
        >
          立即开始
          <u-icon
            name="arrow-right"
            size="16"
            color="#ffffff"
            :margin-left="8"
          />
        </up-button>
        <up-button
          type="info"
          size="large"
          :plain="true"
          @click="handleSecondaryClick"
          :custom-style="secondaryButtonStyle"
        >
          浏览数据集
        </up-button>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
interface Emits {
  (e: "primary-click"): void;
  (e: "secondary-click"): void;
}

const emit = defineEmits<Emits>();

const primaryButtonStyle = {
  background: "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
  border: "none",
  borderRadius: "12rpx",
  boxShadow: "0 4rpx 16rpx rgba(59, 130, 246, 0.3)",
};

const secondaryButtonStyle = {
  background: "#ffffff",
  color: "#3b82f6",
  border: "2rpx solid #e5e7eb",
  borderRadius: "12rpx",
};

const handlePrimaryClick = () => {
  emit("primary-click");
};

const handleSecondaryClick = () => {
  emit("secondary-click");
};
</script>

<style lang="scss" scoped>
.hero-section {
  padding: 120rpx 0 80rpx;
  text-align: center;
  width: 100%;
  background: #ffffff;
}

.hero-title {
  display: block;
  font-size: 64rpx;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: 12rpx;
  line-height: 1.1;
}

.hero-subtitle {
  display: block;
  font-size: 40rpx;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 32rpx;
  letter-spacing: -0.01em;
}

.hero-description {
  display: block;
  font-size: 28rpx;
  color: #6b7280;
  line-height: 1.6;
  max-width: 600rpx;
  margin: 0 auto 48rpx;
  white-space: pre-line;
}

.hero-cta {
  padding: 0 82rpx;
  display: flex;
  gap: 24rpx;
  justify-content: center;
  flex-wrap: wrap;
}

@media screen and (max-width: 768rpx) {
  .hero-title {
    font-size: 48rpx;
  }

  .hero-subtitle {
    font-size: 32rpx;
  }

  .hero-description {
    font-size: 24rpx;
  }

  .hero-cta {
    flex-direction: column;
    align-items: center;
  }
}
</style>
