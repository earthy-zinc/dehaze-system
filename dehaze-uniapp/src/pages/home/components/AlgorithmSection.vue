<template>
  <view class="algorithm-section">
    <view class="algorithm-content">
      <view class="algorithm-text">
        <text class="section-title">多算法智能选择</text>
        <text class="section-subtitle">
          支持DCP、AOD-Net、DehazeNet等多种先进算法
        </text>
        <view class="algorithm-features">
          <view
            v-for="(feature, index) in algorithmFeatures"
            :key="index"
            class="feature-item"
          >
            <u-icon name="check-circle" size="20" color="#34d399" />
            <text class="feature-text">{{ feature }}</text>
          </view>
        </view>
        <up-button
          type="primary"
          :plain="true"
          @click="handleLearnMore"
          :custom-style="learnMoreButtonStyle"
        >
          了解更多算法详情
          <u-icon
            name="arrow-right"
            size="16"
            color="#3b82f6"
            :margin-left="8"
          />
        </up-button>
      </view>
      <view class="algorithm-visual">
        <image
          src="https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/f49e4b9e-6079-4a0b-8f91-bcab5deec2c7/image_1763727581_1_3.jpg"
          mode="widthFix"
          class="algorithm-image"
        />
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { homeData } from "../data/homeData";

interface Emits {
  (e: "learn-more"): void;
}

const emit = defineEmits<Emits>();
const algorithmFeatures = ref(homeData.algorithmFeatures);

const learnMoreButtonStyle = {
  background: "#ffffff",
  color: "#3b82f6",
  border: "2rpx solid #ffffff",
  borderRadius: "10rpx",
};

const handleLearnMore = () => {
  emit("learn-more");
};
</script>

<style lang="scss" scoped>
.algorithm-section {
  padding: 80rpx 40rpx;
  background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
  color: white;
}

.algorithm-content {
  max-width: 100%;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60rpx;
  align-items: center;
}

.section-title {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 16rpx;
  letter-spacing: -0.02em;
}

.section-subtitle {
  display: block;
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  margin-bottom: 32rpx;
}

.algorithm-features {
  list-style: none;
  padding: 0;
  margin: 0 0 32rpx 0;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 12rpx;
  margin-bottom: 16rpx;
}

.feature-text {
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.95);
  line-height: 1.6;
}

.algorithm-image {
  width: 100%;
  height: auto;
  border-radius: 20rpx;
  box-shadow: 0 20rpx 60rpx rgba(0, 0, 0, 0.3);
}

@media screen and (max-width: 768rpx) {
  .algorithm-content {
    grid-template-columns: 1fr;
    gap: 40rpx;
  }

  .section-title {
    font-size: 36rpx;
  }

  .section-subtitle {
    font-size: 24rpx;
  }

  .feature-text {
    font-size: 24rpx;
  }
}
</style>
