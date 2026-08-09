<template>
  <view class="algorithm-section">
    <text class="section-title">多算法智能选择</text>
    <text class="section-subtitle">
      支持DCP、AOD-Net、DehazeNet等多种先进算法
    </text>
    <view class="algorithm-visual">
      <view class="algorithm-features">
        <view
          v-for="(feature, index) in algorithmFeatures"
          :key="index"
          class="feature-item"
        >
          <SvgIcon name="checkmark-circle" size="20" color="#34d399" />
          <text class="feature-text">{{ feature }}</text>
        </view>
      </view>
      <image :src="algorithmImageUrl" mode="widthFix" class="algorithm-image" />
    </view>

    <button class="learn-more-btn" @click="handleLearnMore">
      了解更多算法详情
      <SvgIcon name="arrow-right" size="16" color="#3b82f6" :margin-left="8" />
    </button>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { homeData } from "../data/homeData";

/** 数据集静态文件服务地址（来自 .env 的 VITE_DATASET_HOST） */
const DATASET_BASE_URL =
  import.meta.env.VITE_DATASET_HOST || "http://127.0.0.1:9000";

interface Emits {
  (e: "learn-more"): void;
}

const emit = defineEmits<Emits>();
const algorithmFeatures = ref(homeData.algorithmFeatures);

// 使用真实数据集图片
const algorithmImageUrl = `${DATASET_BASE_URL}/datasets/NH-HAZE-2023/clean/003.JPG`;

const handleLearnMore = () => {
  emit("learn-more");
};
</script>

<style lang="scss" scoped>
.algorithm-section {
  padding: 80rpx 40rpx;
  background: linear-gradient(135deg, #1e3a8a 0%, $color-primary 100%);
  color: white;
}

.learn-more-btn {
  background: $color-white;
  color: $color-primary;
  border: 2rpx solid $color-white;
  border-radius: 10rpx;
  padding: 8rpx 20rpx;
  font-size: 26rpx;

  &::after {
    border: none;
  }
}

.section-title {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  color: $color-white;
  margin-bottom: 16rpx;
  letter-spacing: -0.02em;
}

.section-subtitle {
  display: block;
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  margin-bottom: 64rpx;
}

.algorithm-visual {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 64rpx;
}

.algorithm-features {
  list-style: none;
  padding: 0;
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
  width: 40vw;
  height: auto;
  margin-left: 32rpx;
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
