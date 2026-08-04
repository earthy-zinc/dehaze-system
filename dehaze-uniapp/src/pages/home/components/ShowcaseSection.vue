<template>
  <view class="showcase-section">
    <SectionHeader
      title="一键去雾，效果显著"
      subtitle="智能算法自动识别雾霾程度，精准还原图像细节"
    />
    <view class="comparison-showcase">
      <view class="comparison-pair">
        <view class="comparison-item">
          <image :src="hazyUrl" mode="widthFix" class="showcase-image" />
          <view class="image-label label-hazy">去雾前</view>
        </view>
        <view class="comparison-item">
          <image :src="cleanUrl" mode="widthFix" class="showcase-image" />
          <view class="image-label label-clean">去雾后</view>
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import SectionHeader from "@/components/common/SectionHeader.vue";

/** 数据集静态文件服务地址（来自 .env 的 VITE_DATASET_HOST） */
const DATASET_BASE_URL =
  import.meta.env.VITE_DATASET_HOST || "http://127.0.0.1:9000";

// 使用真实数据集图片（NH-HAZE-2023 样例）
const hazyUrl = `${DATASET_BASE_URL}/datasets/NH-HAZE-2023/hazy/001.JPG`;
const cleanUrl = `${DATASET_BASE_URL}/datasets/NH-HAZE-2023/clean/001.JPG`;
</script>

<style lang="scss" scoped>
.showcase-section {
  padding: 80rpx 40rpx;
  background: #f9fafb;
}

.comparison-showcase {
  position: relative;
  border-radius: 24rpx;
  overflow: hidden;
  box-shadow: 0 20rpx 60rpx rgba(0, 0, 0, 0.15);
}

.comparison-pair {
  display: flex;
  gap: 4rpx;
}

.comparison-item {
  position: relative;
  flex: 1;
}

.showcase-image {
  width: 100%;
  height: auto;
  display: block;
}

.image-label {
  position: absolute;
  top: 16rpx;
  left: 16rpx;
  padding: 6rpx 16rpx;
  border-radius: 100rpx;
  font-size: 22rpx;
  font-weight: 600;
  backdrop-filter: blur(10rpx);
}

.label-hazy {
  background: rgba(251, 191, 36, 0.85);
  color: #ffffff;
}

.label-clean {
  background: rgba(52, 211, 153, 0.85);
  color: #ffffff;
}
</style>
