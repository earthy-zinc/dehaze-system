<template>
  <view class="dataset-info">
    <text class="info-title">{{ dataset.name }}</text>
    <text class="info-desc">{{ dataset.description || "暂无描述" }}</text>
    <view class="stats-grid">
      <view class="stat-box">
        <text class="stat-value">{{ dataset.total_images }}</text>
        <text class="stat-label">总计</text>
      </view>
      <view class="stat-box">
        <text class="stat-value">{{ dataset.foggy_count }}</text>
        <text class="stat-label">有雾</text>
      </view>
      <view class="stat-box">
        <text class="stat-value">{{ dataset.clear_count }}</text>
        <text class="stat-label">无雾</text>
      </view>
      <view class="stat-box">
        <text class="stat-value">{{ dataset.annotated_count }}</text>
        <text class="stat-label">标注</text>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import type { Dataset } from "../data/datasetData";

interface Props {
  dataset: Dataset;
}

defineProps<Props>();
</script>

<style lang="scss" scoped>
.dataset-info {
  background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
  border-radius: 24rpx;
  padding: 32rpx;
  color: #ffffff;
}

.info-title {
  display: block;
  font-size: 40rpx;
  font-weight: 700;
  margin-bottom: 12rpx;
}

.info-desc {
  display: block;
  font-size: 26rpx;
  opacity: 0.9;
  margin-bottom: 32rpx;
  line-height: 1.5;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16rpx;
}

.stat-box {
  text-align: center;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 16rpx;
  padding: 16rpx 8rpx;
}

.stat-value {
  display: block;
  font-size: 40rpx;
  font-weight: 700;
  margin-bottom: 4rpx;
}

.stat-label {
  display: block;
  font-size: 22rpx;
  opacity: 0.8;
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .dataset-info {
    padding: 24rpx;
  }

  .info-title {
    font-size: 36rpx;
  }

  .stat-value {
    font-size: 32rpx;
  }

  .stat-label {
    font-size: 20rpx;
  }
}
</style>
