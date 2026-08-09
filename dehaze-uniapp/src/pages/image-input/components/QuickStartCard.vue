<template>
  <view class="quick-start-card">
    <view class="card-content">
      <view class="card-icon">
        <SvgIcon name="play-circle-fill" size="32" color="#ffffff" />
      </view>
      <view class="card-text">
        <text class="card-title">快速体验</text>
        <text class="card-desc">使用样例图片快速体验去雾效果</text>
      </view>
    </view>
    <view class="card-action" @click="handleQuickStart">
      <text class="action-text">立即体验</text>
      <SvgIcon name="arrow-right" size="16" color="#3b82f6" />
    </view>
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";
import type { ImageData } from "../data/imageInputData";
import { fetchSampleImages } from "../services/sampleService";
import { getImageInfo } from "../utils/image";

const emit = defineEmits<{
  (e: "start", data: ImageData): void;
}>();

const handleQuickStart = async () => {
  uni.showLoading({ title: "加载样例图片..." });

  try {
    // 从 API 获取样例图片，随机选择一张
    const samples = await fetchSampleImages("all");
    if (samples.length === 0) {
      uni.showToast({ title: "暂无可用样例图片", icon: "none" });
      return;
    }

    const sample = samples[Math.floor(Math.random() * samples.length)];
    if (!sample) {
      uni.showToast({ title: "暂无可用样例图片", icon: "none" });
      return;
    }

    const downloadResult = await downloadImage(sample.url);
    const imageInfo = await getImageInfo(downloadResult.tempFilePath);

    const imageData: ImageData = {
      url: downloadResult.tempFilePath,
      width: imageInfo.width,
      height: imageInfo.height,
      name: sample.name + ".jpg",
      sampleInfo: sample,
    };

    emit("start", imageData);
    uni.showToast({ title: "已加载样例图片", icon: "success" });
  } catch {
    uni.showToast({ title: "加载失败，请重试", icon: "none" });
  } finally {
    uni.hideLoading();
  }
};

const downloadImage = (url: string): Promise<{ tempFilePath: string }> => {
  return new Promise((resolve, reject) => {
    uni.downloadFile({
      url,
      success: (res) => {
        if (res.statusCode === 200) {
          resolve({ tempFilePath: res.tempFilePath });
        } else {
          reject(new Error("下载失败"));
        }
      },
      fail: reject,
    });
  });
};
</script>

<style lang="scss" scoped>
.quick-start-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 32rpx;
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
  border-radius: 24rpx;
  box-shadow: 0 8rpx 32rpx rgba(59, 130, 246, 0.3);
}

.card-content {
  display: flex;
  align-items: center;
  gap: 20rpx;
}

.card-icon {
  width: 72rpx;
  height: 72rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 16rpx;
}

.card-text {
  display: flex;
  flex-direction: column;
  gap: 6rpx;
}

.card-title {
  font-size: 32rpx;
  font-weight: 700;
  color: #ffffff;
}

.card-desc {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.85);
}

.card-action {
  display: flex;
  align-items: center;
  gap: 8rpx;
  padding: 16rpx 28rpx;
  background: #ffffff;
  border-radius: 32rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.95);
    opacity: 0.9;
  }
}

.action-text {
  font-size: 26rpx;
  font-weight: 600;
  color: #3b82f6;
}
</style>
