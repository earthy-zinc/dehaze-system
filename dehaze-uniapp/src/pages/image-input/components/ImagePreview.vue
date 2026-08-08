<template>
  <view class="image-preview">
    <view class="preview-card">
      <!-- 头部 -->
      <view class="preview-header">
        <text class="preview-title">图片预览</text>
        <view class="close-btn" @click="handleRemove">
          <SvgIcon name="close" size="18" color="#6b7280" />
        </view>
      </view>

      <!-- 图片区域 -->
      <view class="preview-image">
        <up-image
          :src="image.url"
          mode="aspectFit"
          width="100%"
          height="400rpx"
          :lazy-load="false"
          @click="handlePreviewImage"
        />
      </view>

      <!-- 图片信息 -->
      <view class="preview-info">
        <view v-if="image.size != null" class="info-item">
          <SvgIcon name="photo" size="16" color="#3b82f6" />
          <text class="info-text">{{ formatFileSize(image.size) }}</text>
        </view>
        <view v-if="image.width != null && image.height != null" class="info-item">
          <SvgIcon name="scan" size="16" color="#10b981" />
          <text class="info-text">{{ image.width }} × {{ image.height }}</text>
        </view>
      </view>

      <!-- 样例信息（如果有） -->
      <view v-if="image.sampleInfo" class="sample-info">
        <view class="sample-tag">
          <SvgIcon name="star" size="14" color="#f59e0b" />
          <text class="tag-text">样例图片</text>
        </view>
        <text class="sample-name">{{ image.sampleInfo.name }}</text>
        <view v-if="image.sampleInfo.recommendAlgorithm" class="recommend-algo">
          <text class="recommend-label">推荐算法:</text>
          <text class="recommend-value">{{ image.sampleInfo.recommendAlgorithm }}</text>
        </view>
      </view>

      <!-- 操作按钮 -->
      <view class="preview-actions">
        <view class="action-btn primary" @click="handleNext">
          <SvgIcon name="arrow-right" size="18" color="#ffffff" />
          <text class="action-text">下一步：选择算法</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import type { ImageData } from "../data/imageInputData";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { formatFileSize } from "@/utils/format";

interface Props {
  image: ImageData;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "remove"): void;
  (e: "next"): void;
}>();

const handleRemove = () => {
  emit("remove");
};

const handleNext = () => {
  emit("next");
};

const handlePreviewImage = () => {
  uni.previewImage({
    urls: [props.image.url],
    current: props.image.url,
  });
};
</script>

<style lang="scss" scoped>
.image-preview {
  margin-top: 32rpx;
}

.preview-card {
  background: #ffffff;
  border-radius: 24rpx;
  box-shadow: 0 8rpx 32rpx rgba(0, 0, 0, 0.08);
  overflow: hidden;
}

.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24rpx 28rpx;
  border-bottom: 2rpx solid #f3f4f6;
}

.preview-title {
  font-size: 32rpx;
  font-weight: 700;
  color: #1f2937;
}

.close-btn {
  width: 56rpx;
  height: 56rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 50%;

  &:active {
    background: #e5e7eb;
  }
}

.preview-image {
  padding: 24rpx;
  background: #f9fafb;
}

.preview-info {
  display: flex;
  gap: 32rpx;
  padding: 20rpx 28rpx;
  border-bottom: 2rpx solid #f3f4f6;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 8rpx;
}

.info-text {
  font-size: 26rpx;
  color: #4b5563;
}

.sample-info {
  padding: 20rpx 28rpx;
  background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
  border-bottom: 2rpx solid #fde68a;
}

.sample-tag {
  display: inline-flex;
  align-items: center;
  gap: 8rpx;
  padding: 6rpx 16rpx;
  background: #ffffff;
  border-radius: 20rpx;
  margin-bottom: 12rpx;
}

.tag-text {
  font-size: 22rpx;
  font-weight: 600;
  color: #f59e0b;
}

.sample-name {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: #92400e;
  margin-bottom: 8rpx;
}

.recommend-algo {
  display: flex;
  align-items: center;
  gap: 8rpx;
}

.recommend-label {
  font-size: 24rpx;
  color: #b45309;
}

.recommend-value {
  font-size: 24rpx;
  font-weight: 600;
  color: #92400e;
}

.preview-actions {
  padding: 24rpx 28rpx;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12rpx;
  padding: 28rpx;
  border-radius: 16rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.98);
  }

  &.primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    box-shadow: 0 4rpx 16rpx rgba(59, 130, 246, 0.3);
  }
}

.action-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #ffffff;
}
</style>
