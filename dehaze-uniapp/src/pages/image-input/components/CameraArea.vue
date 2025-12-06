<template>
  <view class="camera-area">
    <view class="camera-placeholder">
      <view class="camera-icon">
        <u-icon name="camera" size="56" color="#9ca3af" />
      </view>
      <text class="camera-text">点击下方按钮打开相机</text>
      <text class="camera-hint">拍摄需要去雾的图片</text>
    </view>

    <view class="camera-actions">
      <view class="action-btn" @click="handleOpenCamera">
        <u-icon name="camera-fill" size="24" color="#ffffff" />
        <text class="action-text">打开相机</text>
      </view>
    </view>

    <!-- 拍照进度 -->
    <view v-if="processing" class="processing-overlay">
      <view class="processing-content">
        <up-loading-icon mode="circle" size="32" color="#3b82f6" />
        <text class="processing-text">处理中...</text>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import type { ImageData } from "../data/imageInputData";

const emit = defineEmits<{
  (e: "capture", data: ImageData): void;
}>();

const processing = ref(false);

/** 打开相机 */
const handleOpenCamera = () => {
  uni.chooseImage({
    count: 1,
    sizeType: ["compressed"],
    sourceType: ["camera"],
    success: async (res) => {
      const tempFilePath = res.tempFilePaths[0];
      const tempFile = res.tempFiles[0] as { size: number };

      processing.value = true;

      try {
        // 获取图片信息
        const imageInfo = await getImageInfo(tempFilePath);

        const imageData: ImageData = {
          url: tempFilePath,
          width: imageInfo.width,
          height: imageInfo.height,
          size: tempFile.size,
          name: `photo_${Date.now()}.jpg`,
        };

        emit("capture", imageData);
      } catch (error) {
        console.error("处理拍照图片失败:", error);
        uni.showToast({
          title: "图片处理失败，请重试",
          icon: "none",
        });
      } finally {
        processing.value = false;
      }
    },
    fail: (err) => {
      if (err.errMsg.includes("auth deny")) {
        uni.showModal({
          title: "相机权限",
          content: "需要相机权限才能拍照，是否前往设置开启？",
          confirmText: "去设置",
          success: (res) => {
            if (res.confirm) {
              uni.openSetting({});
            }
          },
        });
      } else if (err.errMsg !== "chooseImage:fail cancel") {
        uni.showToast({
          title: "打开相机失败",
          icon: "none",
        });
      }
    },
  });
};

/** 获取图片信息 */
const getImageInfo = (
  src: string
): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    uni.getImageInfo({
      src,
      success: (res) => resolve({ width: res.width, height: res.height }),
      fail: reject,
    });
  });
};
</script>

<style lang="scss" scoped>
.camera-area {
  position: relative;
}

.camera-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 64rpx 32rpx;
  background: #f9fafb;
  border-radius: 20rpx;
}

.camera-icon {
  width: 120rpx;
  height: 120rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e5e7eb;
  border-radius: 50%;
  margin-bottom: 24rpx;
}

.camera-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #374151;
  margin-bottom: 8rpx;
}

.camera-hint {
  font-size: 24rpx;
  color: #9ca3af;
}

.camera-actions {
  margin-top: 24rpx;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12rpx;
  padding: 24rpx 48rpx;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  border-radius: 16rpx;
  box-shadow: 0 4rpx 16rpx rgba(59, 130, 246, 0.3);

  &:active {
    transform: scale(0.95);
    opacity: 0.9;
  }
}

.action-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #ffffff;
}

.processing-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 20rpx;
}

.processing-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
}

.processing-text {
  font-size: 28rpx;
  color: #3b82f6;
}
</style>
