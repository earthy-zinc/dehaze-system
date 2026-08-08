<template>
  <view class="camera-area">
    <view class="camera-placeholder">
      <view class="camera-icon">
        <SvgIcon name="camera" size="56" color="#9ca3af" />
      </view>
      <text class="camera-text">点击下方按钮打开相机</text>
      <text class="camera-hint">拍摄需要去雾的图片</text>
    </view>

    <view class="camera-actions">
      <view class="action-btn" @click="handleOpenCamera">
        <SvgIcon name="camera-fill" size="24" color="#ffffff" />
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

    <!-- 错误提示 -->
    <view v-if="errorMsg" class="camera-error">
      <text>{{ errorMsg }}</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import type { ImageData } from "../data/imageInputData";
import { getImageInfo } from "../utils/image";

const emit = defineEmits<{
  (e: "capture", data: ImageData): void;
}>();

const processing = ref(false);
const errorMsg = ref<string | null>(null);

const handleOpenCamera = () => {
  errorMsg.value = null;

  // 先检查权限
  uni.getSetting({
    success: (setting) => {
      if (setting.authSetting["scope.camera"] === false) {
        // 之前被拒绝过，引导去设置
        uni.showModal({
          title: "需要相机权限",
          content: "请在设置中开启相机权限，用于拍摄需要去雾的图片",
          confirmText: "去设置",
          success: (res) => {
            if (res.confirm) {
              uni.openSetting({});
            }
          },
        });
        return;
      }

      // 未拒绝或已授权，直接调起相机
      openCamera();
    },
    fail: () => {
      openCamera();
    },
  });
};

const openCamera = () => {
  uni.chooseImage({
    count: 1,
    sizeType: ["compressed"],
    sourceType: ["camera"],
    success: async (res) => {
      const tempFilePath = res.tempFilePaths[0];
      const tempFiles = Array.isArray(res.tempFiles) ? res.tempFiles : [res.tempFiles];
      const tempFile = tempFiles[0];
      if (!tempFilePath || !tempFile) return;

      processing.value = true;

      try {
        const imageInfo = await getImageInfo(tempFilePath);

        const imageData: ImageData = {
          url: tempFilePath,
          width: imageInfo.width,
          height: imageInfo.height,
          size: tempFile.size,
          name: `photo_${Date.now()}.jpg`,
        };

        emit("capture", imageData);
      } catch {
        errorMsg.value = "图片处理失败，请重试";
        uni.showToast({ title: "图片处理失败，请重试", icon: "none" });
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
        errorMsg.value = "打开相机失败";
        uni.showToast({ title: "打开相机失败", icon: "none" });
      }
    },
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

.camera-error {
  margin-top: 16rpx;
  padding: 16rpx 24rpx;
  background: #fef2f2;
  border-radius: 12rpx;

  text {
    font-size: 26rpx;
    color: #ef4444;
  }
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
