<template>
  <view class="upload-area">
    <!-- 上传区域 -->
    <view class="upload-zone" @click="handleChooseImage">
      <view class="upload-icon">
        <u-icon name="plus" size="48" color="#9ca3af" />
      </view>
      <text class="upload-text">点击选择图片</text>
      <text class="upload-hint">支持 JPG、PNG、WEBP、HEIC 格式</text>
      <text class="upload-hint">单张图片最大 20MB</text>
    </view>

    <!-- 上传进度 -->
    <view v-if="uploading" class="upload-progress">
      <view class="progress-content">
        <up-loading-icon mode="circle" size="32" color="#3b82f6" />
        <text class="progress-text">正在处理...</text>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import {
  MAX_FILE_SIZE,
  COMPRESS_THRESHOLD,
  formatFileSize,
} from "../data/imageInputData";
import type { ImageData } from "../data/imageInputData";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const uploading = ref(false);

/** 选择图片 */
const handleChooseImage = () => {
  uni.chooseImage({
    count: 1,
    sizeType: ["original", "compressed"],
    sourceType: ["album"],
    success: async (res) => {
      const tempFilePath = res.tempFilePaths[0];
      const tempFile = res.tempFiles[0] as { size: number; path: string };

      // 检查文件大小
      if (tempFile.size > MAX_FILE_SIZE) {
        uni.showToast({
          title: `图片大小超过${formatFileSize(MAX_FILE_SIZE)}，请选择较小的图片`,
          icon: "none",
          duration: 2500,
        });
        return;
      }

      uploading.value = true;

      try {
        let finalPath = tempFilePath;

        // 大于5MB自动压缩
        if (tempFile.size > COMPRESS_THRESHOLD) {
          uni.showToast({
            title: "图片较大，正在压缩...",
            icon: "loading",
            duration: 3000,
          });

          try {
            const compressResult = await compressImage(tempFilePath);
            finalPath = compressResult;
          } catch (e) {
            console.warn("压缩失败，使用原图:", e);
          }
        }

        // 获取图片信息
        const imageInfo = await getImageInfo(finalPath);

        const imageData: ImageData = {
          url: finalPath,
          width: imageInfo.width,
          height: imageInfo.height,
          size: tempFile.size,
          name: extractFileName(tempFilePath),
        };

        emit("select", imageData);
      } catch (error) {
        console.error("处理图片失败:", error);
        uni.showToast({
          title: "图片处理失败，请重试",
          icon: "none",
        });
      } finally {
        uploading.value = false;
        uni.hideToast();
      }
    },
    fail: (err) => {
      if (err.errMsg !== "chooseImage:fail cancel") {
        uni.showToast({
          title: "选择图片失败",
          icon: "none",
        });
      }
    },
  });
};

/** 压缩图片 */
const compressImage = (src: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    uni.compressImage({
      src,
      quality: 85,
      success: (res) => resolve(res.tempFilePath),
      fail: reject,
    });
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

/** 提取文件名 */
const extractFileName = (path: string): string => {
  const parts = path.split("/");
  return parts[parts.length - 1] || "image.jpg";
};
</script>

<style lang="scss" scoped>
.upload-area {
  position: relative;
}

.upload-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 64rpx 32rpx;
  background: #f9fafb;
  border: 2rpx dashed #d1d5db;
  border-radius: 20rpx;
  transition: all 0.2s ease;

  &:active {
    background: #f3f4f6;
    border-color: #3b82f6;
  }
}

.upload-icon {
  width: 100rpx;
  height: 100rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e5e7eb;
  border-radius: 50%;
  margin-bottom: 24rpx;
}

.upload-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #374151;
  margin-bottom: 12rpx;
}

.upload-hint {
  font-size: 24rpx;
  color: #9ca3af;
  line-height: 1.5;
}

.upload-progress {
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

.progress-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
}

.progress-text {
  font-size: 28rpx;
  color: #3b82f6;
}
</style>
