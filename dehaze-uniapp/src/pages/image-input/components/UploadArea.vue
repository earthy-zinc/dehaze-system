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
        <text class="progress-text">正在上传... {{ uploadProgress }}%</text>
        <view class="progress-bar">
          <view
            class="progress-fill"
            :style="{ width: uploadProgress + '%' }"
          />
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { uploadImage } from "@/api/file";
import { useProcessingStore } from "@/store/processing";
import { MAX_FILE_SIZE, COMPRESS_THRESHOLD } from "../data/imageInputData";
import { formatFileSize } from "@/utils/format";
import type { ImageData } from "../data/imageInputData";
import { getImageInfo } from "../utils/image";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const processingStore = useProcessingStore();

const uploading = ref(false);
const uploadProgress = ref(0);

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
      uploadProgress.value = 0;

      try {
        let finalPath = tempFilePath;

        // 大于 5MB 自动压缩
        if (tempFile.size > COMPRESS_THRESHOLD) {
          uni.showToast({
            title: "图片较大，正在压缩...",
            icon: "loading",
            duration: 2000,
          });

          try {
            finalPath = await compressImage(tempFilePath);
          } catch {
            // 压缩失败时回退使用原图
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

        processingStore.startUploading();

        // 上传到后端
        try {
          const fileInfo = await uploadImage(imageData, (progress) => {
            uploadProgress.value = progress;
          });

          imageData.fileId = fileInfo.id;
          imageData.remoteUrl = fileInfo.url;

          uni.showToast({ title: "上传成功", icon: "success" });
        } catch {
          uni.showToast({
            title: "上传失败，将使用本地图片处理",
            icon: "none",
            duration: 2000,
          });
        }

        processingStore.setImage(imageData);
        emit("select", imageData);
      } catch {
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
  width: 70%;
}

.progress-text {
  font-size: 28rpx;
  color: #3b82f6;
}

.progress-bar {
  width: 100%;
  height: 8rpx;
  background: #e5e7eb;
  border-radius: 4rpx;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #6366f1);
  border-radius: 4rpx;
  transition: width 0.3s ease;
}
</style>
