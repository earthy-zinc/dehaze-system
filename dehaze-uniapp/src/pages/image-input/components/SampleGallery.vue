<template>
  <view class="sample-gallery">
    <!-- 分类Tab -->
    <view class="category-tabs">
      <scroll-view scroll-x class="tabs-scroll">
        <view class="tabs-wrapper">
          <view
            v-for="tab in CATEGORY_TABS"
            :key="tab.key"
            class="tab-item"
            :class="{ active: activeCategory === tab.key }"
            @click="handleCategoryChange(tab.key)"
          >
            <text class="tab-text">{{ tab.label }}</text>
          </view>
        </view>
      </scroll-view>
    </view>

    <!-- 样例图片网格 -->
    <view v-if="filteredSamples.length > 0" class="sample-grid">
      <SampleCard
        v-for="sample in filteredSamples"
        :key="sample.id"
        :sample="sample"
        @click="handleSampleClick"
      />
    </view>

    <!-- 空状态 -->
    <view v-else class="empty-state">
      <up-empty mode="search" text="暂无样例图片" />
    </view>

    <!-- 加载状态 -->
    <view v-if="loading" class="loading-overlay">
      <up-loading-icon mode="circle" size="32" color="#3b82f6" />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, computed } from "vue";
import SampleCard from "./SampleCard.vue";
import type { SampleImage, FogLevel, ImageData } from "../data/imageInputData";
import {
  CATEGORY_TABS,
  getSampleImagesByCategory,
} from "../data/imageInputData";
import { getImageInfo } from "../utils/image";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const activeCategory = ref<FogLevel>("all");
const loading = ref(false);

/** 过滤后的样例图片 */
const filteredSamples = computed(() => {
  return getSampleImagesByCategory(activeCategory.value);
});

/** 切换分类 */
const handleCategoryChange = (category: FogLevel) => {
  activeCategory.value = category;
};

/** 选择样例图片 */
const handleSampleClick = async (sample: SampleImage) => {
  loading.value = true;

  uni.showLoading({ title: "加载中..." });

  try {
    // 下载远程图片到本地
    const downloadResult = await downloadImage(sample.url);

    // 获取图片信息
    const imageInfo = await getImageInfo(downloadResult.tempFilePath);

    const imageData: ImageData = {
      url: downloadResult.tempFilePath,
      width: imageInfo.width,
      height: imageInfo.height,
      size: downloadResult.dataLength || 0,
      name: sample.name + ".jpg",
      sampleInfo: sample,
    };

    emit("select", imageData);

    uni.showToast({
      title: "样例图片加载成功",
      icon: "success",
    });
  } catch {
    uni.showToast({
      title: "加载失败，请重试",
      icon: "none",
    });
  } finally {
    loading.value = false;
    uni.hideLoading();
  }
};

/** 下载图片 */
const downloadImage = (
  url: string
): Promise<{ tempFilePath: string; dataLength?: number }> => {
  return new Promise((resolve, reject) => {
    uni.downloadFile({
      url,
      success: (res) => {
        if (res.statusCode === 200) {
          resolve({
            tempFilePath: res.tempFilePath,
            dataLength: res.dataLength,
          });
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
.sample-gallery {
  position: relative;
}

.category-tabs {
  margin-bottom: 24rpx;
}

.tabs-scroll {
  white-space: nowrap;
}

.tabs-wrapper {
  display: inline-flex;
  gap: 16rpx;
  padding: 4rpx;
}

.tab-item {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 16rpx 28rpx;
  background: #f3f4f6;
  border-radius: 32rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    box-shadow: 0 4rpx 12rpx rgba(59, 130, 246, 0.3);

    .tab-text {
      color: #ffffff;
    }
  }
}

.tab-text {
  font-size: 26rpx;
  font-weight: 500;
  color: #4b5563;
  white-space: nowrap;
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20rpx;
}

.empty-state {
  padding: 80rpx 0;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 20rpx;
  z-index: 10;
}

/* 响应式适配 */
@media (min-width: 768px) {
  .sample-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1024px) {
  .sample-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
</style>
