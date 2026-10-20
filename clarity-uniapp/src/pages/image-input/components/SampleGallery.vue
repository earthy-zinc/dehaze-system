<template>
  <view class="sample-gallery">
    <!-- 分类Tab -->
    <view class="category-tabs">
      <scroll-view scroll-x class="tabs-scroll">
        <view class="tabs-wrapper">
          <view
            v-for="tab in categoryTabs"
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

    <!-- 加载状态 -->
    <view v-if="loading" class="loading-container">
      <view
        class="loading-spinner"
        style="border-top-color: $color-text-placeholder"
      />
      <text class="loading-text">加载中...</text>
    </view>

    <!-- 样例图片网格 -->
    <view v-else-if="filteredSamples.length > 0" class="sample-grid">
      <SampleCard
        v-for="sample in filteredSamples"
        :key="sample.id"
        :sample="sample"
        @click="handleSampleClick"
      />
    </view>

    <!-- 空状态 -->
    <view v-else class="empty-state">
      <view class="empty-tip">暂无样例图片</view>
    </view>

    <!-- 快速体验提示 -->
    <view class="quick-tip">
      <text class="tip-text">点击任意图片即可快速体验去雾效果</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, computed, watch, onMounted } from "vue";
import SampleCard from "./SampleCard.vue";
import type { SampleImage, FogLevel, ImageData } from "../data/imageInputData";
import { CATEGORY_TABS } from "../data/imageInputData";
import { fetchSampleImages } from "../services/sampleService";
import { getImageInfo } from "../utils/image";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const activeCategory = ref<FogLevel>("all");
const loading = ref(false);
const allSamples = ref<SampleImage[]>([]);

const categoryTabs = CATEGORY_TABS;

const filteredSamples = computed(() => {
  if (activeCategory.value === "all") return allSamples.value;
  return allSamples.value.filter((s) => s.category === activeCategory.value);
});

const loadSamples = async () => {
  loading.value = true;
  try {
    allSamples.value = await fetchSampleImages(activeCategory.value);
  } catch {
    allSamples.value = [];
  } finally {
    loading.value = false;
  }
};

const handleCategoryChange = (category: FogLevel) => {
  activeCategory.value = category;
};

watch(activeCategory, () => {
  loadSamples();
});

const handleSampleClick = async (sample: SampleImage) => {
  loading.value = true;
  uni.showLoading({ title: "加载中..." });

  try {
    const downloadResult = await downloadImage(sample.url);
    const imageInfo = await getImageInfo(downloadResult.tempFilePath);

    const imageData: ImageData = {
      url: downloadResult.tempFilePath,
      width: imageInfo.width,
      height: imageInfo.height,
      name: sample.name + ".jpg",
      sampleInfo: sample,
    };

    emit("select", imageData);
    uni.showToast({ title: "样例图片加载成功", icon: "success" });
  } catch {
    uni.showToast({ title: "加载失败，请重试", icon: "none" });
  } finally {
    loading.value = false;
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

onMounted(() => {
  loadSamples();
});
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
  background: $color-bg-secondary;
  border-radius: 32rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    background: linear-gradient(135deg, $color-primary, $color-secondary);
    box-shadow: 0 4rpx 12rpx rgba(59, 130, 246, 0.3);

    .tab-text {
      color: $color-white;
    }
  }
}

.tab-text {
  font-size: 26rpx;
  font-weight: 500;
  color: #4b5563;
  white-space: nowrap;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.loading-text {
  margin-top: 16rpx;
  font-size: 26rpx;
  color: $color-text-placeholder;
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20rpx;
}

.empty-state {
  padding: 80rpx 0;
}

.quick-tip {
  display: flex;
  justify-content: center;
  padding: 24rpx 0;
}

.tip-text {
  font-size: 24rpx;
  color: $color-text-placeholder;
}

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
