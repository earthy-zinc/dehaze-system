<template>
  <PageLayout class="dataset-page">
    <view class="main-content">
      <!-- 页面标题卡片 -->
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="server-fill" size="28" color="#14b8a6" />
        </view>
        <view class="header-text">
          <text class="header-title">数据集管理</text>
          <text class="header-subtitle">浏览和管理图像去雾数据集</text>
        </view>
      </view>

      <!-- 搜索栏 -->
      <view class="search-wrapper">
        <up-search
          v-model="searchKeyword"
          placeholder="搜索数据集或图片..."
          :show-action="false"
          bg-color="#f3f4f6"
          shape="round"
          @search="handleSearch"
          @clear="handleClearSearch"
        />
      </view>

      <!-- 数据集列表视图 -->
      <view v-if="currentView === 'list'" class="list-view">
        <!-- 加载状态 -->
        <view v-if="listLoading" class="loading-container">
          <up-loading-icon mode="circle" size="40" color="#14b8a6" />
          <text class="loading-text">加载中...</text>
        </view>

        <!-- 数据集列表 -->
        <view v-else-if="datasets.length > 0" class="dataset-list">
          <DatasetCard
            v-for="dataset in datasets"
            :key="dataset.id"
            :dataset="dataset"
            @click="handleDatasetClick"
          />
        </view>

        <!-- 空状态 -->
        <view v-else class="empty-container">
          <up-empty mode="search" text="暂无数据集" />
        </view>
      </view>

      <!-- 数据集详情视图 -->
      <view v-else class="detail-view">
        <!-- 返回按钮 -->
        <view class="back-btn" @click="handleBackToList">
          <u-icon name="arrow-left" size="18" color="#4b5563" />
          <text class="back-text">返回列表</text>
        </view>

        <!-- 数据集信息 -->
        <DatasetInfo v-if="currentDataset" :dataset="currentDataset" />

        <!-- 图片网格/瀑布流 -->
        <ImageGrid
          :images="images"
          :loading="imagesLoading"
          :has-more="hasMore"
          :initial-mode="displayMode"
          @image-click="handleImageClick"
          @load-more="handleLoadMore"
          @mode-change="handleModeChange"
        />
      </view>

      <!-- 使用提示 -->
      <view v-if="currentView === 'list'" class="tips-card">
        <view class="tips-header">
          <u-icon name="info-circle" size="18" color="#14b8a6" />
          <text class="tips-title">使用提示</text>
        </view>
        <view class="tips-list">
          <text class="tips-item">• 点击数据集卡片查看详细信息</text>
          <text class="tips-item">• 点击图片可查看大图</text>
          <text class="tips-item">• 支持网格和瀑布流两种展示模式</text>
        </view>
      </view>
    </view>

    <!-- 图片预览弹窗 -->
    <up-popup
      :show="showImageViewer"
      mode="center"
      :round="16"
      :close-on-click-overlay="true"
      @close="closeImageViewer"
    >
      <view v-if="selectedImage" class="image-viewer">
        <view class="viewer-header">
          <text class="viewer-title">{{ selectedImage.filename }}</text>
          <view class="viewer-close" @click="closeImageViewer">
            <u-icon name="close" size="24" color="#ffffff" />
          </view>
        </view>
        <view class="viewer-image">
          <up-image
            :src="selectedImage.imageUrl"
            mode="widthFix"
            width="100%"
            :lazy-load="false"
          />
        </view>
        <view class="viewer-info">
          <view class="info-row">
            <text class="info-label">类型:</text>
            <text class="info-value">{{
              getTypeLabel(selectedImage.type)
            }}</text>
          </view>
          <view class="info-row">
            <text class="info-label">雾霾程度:</text>
            <text class="info-value">{{
              getHazeLevelLabel(selectedImage.hazeLevel)
            }}</text>
          </view>
          <view class="info-row">
            <text class="info-label">尺寸:</text>
            <text class="info-value"
              >{{ selectedImage.width }} × {{ selectedImage.height }}</text
            >
          </view>
          <view class="info-row">
            <text class="info-label">大小:</text>
            <text class="info-value">{{
              formatFileSize(selectedImage.fileSize)
            }}</text>
          </view>
        </view>
      </view>
    </up-popup>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import DatasetCard from "./components/DatasetCard.vue";
import DatasetInfo from "./components/DatasetInfo.vue";
import ImageGrid from "./components/ImageGrid.vue";
import type { Dataset, DatasetImageItem, DisplayMode } from "./data/datasetData";
import { formatHazeLevel, IMAGE_TYPE_LABELS } from "./data/datasetData";
import {
  getDatasets,
  getDatasetDetail,
  getDatasetItems,
  flattenDatasetItems,
} from "@/api/dataset";
import { formatFileSize } from "@/utils/format";

// ==================== 状态定义 ====================

/** 当前视图 */
const currentView = ref<"list" | "detail">("list");

/** 搜索关键词 */
const searchKeyword = ref("");

/** 数据集列表 */
const datasets = ref<Dataset[]>([]);

/** 列表加载状态 */
const listLoading = ref(false);

/** 当前选中的数据集 */
const currentDataset = ref<Dataset | null>(null);

/** 图片列表 */
const images = ref<DatasetImageItem[]>([]);

/** 图片加载状态 */
const imagesLoading = ref(false);

/** 当前页码 */
const currentPage = ref(1);

/** 是否有更多数据 */
const hasMore = ref(true);

/** 展示模式 */
const displayMode = ref<DisplayMode>("grid");

/** 图片查看器显示状态 */
const showImageViewer = ref(false);

/** 当前选中的图片 */
const selectedImage = ref<DatasetImageItem | null>(null);

/** 每页大小 */
const PAGE_SIZE = 20;

// ==================== 方法定义 ====================

/** 获取图片类型标签 */
const getTypeLabel = (type: string) => IMAGE_TYPE_LABELS[type] || type;

/** 获取雾霾程度展示文本 */
const getHazeLevelLabel = (level?: string) => {
  const label = formatHazeLevel(level);
  return label || "未标注";
};

/** 加载数据集列表 */
const loadDatasets = async () => {
  if (listLoading.value) return;

  listLoading.value = true;
  try {
    const result = await getDatasets(1, searchKeyword.value);
    datasets.value = result.list;
  } catch (error) {
    console.error("加载数据集失败:", error);
    uni.showToast({
      title: "加载失败，请重试",
      icon: "none",
    });
  } finally {
    listLoading.value = false;
  }
};

/** 加载图片列表 */
const loadImages = async (append = false) => {
  if (!currentDataset.value || imagesLoading.value) return;

  imagesLoading.value = true;
  try {
    const result = await getDatasetItems(currentDataset.value.id, {
      pageNum: currentPage.value,
      pageSize: PAGE_SIZE,
      keyword: searchKeyword.value || undefined,
    });

    const imageItems = flattenDatasetItems(result.list);
    if (append) {
      images.value = [...images.value, ...imageItems];
    } else {
      images.value = imageItems;
    }
    hasMore.value = currentPage.value * PAGE_SIZE < result.total;
  } catch (error) {
    console.error("加载图片失败:", error);
    uni.showToast({
      title: "加载失败，请重试",
      icon: "none",
    });
  } finally {
    imagesLoading.value = false;
  }
};

/** 搜索处理 */
const handleSearch = () => {
  if (currentView.value === "list") {
    loadDatasets();
  } else {
    currentPage.value = 1;
    loadImages();
  }
};

/** 清除搜索 */
const handleClearSearch = () => {
  searchKeyword.value = "";
  handleSearch();
};

/** 数据集点击 */
const handleDatasetClick = async (dataset: Dataset) => {
  uni.showLoading({ title: "加载中..." });

  try {
    const detail = await getDatasetDetail(dataset.id);
    currentDataset.value = detail;
    currentView.value = "detail";
    currentPage.value = 1;
    images.value = [];
    hasMore.value = true;

    await loadImages();
  } catch (error) {
    console.error("加载详情失败:", error);
    uni.showToast({
      title: "加载失败，请重试",
      icon: "none",
    });
  } finally {
    uni.hideLoading();
  }
};

/** 返回列表 */
const handleBackToList = () => {
  currentView.value = "list";
  currentDataset.value = null;
  currentPage.value = 1;
  images.value = [];
  hasMore.value = true;
};

/** 加载更多 */
const handleLoadMore = () => {
  if (hasMore.value && !imagesLoading.value) {
    currentPage.value++;
    loadImages(true);
  }
};

/** 展示模式变更 */
const handleModeChange = (mode: DisplayMode) => {
  displayMode.value = mode;
};

/** 图片点击 */
const handleImageClick = (image: DatasetImageItem) => {
  selectedImage.value = image;
  showImageViewer.value = true;
};

/** 关闭图片查看器 */
const closeImageViewer = () => {
  showImageViewer.value = false;
  selectedImage.value = null;
};

// ==================== 生命周期 ====================

onMounted(() => {
  loadDatasets();
});
</script>

<style lang="scss" scoped>
.dataset-page {
  width: 100%;
  min-height: 100vh;
  background: #f9fafb;
}

.main-content {
  padding: 24rpx;
  padding-bottom: calc(120rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(120rpx + env(safe-area-inset-bottom));
}

/* 页面标题卡片 */
.page-header-card {
  display: flex;
  align-items: center;
  gap: 24rpx;
  background: #ffffff;
  border-radius: 24rpx;
  padding: 32rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
}

.header-icon {
  width: 80rpx;
  height: 80rpx;
  background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%);
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.header-text {
  flex: 1;
}

.header-title {
  display: block;
  font-size: 36rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 8rpx;
}

.header-subtitle {
  display: block;
  font-size: 26rpx;
  color: #6b7280;
}

/* 搜索栏 */
.search-wrapper {
  margin-bottom: 24rpx;
}

/* 列表视图 */
.list-view {
  min-height: 400rpx;
}

.dataset-list {
  display: flex;
  flex-direction: column;
  gap: 24rpx;
}

/* 详情视图 */
.detail-view {
  min-height: 400rpx;
}

/* 返回按钮 */
.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 8rpx;
  padding: 16rpx 24rpx;
  background: #ffffff;
  border-radius: 16rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.06);

  &:active {
    transform: scale(0.95);
  }
}

.back-text {
  font-size: 28rpx;
  color: #4b5563;
}

/* 加载状态 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 120rpx 0;
}

.loading-text {
  margin-top: 24rpx;
  font-size: 28rpx;
  color: #9ca3af;
}

/* 空状态 */
.empty-container {
  padding: 80rpx 0;
}

/* 使用提示 */
.tips-card {
  background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
  border: 2rpx solid #99f6e4;
  border-radius: 24rpx;
  padding: 32rpx;
  margin-top: 32rpx;
}

.tips-header {
  display: flex;
  align-items: center;
  gap: 12rpx;
  margin-bottom: 20rpx;
}

.tips-title {
  font-size: 30rpx;
  font-weight: 600;
  color: #0f766e;
}

.tips-list {
  display: flex;
  flex-direction: column;
  gap: 12rpx;
}

.tips-item {
  font-size: 26rpx;
  color: #115e59;
  line-height: 1.5;
}

/* 图片查看器 */
.image-viewer {
  width: 90vw;
  max-width: 800rpx;
  max-height: 90vh;
  background: #000000;
  border-radius: 24rpx;
  overflow: hidden;
}

.viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24rpx;
  background: rgba(0, 0, 0, 0.8);
}

.viewer-title {
  font-size: 28rpx;
  color: #ffffff;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.viewer-close {
  width: 64rpx;
  height: 64rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);

  &:active {
    background: rgba(255, 255, 255, 0.2);
  }
}

.viewer-image {
  width: 100%;
  background: #000000;
}

.viewer-info {
  padding: 24rpx;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.9), transparent);
}

.info-row {
  display: flex;
  gap: 16rpx;
  margin-bottom: 12rpx;

  &:last-child {
    margin-bottom: 0;
  }
}

.info-label {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.7);
  flex-shrink: 0;
}

.info-value {
  font-size: 26rpx;
  color: #ffffff;
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .main-content {
    padding: 16rpx;
  }

  .page-header-card {
    padding: 24rpx;
  }

  .header-title {
    font-size: 32rpx;
  }

  .tips-card {
    padding: 24rpx;
  }
}
</style>
