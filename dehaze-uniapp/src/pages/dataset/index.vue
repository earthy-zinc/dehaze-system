<template>
  <PageLayout level="L2" title="数据集" class="dataset-page">
    <view class="main-content">
      <!-- 页面标题卡片 -->
      <PageHeaderCard
        icon="server-fill"
        icon-color="#14b8a6"
        icon-bg="linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%)"
        title="数据集"
        subtitle="浏览公开和共享数据集"
      />

      <!-- 搜索栏 -->
      <view class="search-wrapper">
        <view class="search-bar">
          <SvgIcon name="search" size="18" color="#9ca3af" />
          <input
            v-model="searchKeyword"
            class="search-input"
            type="text"
            placeholder="搜索数据集或图片..."
            placeholder-class="search-placeholder"
            @confirm="handleSearch"
          />
          <view
            v-if="searchKeyword"
            class="search-clear"
            @click="handleClearSearch"
          >
            <SvgIcon name="close-circle-fill" size="16" color="#9ca3af" />
          </view>
        </view>
      </view>

      <!-- 数据集列表视图 -->
      <view v-if="currentView === 'list'" class="list-view">
        <!-- 加载状态 -->
        <view v-if="listLoading" class="loading-container">
          <view class="loading-spinner" />
          <text class="loading-text">加载中...</text>
        </view>

        <!-- 数据集列表 -->
        <view v-else-if="datasets.length > 0" class="dataset-list">
          <view
            v-for="dataset in datasets"
            :key="dataset.id"
            class="dataset-card"
            @click="handleDatasetClick(dataset)"
          >
            <view class="card-thumbnail">
              <image
                v-if="thumbnailMap[dataset.id]"
                :src="thumbnailMap[dataset.id]"
                mode="aspectFill"
                class="thumbnail-img"
              />
            </view>
            <view class="card-content">
              <text class="card-title">{{ dataset.name }}</text>
              <text class="card-desc">
                {{ dataset.description || "暂无描述" }}
              </text>
              <view class="card-stats">
                <view class="stat-item">
                  <SvgIcon name="photo" size="14" color="#14b8a6" />
                  <text class="stat-text">{{ dataset.total ?? 0 }}</text>
                </view>
                <view class="stat-item">
                  <SvgIcon name="clock" size="14" color="#9ca3af" />
                  <text class="stat-text">
                    {{
                      formatRelativeTime(
                        dataset.createTime ? String(dataset.createTime) : ""
                      )
                    }}
                  </text>
                </view>
              </view>
            </view>
          </view>
        </view>

        <!-- 空状态 -->
        <view v-else class="empty-container">
          <view class="empty-tip">暂无数据集</view>
        </view>
      </view>

      <!-- 数据集详情视图 -->
      <view v-else class="detail-view">
        <!-- 返回按钮 -->
        <view class="back-btn" @click="handleBackToList">
          <SvgIcon name="arrow-left" size="18" color="#4b5563" />
          <text class="back-text">返回列表</text>
        </view>

        <!-- 数据集信息 -->
        <view v-if="currentDataset" class="dataset-info">
          <text class="info-title">{{ currentDataset.name }}</text>
          <text class="info-desc">
            {{ currentDataset.description || "暂无描述" }}
          </text>
          <view class="stats-grid">
            <view class="stat-box">
              <text class="stat-value">
                {{
                  currentDataset.total ??
                  currentDataset.statistics?.itemCount ??
                  0
                }}
              </text>
              <text class="stat-label">总计</text>
            </view>
            <view class="stat-box">
              <text class="stat-value">
                {{ currentDataset.statistics?.annotatedCount ?? 0 }}
              </text>
              <text class="stat-label">已标注</text>
            </view>
            <view class="stat-box">
              <text class="stat-value">
                {{ currentDataset.statistics?.unannotatedCount ?? 0 }}
              </text>
              <text class="stat-label">未标注</text>
            </view>
          </view>
        </view>

        <!-- 图片网格 -->
        <view class="image-section">
          <view v-if="imagesLoading" class="loading-container">
            <view class="loading-spinner" />
            <text class="loading-text">加载图片中...</text>
          </view>

          <view v-else-if="images.length > 0" class="image-grid">
            <view
              v-for="image in images"
              :key="image.id"
              class="image-card"
              @click="handleImageClick(image)"
            >
              <image
                :src="image.imageUrl"
                mode="aspectFill"
                class="grid-image"
              />
              <view class="image-overlay">
                <text class="image-type">
                  {{ getTypeLabel(image.type) }}
                </text>
              </view>
            </view>
          </view>

          <view v-else class="empty-container">
            <view class="empty-tip">暂无图片</view>
          </view>
        </view>
      </view>

      <!-- 使用提示 -->
      <view v-if="currentView === 'list'" class="tips-card">
        <view class="tips-header">
          <SvgIcon name="info-circle" size="18" color="#14b8a6" />
          <text class="tips-title">使用提示</text>
        </view>
        <view class="tips-list">
          <text class="tips-item">• 点击数据集卡片查看详细信息</text>
          <text class="tips-item">• 点击图片可查看大图</text>
          <text class="tips-item">• 使用该数据集可带入去雾流程</text>
        </view>
      </view>
    </view>

    <!-- 图片预览弹窗 -->
    <Popup
      :show="showImageViewer"
      mode="center"
      round
      @close="closeImageViewer"
    >
      <view v-if="selectedImage" class="image-viewer">
        <view class="viewer-header">
          <text class="viewer-title">{{ selectedImage.filename }}</text>
          <view class="viewer-close" @click="closeImageViewer">
            <SvgIcon name="close" size="24" color="#ffffff" />
          </view>
        </view>
        <view class="viewer-image">
          <image
            :src="selectedImage.imageUrl"
            mode="widthFix"
            class="viewer-img"
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
    </Popup>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted, reactive } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import type { DatasetImageItem } from "./data/datasetData";
import {
  formatHazeLevel,
  IMAGE_TYPE_LABELS,
  flattenDatasetItems,
} from "./data/datasetData";
import { DatasetAPI, DatasetItemAPI } from "dehaze-sdk-js";
import type { Dataset } from "dehaze-sdk-js";
import { formatFileSize, formatRelativeTime } from "@/utils/format";

// ==================== 状态定义 ====================

const currentView = ref<"list" | "detail">("list");
const searchKeyword = ref("");
const datasets = ref<Dataset[]>([]);
const listLoading = ref(false);
const currentDataset = ref<Dataset | null>(null);
const images = ref<DatasetImageItem[]>([]);
const imagesLoading = ref(false);
const showImageViewer = ref(false);
const selectedImage = ref<DatasetImageItem | null>(null);
const thumbnailMap = reactive<Record<number, string>>({});

const getTypeLabel = (type: string) => IMAGE_TYPE_LABELS[type] || type;
const getHazeLevelLabel = (level?: string) =>
  formatHazeLevel(level) || "未标注";

// ==================== 数据加载 ====================

async function loadThumbnail(datasetId: number) {
  try {
    const result = await DatasetItemAPI.getList({
      datasetId,
      pageNum: 1,
      pageSize: 1,
    });
    const first = flattenDatasetItems(result.list)[0];
    if (first?.imageUrl) {
      thumbnailMap[datasetId] = first.imageUrl;
    }
  } catch {
    // 忽略
  }
}

const loadDatasets = async () => {
  if (listLoading.value) return;
  listLoading.value = true;
  try {
    const result = await DatasetAPI.getList({
      pageNum: 1,
      pageSize: 10,
      keyword: searchKeyword.value || undefined,
    });
    datasets.value = result.list;
    // 异步加载缩略图
    datasets.value.forEach((ds) => loadThumbnail(ds.id));
  } catch {
    uni.showToast({ title: "加载失败，请重试", icon: "none" });
  } finally {
    listLoading.value = false;
  }
};

const loadImages = async () => {
  if (!currentDataset.value || imagesLoading.value) return;
  imagesLoading.value = true;
  try {
    const result = await DatasetItemAPI.getList({
      datasetId: currentDataset.value.id,
      pageNum: 1,
      pageSize: 50,
      keyword: searchKeyword.value || undefined,
    });
    images.value = flattenDatasetItems(result.list);
  } catch {
    uni.showToast({ title: "加载失败，请重试", icon: "none" });
  } finally {
    imagesLoading.value = false;
  }
};

// ==================== 事件处理 ====================

const handleSearch = () => {
  if (currentView.value === "list") {
    loadDatasets();
  } else {
    loadImages();
  }
};

const handleClearSearch = () => {
  searchKeyword.value = "";
  handleSearch();
};

const handleDatasetClick = async (dataset: Dataset) => {
  uni.showLoading({ title: "加载中..." });
  try {
    const detail = await DatasetAPI.getDatasetInfoById(dataset.id);
    currentDataset.value = detail;
    currentView.value = "detail";
    await loadImages();
  } catch {
    uni.showToast({ title: "加载失败，请重试", icon: "none" });
  } finally {
    uni.hideLoading();
  }
};

const handleBackToList = () => {
  currentView.value = "list";
  currentDataset.value = null;
  images.value = [];
};

const handleImageClick = (image: DatasetImageItem) => {
  selectedImage.value = image;
  showImageViewer.value = true;
};

const closeImageViewer = () => {
  showImageViewer.value = false;
  selectedImage.value = null;
};

// ==================== 生命周期 ====================

onMounted(() => {
  // 支持通过路由参数直接进入数据集详情
  const pages = getCurrentPages();
  const currentPage = pages[pages.length - 1] as any;
  const datasetId = currentPage?.$page?.options?.datasetId;
  if (datasetId) {
    const id = Number(datasetId);
    if (!Number.isNaN(id)) {
      DatasetAPI.getDatasetInfoById(id)
        .then((detail) => {
          currentDataset.value = detail;
          currentView.value = "detail";
          loadImages();
        })
        .catch(() => {
          uni.showToast({ title: "加载失败", icon: "none" });
          loadDatasets();
        });
      return;
    }
  }
  loadDatasets();
});
</script>

<style lang="scss" scoped>
@import "@/styles/mixins.scss";

.dataset-page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: $spacing-md;
  @include safe-area-bottom(120rpx);
}

/* 搜索栏 */
.search-wrapper {
  margin-bottom: $spacing-md;
}

/* 列表视图 */
.list-view {
  min-height: 400rpx;
}

.dataset-list {
  display: flex;
  flex-direction: column;
  gap: $spacing-md;
}

.dataset-card {
  display: flex;
  background: $color-white;
  border-radius: $radius-xl;
  overflow: hidden;
  box-shadow: $shadow-md;

  &:active {
    transform: scale(0.98);
  }
}

.card-thumbnail {
  width: 240rpx;
  height: 240rpx;
  flex-shrink: 0;
  background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
}

.card-content {
  flex: 1;
  padding: 24rpx;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-width: 0;
}

.card-title {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
  margin-bottom: 12rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-desc {
  font-size: $font-sm;
  color: $color-text-secondary;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-bottom: 16rpx;
}

.card-stats {
  display: flex;
  align-items: center;
  gap: 32rpx;
  margin-bottom: 16rpx;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 8rpx;
}

.stat-text {
  font-size: 24rpx;
  color: $color-text-placeholder;
}

/* 详情视图 */
.detail-view {
  min-height: 400rpx;
}

.back-btn {
  display: inline-flex;
  align-items: center;
  gap: $spacing-xs;
  padding: $spacing-sm $spacing-md;
  background: $color-white;
  border-radius: $radius-lg;
  margin-bottom: $spacing-md;
  box-shadow: $shadow-sm;

  &:active {
    transform: scale(0.95);
  }
}

.back-text {
  font-size: $font-md;
  color: #4b5563;
}

/* 数据集信息 */
.dataset-info {
  background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
  border-radius: $radius-xl;
  padding: $spacing-lg;
  color: $color-white;
  margin-bottom: $spacing-md;
}

.info-title {
  display: block;
  font-size: 40rpx;
  font-weight: 700;
  margin-bottom: 12rpx;
}

.info-desc {
  display: block;
  font-size: $font-sm;
  opacity: 0.9;
  margin-bottom: 32rpx;
  line-height: 1.5;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: $spacing-sm;
}

.stat-box {
  text-align: center;
  background: rgba(255, 255, 255, 0.15);
  border-radius: $radius-lg;
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
  font-size: $font-xs;
  opacity: 0.8;
}

/* 图片网格 */
.image-section {
  min-height: 200rpx;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: $spacing-sm;
}

.image-card {
  position: relative;
  border-radius: $radius-lg;
  overflow: hidden;
  background: $color-bg-secondary;

  &:active {
    opacity: 0.85;
  }
}

.image-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 12rpx 16rpx;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.5), transparent);
}

.image-type {
  font-size: $font-xs;
  color: $color-white;
  font-weight: 500;
}

/* 加载/空状态 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 120rpx 0;
}

.loading-spinner {
  border-top-color: #14b8a6;
}

.loading-text {
  margin-top: $spacing-md;
  font-size: $font-md;
  color: $color-text-placeholder;
}

.empty-container {
  padding: 80rpx 0;
}

/* 使用提示 */
.tips-card {
  background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
  border: 2rpx solid #99f6e4;
  border-radius: $radius-xl;
  padding: $spacing-lg;
  margin-top: $spacing-lg;
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
  font-size: $font-sm;
  color: #115e59;
  line-height: 1.5;
}

/* 图片查看器 */
.image-viewer {
  width: 90vw;
  max-width: 800rpx;
  max-height: 90vh;
  background: $color-black;
  border-radius: $radius-xl;
  overflow: hidden;
}

.viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: $spacing-md;
  background: rgba(0, 0, 0, 0.8);
}

.viewer-title {
  font-size: $font-md;
  color: $color-white;
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
  background: $color-black;
}

.viewer-img {
  width: 100%;
}

.viewer-info {
  padding: $spacing-md;
  background: linear-gradient(to top, rgba(0, 0, 0, 0.9), transparent);
}

.info-row {
  display: flex;
  gap: $spacing-sm;
  margin-bottom: 12rpx;

  &:last-child {
    margin-bottom: 0;
  }
}

.info-label {
  font-size: $font-sm;
  color: rgba(255, 255, 255, 0.7);
  flex-shrink: 0;
}

.info-value {
  font-size: $font-sm;
  color: $color-white;
}

@media (max-width: 375px) {
  .main-content {
    padding: $spacing-sm;
  }
  .tips-card {
    padding: $spacing-md;
  }
  .card-thumbnail {
    width: 200rpx;
    height: 200rpx;
  }
}
</style>
