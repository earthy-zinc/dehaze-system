<template>
  <PageLayout class="algorithm-select-page">
    <view class="main-content">
      <!-- 页面标题 -->
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="gift" size="28" color="#8b5cf6" />
        </view>
        <view class="header-text">
          <text class="header-title">选择算法</text>
          <text class="header-subtitle">选择合适的去雾算法处理图片</text>
        </view>
      </view>

      <!-- 搜索框 -->
      <view class="search-bar">
        <u-icon name="search" size="18" color="#9ca3af" />
        <input
          v-model="searchKeyword"
          class="search-input"
          type="text"
          placeholder="搜索算法名称、类型或描述"
          placeholder-class="search-placeholder"
        />
        <view
          v-if="searchKeyword"
          class="search-clear"
          @click="searchKeyword = ''"
        >
          <u-icon name="close-circle-fill" size="16" color="#9ca3af" />
        </view>
      </view>

      <!-- 已选图片预览 -->
      <view v-if="processingStore.hasImage" class="image-preview-section">
        <text class="section-label">已选图片</text>
        <view class="preview-card">
          <image
            :src="processingStore.originUrl"
            class="preview-image"
            mode="aspectFill"
          />
          <view class="preview-info">
            <text class="preview-name">
              {{ processingStore.currentImage?.name || "图片" }}
            </text>
            <text class="preview-size">
              {{ processingStore.currentImage?.width }} ×
              {{ processingStore.currentImage?.height }}
            </text>
          </view>
        </view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <up-loading-icon mode="circle" size="40" color="#8b5cf6" />
        <text class="loading-text">加载算法列表...</text>
      </view>

      <!-- 收藏筛选 + 算法列表 -->
      <template v-else-if="!error">
        <!-- 收藏筛选 -->
        <view class="filter-row">
          <view
            class="filter-tab"
            :class="{ active: onlyFavorites === false }"
            @click="onlyFavorites = false"
          >
            全部
          </view>
          <view
            class="filter-tab"
            :class="{ active: onlyFavorites === true }"
            @click="onlyFavorites = true"
          >
            <u-icon
              name="star-fill"
              size="14"
              :color="onlyFavorites ? '#8b5cf6' : '#9ca3af'"
            />
            收藏 ({{ favoriteIds.size }})
          </view>
        </view>

        <!-- 算法列表 -->
        <view class="algorithm-section">
          <text class="section-label"
            >可用算法 ({{ filteredList.length
            }}{{ searchKeyword ? "/" + algorithmList.length : "" }})</text
          >
          <view class="algorithm-list">
            <view
              v-for="algorithm in filteredList"
              :key="algorithm.id"
              class="algorithm-card"
              :class="{ selected: selectedId === algorithm.id }"
              @click="handleSelect(algorithm)"
            >
              <view class="algorithm-header">
                <view class="algorithm-name">
                  <text class="name-text">{{ algorithm.name }}</text>
                  <text class="type-badge">{{
                    algorithm.type || "未知类型"
                  }}</text>
                </view>
                <view class="header-actions">
                  <view
                    class="favorite-btn"
                    :class="{ favorited: favoriteIds.has(algorithm.id) }"
                    @click.stop="handleToggleFavorite(algorithm)"
                  >
                    <u-icon
                      :name="
                        favoriteIds.has(algorithm.id) ? 'star-fill' : 'star'
                      "
                      size="24"
                      :color="
                        favoriteIds.has(algorithm.id) ? '#fbbf24' : '#9ca3af'
                      "
                    />
                  </view>
                  <view v-if="selectedId === algorithm.id" class="check-icon">
                    <u-icon
                      name="checkmark-circle-fill"
                      size="24"
                      color="#8b5cf6"
                    />
                  </view>
                </view>
              </view>
              <text class="algorithm-desc">
                {{ algorithm.description || "暂无描述" }}
              </text>
              <view class="algorithm-meta">
                <text v-if="algorithm.version" class="meta-item"
                  >v{{ algorithm.version }}</text
                >
                <text v-if="algorithm.flops" class="meta-item">{{
                  algorithm.flops
                }}</text>
                <text v-if="algorithm.size" class="meta-item">{{
                  algorithm.size
                }}</text>
              </view>
            </view>
          </view>

          <!-- 空状态 -->
          <view v-if="filteredList.length === 0" class="empty-state">
            <up-empty
              :mode="searchKeyword ? 'search' : 'data'"
              :text="
                searchKeyword
                  ? '未找到匹配的算法'
                  : onlyFavorites
                    ? '暂无收藏的算法'
                    : '暂无可用算法'
              "
            />
          </view>
        </view>
      </template>

      <!-- 错误状态 -->
      <view v-if="error" class="error-state">
        <text class="error-text">{{ error }}</text>
        <button class="retry-btn" @click="loadAlgorithms">重新加载</button>
      </view>
    </view>

    <!-- 底部操作栏 -->
    <view v-if="!loading && !error" class="bottom-bar">
      <view class="bar-content">
        <view class="selection-info">
          <text v-if="selectedAlgorithm" class="selected-name">
            已选: {{ selectedAlgorithm.name }}
          </text>
          <text v-else class="no-selection">请选择算法</text>
        </view>
        <button
          :disabled="!selectedAlgorithm || !processingStore.hasImage"
          class="next-btn"
          @click="handleNext"
        >
          下一步
        </button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";
import {
  getAlgorithmList,
  getAlgorithmFavorites,
  toggleAlgorithmFavorite,
} from "@/api/algorithm";
import type { Algorithm } from "@/api/algorithm";
import { getErrorMessage } from "@/utils/error";

// ==================== 状态 ====================

const processingStore = useProcessingStore();
const loading = ref(false);
const error = ref("");
const algorithmList = ref<Algorithm[]>([]);
const selectedId = ref<number | null>(null);
const selectedAlgorithm = ref<Algorithm | null>(null);
const searchKeyword = ref("");
/** 已收藏算法ID集合 */
const favoriteIds = ref<Set<number>>(new Set());
/** 是否仅展示收藏算法 */
const onlyFavorites = ref(false);
/** 收藏切换中的算法ID，防止重复点击 */
const togglingIds = ref<Set<number>>(new Set());

// ==================== 计算属性 ====================

/** 按关键词与收藏筛选过滤算法列表 */
const filteredList = computed<Algorithm[]>(() => {
  let list = algorithmList.value;
  if (onlyFavorites.value) {
    list = list.filter((a) => favoriteIds.value.has(a.id));
  }
  const kw = searchKeyword.value.trim().toLowerCase();
  if (!kw) return list;
  return list.filter(
    (a) =>
      a.name.toLowerCase().includes(kw) ||
      (a.type || "").toLowerCase().includes(kw) ||
      (a.description || "").toLowerCase().includes(kw)
  );
});

// ==================== 方法 ====================

/** 加载算法列表 */
async function loadAlgorithms() {
  if (loading.value) return;

  loading.value = true;
  error.value = "";

  try {
    const [list, favorites] = await Promise.all([
      getAlgorithmList(),
      getAlgorithmFavorites().catch(() => []),
    ]);
    algorithmList.value = list;
    favoriteIds.value = new Set(favorites.map((f) => f.algorithmId));
  } catch (e) {
    const msg = getErrorMessage(e, "加载失败");
    error.value = msg;
    uni.showToast({ title: msg, icon: "none" });
  } finally {
    loading.value = false;
  }
}

/** 选择算法 */
function handleSelect(algorithm: Algorithm) {
  selectedId.value = algorithm.id;
  selectedAlgorithm.value = algorithm;

  // 同步到处理流程 Store
  processingStore.setAlgorithm(algorithm);
}

/** 切换算法收藏状态 */
async function handleToggleFavorite(algorithm: Algorithm) {
  if (togglingIds.value.has(algorithm.id)) return;
  togglingIds.value.add(algorithm.id);
  try {
    const result = await toggleAlgorithmFavorite(algorithm.id);
    const next = new Set(favoriteIds.value);
    if (result.favorited) {
      next.add(algorithm.id);
    } else {
      next.delete(algorithm.id);
    }
    favoriteIds.value = next;
    uni.showToast({
      title: result.favorited ? "已收藏" : "已取消收藏",
      icon: "none",
    });
  } catch (e) {
    const msg = getErrorMessage(e, "操作失败");
    uni.showToast({ title: msg, icon: "none" });
  } finally {
    togglingIds.value.delete(algorithm.id);
  }
}

/** 下一步：跳转到处理页 */
function handleNext() {
  if (!selectedAlgorithm.value) {
    uni.showToast({ title: "请选择算法", icon: "none" });
    return;
  }
  if (!processingStore.hasImage) {
    uni.showToast({ title: "请先选择图片", icon: "none" });
    return;
  }

  uni.navigateTo({
    url: "/pages/processing/index",
    fail: () => {
      uni.showToast({ title: "页面跳转失败", icon: "none" });
    },
  });
}

// ==================== 生命周期 ====================

onMounted(() => {
  // 允许无图浏览算法列表；“下一步”按钮会检查是否已选图片
  loadAlgorithms();
});
</script>

<style lang="scss" scoped>
.algorithm-select-page {
  width: 100%;
  min-height: 100vh;
  background: #f9fafb;
}

.main-content {
  padding: 24rpx;
  padding-bottom: calc(180rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(180rpx + env(safe-area-inset-bottom));
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
  background: linear-gradient(135deg, #ede9fe, #ddd6fe);
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

/* 搜索框 */
.search-bar {
  display: flex;
  align-items: center;
  gap: 16rpx;
  background: #ffffff;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}

.search-input {
  flex: 1;
  font-size: 28rpx;
  color: #1f2937;
}

.search-placeholder {
  color: #9ca3af;
  font-size: 28rpx;
}

.search-clear {
  padding: 8rpx;
}

/* 图片预览 */
.image-preview-section {
  margin-bottom: 24rpx;
}
.section-label {
  font-size: 28rpx;
  font-weight: 600;
  color: #374151;
  margin-bottom: 16rpx;
  display: block;
}

.preview-card {
  display: flex;
  align-items: center;
  gap: 20rpx;
  background: #ffffff;
  border-radius: 20rpx;
  padding: 20rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}

.preview-image {
  width: 120rpx;
  height: 120rpx;
  border-radius: 16rpx;
  background: #f3f4f6;
  flex-shrink: 0;
}

.preview-info {
  flex: 1;
  min-width: 0;
}

.preview-name {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 8rpx;
}

.preview-size {
  font-size: 24rpx;
  color: #9ca3af;
}

/* 算法列表 */
.algorithm-section {
  margin-bottom: 24rpx;
}
.algorithm-list {
  display: flex;
  flex-direction: column;
  gap: 20rpx;
  margin-top: 16rpx;
}

.algorithm-card {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 28rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
  border: 2rpx solid transparent;
  transition: all 0.2s ease;

  &.selected {
    border-color: #8b5cf6;
    background: #faf5ff;
    box-shadow: 0 4rpx 16rpx rgba(139, 92, 246, 0.15);
  }

  &:active {
    transform: scale(0.98);
  }
}

.algorithm-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12rpx;
}

.algorithm-name {
  display: flex;
  align-items: center;
  gap: 12rpx;
  flex: 1;
  min-width: 0;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 16rpx;
  flex-shrink: 0;
}

.favorite-btn {
  width: 56rpx;
  height: 56rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background 0.2s ease;

  &:active {
    background: #f3f4f6;
  }
}

.name-text {
  font-size: 32rpx;
  font-weight: 700;
  color: #1f2937;
}

.type-badge {
  font-size: 22rpx;
  color: #8b5cf6;
  background: #ede9fe;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
  flex-shrink: 0;
}

.check-icon {
  flex-shrink: 0;
}

/* 收藏筛选 */
.filter-row {
  display: flex;
  gap: 16rpx;
  margin-bottom: 16rpx;
}
.filter-tab {
  display: flex;
  align-items: center;
  gap: 8rpx;
  padding: 14rpx 28rpx;
  background: #ffffff;
  border-radius: 32rpx;
  font-size: 26rpx;
  color: #6b7280;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.04);

  &.active {
    background: #ede9fe;
    color: #8b5cf6;
    font-weight: 600;
  }

  &:active {
    opacity: 0.85;
  }
}

.algorithm-desc {
  display: block;
  font-size: 26rpx;
  color: #6b7280;
  line-height: 1.5;
  margin-bottom: 12rpx;
}

.algorithm-meta {
  display: flex;
  gap: 16rpx;
}
.meta-item {
  font-size: 22rpx;
  color: #9ca3af;
  background: #f3f4f6;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}

/* 加载/空/错误 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.loading-text {
  margin-top: 24rpx;
  font-size: 28rpx;
  color: #9ca3af;
}
.empty-state {
  padding: 80rpx 0;
}

.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.error-text {
  font-size: 28rpx;
  color: #ef4444;
  margin-bottom: 24rpx;
}
.retry-btn {
  padding: 16rpx 48rpx;
  background: #8b5cf6;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}

/* 底部操作栏 */
.bottom-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: #ffffff;
  border-top: 1rpx solid #f3f4f6;
  padding: 20rpx 32rpx;
  padding-bottom: calc(20rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(20rpx + env(safe-area-inset-bottom));
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.04);
  z-index: 100;
}

.bar-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24rpx;
}

.selection-info {
  flex: 1;
  min-width: 0;
}

.selected-name {
  font-size: 28rpx;
  font-weight: 600;
  color: #8b5cf6;
}

.no-selection {
  font-size: 26rpx;
  color: #9ca3af;
}

.next-btn {
  padding: 20rpx 48rpx;
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
  color: #ffffff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
  font-weight: 600;
  white-space: nowrap;

  &:disabled {
    background: #d1d5db;
    color: #9ca3af;
  }

  &:active:not(:disabled) {
    opacity: 0.85;
  }
}
</style>
