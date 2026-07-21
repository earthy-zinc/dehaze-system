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
              {{ processingStore.currentImage?.width }} × {{ processingStore.currentImage?.height }}
            </text>
          </view>
        </view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <up-loading-icon mode="circle" size="40" color="#8b5cf6" />
        <text class="loading-text">加载算法列表...</text>
      </view>

      <!-- 算法列表 -->
      <view v-else class="algorithm-section">
        <text class="section-label">可用算法 ({{ algorithmList.length }})</text>
        <view class="algorithm-list">
          <view
            v-for="algorithm in algorithmList"
            :key="algorithm.id"
            class="algorithm-card"
            :class="{ selected: selectedId === algorithm.id }"
            @click="handleSelect(algorithm)"
          >
            <view class="algorithm-header">
              <view class="algorithm-name">
                <text class="name-text">{{ algorithm.name }}</text>
                <text class="type-badge">{{ algorithm.type || "未知类型" }}</text>
              </view>
              <view v-if="selectedId === algorithm.id" class="check-icon">
                <u-icon name="checkmark-circle-fill" size="24" color="#8b5cf6" />
              </view>
            </view>
            <text class="algorithm-desc">
              {{ algorithm.description || "暂无描述" }}
            </text>
            <view class="algorithm-meta">
              <text v-if="algorithm.version" class="meta-item">v{{ algorithm.version }}</text>
              <text v-if="algorithm.flops" class="meta-item">{{ algorithm.flops }}</text>
              <text v-if="algorithm.size" class="meta-item">{{ algorithm.size }}</text>
            </view>
          </view>
        </view>

        <!-- 空状态 -->
        <view v-if="algorithmList.length === 0" class="empty-state">
          <up-empty mode="search" text="暂无可用算法" />
        </view>
      </view>

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
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";
import { getAlgorithmList } from "@/api/algorithm";
import type { Algorithm } from "@/api/algorithm";

// ==================== 状态 ====================

const processingStore = useProcessingStore();
const loading = ref(false);
const error = ref("");
const algorithmList = ref<Algorithm[]>([]);
const selectedId = ref<number | null>(null);
const selectedAlgorithm = ref<Algorithm | null>(null);

// ==================== 方法 ====================

/** 加载算法列表 */
async function loadAlgorithms() {
  if (loading.value) return;

  loading.value = true;
  error.value = "";

  try {
    const list = await getAlgorithmList();
    algorithmList.value = list;
  } catch (e) {
    const msg = (e as { message?: string }).message || "加载失败";
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

.header-text { flex: 1; }
.header-title { display: block; font-size: 36rpx; font-weight: 700; color: #1f2937; margin-bottom: 8rpx; }
.header-subtitle { display: block; font-size: 26rpx; color: #6b7280; }

/* 图片预览 */
.image-preview-section { margin-bottom: 24rpx; }
.section-label { font-size: 28rpx; font-weight: 600; color: #374151; margin-bottom: 16rpx; display: block; }

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
.algorithm-section { margin-bottom: 24rpx; }
.algorithm-list { display: flex; flex-direction: column; gap: 20rpx; margin-top: 16rpx; }

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

  &:active { transform: scale(0.98); }
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

.check-icon { flex-shrink: 0; }

.algorithm-desc {
  display: block;
  font-size: 26rpx;
  color: #6b7280;
  line-height: 1.5;
  margin-bottom: 12rpx;
}

.algorithm-meta { display: flex; gap: 16rpx; }
.meta-item { font-size: 22rpx; color: #9ca3af; background: #f3f4f6; padding: 4rpx 12rpx; border-radius: 8rpx; }

/* 加载/空/错误 */
.loading-container { display: flex; flex-direction: column; align-items: center; padding: 120rpx 0; }
.loading-text { margin-top: 24rpx; font-size: 28rpx; color: #9ca3af; }
.empty-state { padding: 80rpx 0; }

.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.error-text { font-size: 28rpx; color: #ef4444; margin-bottom: 24rpx; }
.retry-btn { padding: 16rpx 48rpx; background: #8b5cf6; color: #fff; border: none; border-radius: 16rpx; font-size: 28rpx; }

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
