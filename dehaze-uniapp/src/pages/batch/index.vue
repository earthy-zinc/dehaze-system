<template>
  <PageLayout level="L2" title="批量处理" class="batch-page">
    <view class="main-content">
      <!-- 页面标题 -->
      <PageHeaderCard
        icon="photo"
        icon-color="#f59e0b"
        icon-bg="linear-gradient(135deg, #fef3c7, #fde68a)"
        title="批量处理"
        :subtitle="subtitleText"
      />

      <!-- 上传区域 -->
      <view class="section">
        <view class="section-header">
          <text class="section-title">选择图片</text>
          <text class="section-hint"
            >最多{{ MAX_IMAGES }}张，已选{{ images.length }}张</text
          >
        </view>

        <view class="image-grid">
          <view v-for="img in images" :key="img.id" class="image-item">
            <image :src="img.localPath" class="image-thumb" mode="aspectFill" />
            <view class="image-remove" @click="removeImage(img.id)">
              <SvgIcon name="close" size="12" color="#fff" />
            </view>
            <view
              v-if="img.status === 'processing'"
              class="image-status processing"
            >
              <view class="loading-spinner" />
            </view>
            <view
              v-else-if="img.status === 'completed'"
              class="image-status completed"
            >
              <text>✓</text>
            </view>
            <view
              v-else-if="img.status === 'failed'"
              class="image-status failed"
            >
              <text>!</text>
            </view>
          </view>

          <view
            v-if="images.length < MAX_IMAGES"
            class="image-add"
            @click="handleChooseImages"
          >
            <SvgIcon name="plus" size="32" color="#3b82f6" />
            <text class="add-text">添加图片</text>
          </view>
        </view>
      </view>

      <!-- 算法选择 -->
      <view class="section">
        <view class="section-header">
          <text class="section-title">选择算法</text>
        </view>

        <view v-if="algoLoading" class="algo-loading">
          <view class="loading-spinner" />
          <text class="algo-loading-text">加载算法中...</text>
        </view>

        <view v-else class="algo-list">
          <view
            v-for="algo in availableAlgos"
            :key="algo.id"
            class="algo-card"
            :class="{ selected: selectedAlgoId === algo.id }"
            @click="selectedAlgoId = algo.id"
          >
            <text class="algo-name">{{ algo.name }}</text>
            <text v-if="algo.description" class="algo-desc">{{
              algo.description
            }}</text>
            <view v-if="selectedAlgoId === algo.id" class="algo-check">✓</view>
          </view>
        </view>
      </view>

      <!-- 参数（可选） -->
      <view class="section">
        <view class="section-header">
          <text class="section-title">参数（可选）</text>
        </view>
        <view class="params-input-wrapper">
          <textarea
            class="params-input"
            placeholder='JSON参数，如 {"strength":80}'
            :value="params"
            @input="(e: any) => (params = e.detail.value)"
          />
        </view>
      </view>

      <!-- 处理进度 -->
      <view v-if="processing && images.length > 0" class="section">
        <view class="section-header">
          <text class="section-title">处理进度</text>
        </view>
        <view class="progress-card">
          <view class="progress-bar-track">
            <view
              class="progress-bar-fill"
              :style="{ width: totalProgress + '%' }"
            />
          </view>
          <text class="progress-text">
            已完成 {{ completedCount }} / 失败 {{ failedCount }} / 总计
            {{ images.length }}
          </text>
        </view>
      </view>

      <!-- 处理结果 -->
      <view
        v-if="!processing && (completedCount > 0 || failedCount > 0)"
        class="section"
      >
        <view class="section-header">
          <text class="section-title">处理结果</text>
        </view>
        <view class="result-list">
          <view
            v-for="img in finishedImages"
            :key="img.id"
            class="result-item"
            :class="img.status"
          >
            <image
              :src="img.localPath"
              class="result-thumb"
              mode="aspectFill"
            />
            <view class="result-info">
              <template v-if="img.status === 'completed'">
                <text class="result-status success">处理完成</text>
                <text v-if="img.time != null" class="result-time"
                  >耗时 {{ img.time }}ms</text
                >
                <view class="result-actions">
                  <button
                    class="result-btn"
                    @click="handlePreview(img.resultUrl!)"
                  >
                    查看结果
                  </button>
                </view>
              </template>
              <template v-else>
                <text class="result-status error">处理失败</text>
                <text v-if="img.errorMessage" class="result-error">{{
                  img.errorMessage
                }}</text>
                <button class="result-btn retry" @click="handleRetryImage(img)">
                  重试
                </button>
              </template>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 底部操作栏 -->
    <view class="bottom-bar">
      <button
        class="process-btn"
        :loading="processing"
        :disabled="images.length === 0 || !selectedAlgoId || processing"
        @click="handleStartBatch"
      >
        {{ processing ? "处理中..." : "开始批量处理" }}
      </button>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { AlgorithmAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import { useProcessingStore } from "@/store/processing";

// ==================== 类型 ====================

interface BatchItem {
  id: string;
  localPath: string;
  status: "pending" | "processing" | "completed" | "failed";
  resultUrl?: string;
  errorMessage?: string;
  time?: number;
  logId?: number;
}

// ==================== 常量 ====================

const MAX_IMAGES = 20;

// ==================== 状态 ====================

const store = useProcessingStore();
const images = ref<BatchItem[]>([]);
const algorithms = ref<Algorithm[]>([]);
const algoLoading = ref(false);
const selectedAlgoId = ref<number | null>(null);
const params = ref("");
const processing = ref(false);

/** 配额耗尽弹窗（去充值） */
function showQuotaExhaustedModal({
  used,
  total,
}: {
  used: number;
  total: number;
}) {
  uni.showModal({
    title: "预测次数不足",
    content: `当前剩余预测次数为 0（已使用 ${used}/${total}），请及时充值。`,
    confirmText: "去充值",
    cancelText: "取消",
    success: (r) => {
      if (r.confirm) {
        uni.navigateTo({ url: "/pages/personal/quota/index" });
      }
    },
  });
}

// ==================== 计算属性 ====================

const subtitleText = computed(() => {
  if (processing.value) return "批量处理进行中...";
  return "批量上传图片，选择算法一键处理";
});

const availableAlgos = computed(() =>
  algorithms.value.filter((a) => a.status === 4)
);

const completedCount = computed(
  () => images.value.filter((i) => i.status === "completed").length
);
const failedCount = computed(
  () => images.value.filter((i) => i.status === "failed").length
);

const totalProgress = computed(() => {
  if (images.value.length === 0) return 0;
  return Math.round(
    ((completedCount.value + failedCount.value) / images.value.length) * 100
  );
});

const finishedImages = computed(() =>
  images.value.filter((i) => i.status === "completed" || i.status === "failed")
);

// ==================== 方法 ====================

/** 选择图片 */
function handleChooseImages() {
  const remain = MAX_IMAGES - images.value.length;
  if (remain <= 0) {
    uni.showToast({ title: `最多${MAX_IMAGES}张图片`, icon: "none" });
    return;
  }
  uni.chooseImage({
    count: remain,
    sizeType: ["compressed"],
    sourceType: ["album", "camera"],
    success: (res) => {
      const paths = Array.isArray(res.tempFilePaths)
        ? res.tempFilePaths
        : [res.tempFilePaths];
      const newImages: BatchItem[] = paths.map((path, idx) => ({
        id: `${Date.now()}_${idx}`,
        localPath: path,
        status: "pending",
      }));
      images.value = [...images.value, ...newImages];
    },
  });
}

/** 移除图片 */
function removeImage(id: string) {
  images.value = images.value.filter((img) => img.id !== id);
}

/** 开始批量处理：逐张调用 store.runPrediction，复用配额校验/重试/计时 */
async function handleStartBatch() {
  if (!selectedAlgoId.value) {
    uni.showToast({ title: "请先选择算法", icon: "none" });
    return;
  }
  if (images.value.length === 0) {
    uni.showToast({ title: "请先上传图片", icon: "none" });
    return;
  }

  processing.value = true;
  const batch = images.value.map((img) => ({
    ...img,
    status: "pending" as const,
  }));

  for (const img of batch) {
    images.value = images.value.map((item) =>
      item.id === img.id ? { ...item, status: "processing" as const } : item
    );

    const res = await store.runPrediction({
      algorithmId: selectedAlgoId.value,
      imageUrl: img.localPath,
      params: params.value || undefined,
      onQuotaExhausted: showQuotaExhaustedModal,
    });

    if (res.ok && res.result) {
      images.value = images.value.map((item) =>
        item.id === img.id
          ? {
              ...item,
              status: "completed" as const,
              resultUrl: res.result!.resultUrl,
              time: res.result!.time,
            }
          : item
      );
    } else if (!res.ok && res.error) {
      images.value = images.value.map((item) =>
        item.id === img.id
          ? { ...item, status: "failed" as const, errorMessage: res.error }
          : item
      );
    } else {
      // 配额耗尽或已取消，停止批量处理
      images.value = images.value.map((item) =>
        item.id === img.id
          ? { ...item, status: "failed" as const, errorMessage: "已取消" }
          : item
      );
      break;
    }
  }

  processing.value = false;
}

/** 重试单张图片 */
async function handleRetryImage(img: BatchItem) {
  if (!selectedAlgoId.value) return;
  images.value = images.value.map((item) =>
    item.id === img.id ? { ...item, status: "processing" as const } : item
  );

  const res = await store.runPrediction({
    algorithmId: selectedAlgoId.value,
    imageUrl: img.localPath,
    params: params.value || undefined,
    onQuotaExhausted: showQuotaExhaustedModal,
  });

  if (res.ok && res.result) {
    images.value = images.value.map((item) =>
      item.id === img.id
        ? {
            ...item,
            status: "completed" as const,
            resultUrl: res.result!.resultUrl,
            time: res.result!.time,
          }
        : item
    );
  } else {
    images.value = images.value.map((item) =>
      item.id === img.id
        ? {
            ...item,
            status: "failed" as const,
            errorMessage: res.error || "已取消",
          }
        : item
    );
  }
}

/** 预览结果图 */
function handlePreview(url: string) {
  uni.previewImage({ urls: [url], current: url });
}

/** 加载算法 */
async function loadAlgorithms() {
  algoLoading.value = true;
  try {
    const data = await AlgorithmAPI.getList();
    algorithms.value = data || [];
  } catch {
    uni.showToast({ title: "加载算法失败", icon: "none" });
  } finally {
    algoLoading.value = false;
  }
}

onMounted(() => {
  loadAlgorithms();
});
</script>

<style lang="scss" scoped>
.batch-page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: $spacing-md;
  padding-bottom: calc(160rpx + $safe-area-bottom-env);
}

.section {
  margin-bottom: $spacing-md;
  background: $color-white;
  border-radius: $radius-xl;
  padding: $spacing-lg;
  box-shadow: $shadow-sm;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: $spacing-md;
}

.section-title {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
}

.section-hint {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

/* 图片网格 */
.image-grid {
  display: flex;
  flex-wrap: wrap;
  gap: $spacing-sm;
}

.image-item {
  position: relative;
  width: calc(25% - 12rpx);
  aspect-ratio: 1;
  border-radius: $radius-md;
  overflow: hidden;
  background: $color-bg-secondary;
}

.image-thumb {
  width: 100%;
  height: 100%;
}

.image-remove {
  position: absolute;
  top: 4rpx;
  right: 4rpx;
  width: 36rpx;
  height: 36rpx;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-status {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 4rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: $font-xs;
  color: $color-white;

  &.processing {
    background: rgba(245, 158, 11, 0.8);
  }
  &.completed {
    background: rgba(16, 185, 129, 0.8);
  }
  &.failed {
    background: rgba(239, 68, 68, 0.8);
  }
}

.image-add {
  width: calc(25% - 12rpx);
  aspect-ratio: 1;
  border: 2rpx dashed $color-text-disabled;
  border-radius: $radius-md;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  background: $color-bg-secondary;

  &:active {
    opacity: 0.7;
  }
}

.add-text {
  font-size: $font-xs;
  color: $color-primary;
}

/* 算法选择 */
.algo-loading {
  display: flex;
  align-items: center;
  gap: 12rpx;
  padding: 24rpx;
  justify-content: center;
}

.algo-loading-text {
  font-size: $font-sm;
  color: $color-text-placeholder;
}

.loading-spinner {
  width: 24rpx;
  height: 24rpx;
  border: 3rpx solid rgba(245, 158, 11, 0.3);
  border-top-color: $color-warning;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.algo-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.algo-card {
  display: flex;
  flex-direction: column;
  background: $color-bg-secondary;
  border-radius: $radius-lg;
  padding: 24rpx;
  border: 2rpx solid transparent;
  position: relative;

  &.selected {
    border-color: $color-warning;
    background: $color-warning-bg;
  }

  &:active {
    opacity: 0.85;
  }
}

.algo-name {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  margin-bottom: 8rpx;
}

.algo-desc {
  font-size: 24rpx;
  color: $color-text-secondary;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.algo-check {
  position: absolute;
  top: 16rpx;
  right: 16rpx;
  color: $color-warning;
  font-size: $font-lg;
  font-weight: 700;
}

/* 参数输入 */
.params-input-wrapper {
  background: $color-bg-secondary;
  border-radius: $radius-md;
  padding: $spacing-sm;
}

.params-input {
  width: 100%;
  min-height: 120rpx;
  font-size: $font-sm;
  color: $color-text-primary;
  background: transparent;
}

/* 进度 */
.progress-card {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.progress-bar-track {
  width: 100%;
  height: 12rpx;
  background: $color-bg-secondary;
  border-radius: 6rpx;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(135deg, $color-warning, #d97706);
  border-radius: 6rpx;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: $font-xs;
  color: $color-text-secondary;
  text-align: center;
}

/* 结果列表 */
.result-list {
  display: flex;
  flex-direction: column;
  gap: $spacing-sm;
}

.result-item {
  display: flex;
  gap: $spacing-sm;
  padding: $spacing-sm;
  background: $color-bg-secondary;
  border-radius: $radius-md;

  &.completed {
    background: $color-success-bg;
  }
  &.failed {
    background: $color-danger-bg;
  }
}

.result-thumb {
  width: 120rpx;
  height: 120rpx;
  border-radius: $radius-md;
  flex-shrink: 0;
}

.result-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8rpx;
}

.result-status {
  font-size: $font-sm;
  font-weight: 600;

  &.success {
    color: $color-success;
  }
  &.error {
    color: $color-danger;
  }
}

.result-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.result-error {
  font-size: $font-xs;
  color: $color-danger;
}

.result-actions {
  display: flex;
  gap: $spacing-sm;
  margin-top: 4rpx;
}

.result-btn {
  padding: 8rpx 24rpx;
  background: $color-primary;
  color: $color-white;
  border: none;
  border-radius: $radius-md;
  font-size: $font-xs;

  &.retry {
    background: $color-danger;
  }

  &:active {
    opacity: 0.85;
  }
}

/* 底部操作栏 */
.bottom-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: $color-white;
  border-top: 1rpx solid $color-border-light;
  padding: $spacing-md $spacing-lg;
  padding-bottom: calc($spacing-md + $safe-area-bottom-env);
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.04);
  z-index: 100;
}

.process-btn {
  width: 100%;
  padding: 24rpx;
  background: linear-gradient(135deg, $color-warning, #d97706);
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-lg;
  font-weight: 700;

  &:disabled {
    background: $color-text-disabled;
    color: $color-text-placeholder;
  }

  &:active:not(:disabled) {
    opacity: 0.85;
  }
}

@media (max-width: 375px) {
  .image-item,
  .image-add {
    width: calc(33.33% - 11rpx);
  }
}
</style>
