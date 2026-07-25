<template>
  <PageLayout class="processing-page">
    <view class="main-content">
      <!-- 页面标题 -->
      <PageHeaderCard
        icon="setting"
        icon-color="#f59e0b"
        icon-bg="linear-gradient(135deg, #fef3c7, #fde68a)"
        title="去雾处理"
        :subtitle="statusText"
      />

      <!-- 处理信息区 -->
      <view class="info-section">
        <view class="info-row">
          <text class="info-label">选择算法</text>
          <text class="info-value">{{
            store.selectedAlgorithm?.name || "-"
          }}</text>
        </view>
        <view class="info-row">
          <text class="info-label">图片文件</text>
          <text class="info-value ellipsis">{{
            store.currentImage?.name || "-"
          }}</text>
        </view>
      </view>

      <!-- 参数调节面板 -->
      <view class="params-card">
        <view class="params-header">
          <text class="params-title">处理参数</text>
          <text class="params-reset" @click="resetParams">重置</text>
        </view>

        <view class="param-item">
          <view class="param-label">
            <text>去雾强度</text>
            <text class="param-value">{{ store.params.strength }}</text>
          </view>
          <slider
            :value="store.params.strength"
            :min="0"
            :max="100"
            :step="5"
            active-color="#f59e0b"
            block-size="20"
            @change="onStrengthChange"
          />
        </view>

        <view class="param-item">
          <view class="param-label">
            <text>色彩饱和度</text>
            <text class="param-value">{{ store.params.saturation }}</text>
          </view>
          <slider
            :value="store.params.saturation"
            :min="0"
            :max="200"
            :step="5"
            active-color="#f59e0b"
            block-size="20"
            @change="onSaturationChange"
          />
        </view>

        <view class="param-item">
          <view class="param-label">
            <text>对比度</text>
            <text class="param-value">{{ store.params.contrast }}</text>
          </view>
          <slider
            :value="store.params.contrast"
            :min="0"
            :max="200"
            :step="5"
            active-color="#f59e0b"
            block-size="20"
            @change="onContrastChange"
          />
        </view>

        <view class="param-item">
          <view class="param-label">
            <text>锐化程度</text>
            <text class="param-value">{{ store.params.sharpness }}</text>
          </view>
          <slider
            :value="store.params.sharpness"
            :min="0"
            :max="100"
            :step="5"
            active-color="#f59e0b"
            block-size="20"
            @change="onSharpnessChange"
          />
        </view>
      </view>

      <!-- 错误信息 -->
      <view v-if="store.errorMessage" class="error-card">
        <u-icon name="error-circle" size="20" color="#ef4444" />
        <text class="error-msg">{{ store.errorMessage }}</text>
      </view>

      <!-- 结果展示 -->
      <view v-if="store.isCompleted && store.result" class="result-section">
        <text class="section-label">处理结果</text>
        <view class="result-card">
          <image
            :src="store.result.resultUrl"
            class="result-image"
            mode="widthFix"
            @click="handlePreviewResult"
          />
          <view class="result-info">
            <text class="result-time">处理耗时: {{ store.result.time }}s</text>
            <text v-if="store.result.fromCache" class="cache-badge"
              >缓存命中</text
            >
          </view>
        </view>
      </view>
    </view>

    <!-- 底部操作栏 -->
    <view class="bottom-bar">
      <view class="bar-content">
        <!-- 处理前：开始按钮 -->
        <template
          v-if="store.status === 'algorithm' || store.status === 'failed'"
        >
          <button
            class="process-btn"
            :disabled="processing"
            @click="handleProcess"
          >
            {{ processing ? "处理中..." : "开始处理" }}
          </button>
        </template>

        <!-- 处理中 -->
        <template v-else-if="store.status === 'processing'">
          <view class="processing-indicator">
            <up-loading-icon mode="circle" size="24" color="#ffffff" />
            <text class="processing-text">去雾处理中...</text>
          </view>
        </template>

        <!-- 完成后 -->
        <template v-else-if="store.isCompleted">
          <button class="compare-btn" @click="handleCompare">
            查看对比效果
          </button>
        </template>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import { useProcessingStore, DEFAULT_DEHAZE_PARAMS } from "@/store/processing";
import { ModelAPI } from "dehaze-sdk-js";
import type { PredictionResultVO } from "dehaze-sdk-js";
import type { SliderChangeEvent } from "@/types/uni-events";
import { getErrorMessage } from "@/utils/error";

// ==================== 状态 ====================

const store = useProcessingStore();
const processing = ref(false);

// ==================== 计算属性 ====================

const statusText = computed(() => {
  const map: Record<string, string> = {
    idle: "等待开始",
    selected: "已选择图片",
    algorithm: "准备就绪",
    processing: "正在处理...",
    completed: "处理完成",
    failed: "处理失败",
  };
  return map[store.status] || store.status;
});

// ==================== 方法 ====================

type ParamKey = "strength" | "saturation" | "contrast" | "sharpness";

/** 更新单个参数 */
function updateParam(key: ParamKey, value: number) {
  store.updateParams({ [key]: value });
}

const onStrengthChange = (e: SliderChangeEvent) =>
  updateParam("strength", e.detail.value);
const onSaturationChange = (e: SliderChangeEvent) =>
  updateParam("saturation", e.detail.value);
const onContrastChange = (e: SliderChangeEvent) =>
  updateParam("contrast", e.detail.value);
const onSharpnessChange = (e: SliderChangeEvent) =>
  updateParam("sharpness", e.detail.value);

/** 重置参数 */
function resetParams() {
  store.updateParams({ ...DEFAULT_DEHAZE_PARAMS });
}

/** 构建预测参数：仅提交与默认值不同的参数 */
function buildPredictParams(): string | undefined {
  const paramsObj: Record<string, number> = {};
  (Object.keys(DEFAULT_DEHAZE_PARAMS) as ParamKey[]).forEach((key) => {
    if (store.params[key] !== DEFAULT_DEHAZE_PARAMS[key]) {
      paramsObj[key] = store.params[key];
    }
  });
  return Object.keys(paramsObj).length > 0
    ? JSON.stringify(paramsObj)
    : undefined;
}

/** 开始处理 */
async function handleProcess() {
  if (processing.value) return;
  if (!store.selectedAlgorithm || !store.currentImage) {
    uni.showToast({ title: "缺少图片或算法", icon: "none" });
    return;
  }

  processing.value = true;
  store.startProcessing();

  try {
    const result: PredictionResultVO = await ModelAPI.predictAndWait({
      algorithmId: store.selectedAlgorithm.id,
      fileId: store.fileId ?? undefined,
      imageUrl: !store.fileId ? store.currentImage.url : undefined,
      params: buildPredictParams(),
    });

    if (result.status === "failed") {
      throw new Error(result.errorMessage || "处理失败");
    }

    store.complete(result);

    uni.showToast({
      title: `处理完成，耗时${result.time ?? 0}s`,
      icon: "success",
      duration: 2000,
    });
  } catch (error) {
    const msg = getErrorMessage(error, "处理失败");
    store.fail(msg);
    uni.showToast({ title: msg, icon: "none", duration: 2500 });
  } finally {
    processing.value = false;
  }
}

/** 查看对比效果 */
function handleCompare() {
  uni.navigateTo({
    url: "/pages/side-by-side/index",
    fail: () => {
      uni.showToast({ title: "对比页面开发中", icon: "none" });
    },
  });
}

/** 预览结果大图 */
function handlePreviewResult() {
  if (!store.result?.resultUrl) return;
  uni.previewImage({
    urls: [store.result.resultUrl],
    current: store.result.resultUrl,
  });
}

// ==================== 生命周期 ====================

onMounted(() => {
  if (!store.hasAlgorithm) {
    uni.showToast({ title: "请先选择算法", icon: "none", duration: 2000 });
    setTimeout(() => uni.navigateBack(), 2000);
  }
});
</script>

<style lang="scss" scoped>
.processing-page {
  width: 100%;
  min-height: 100vh;
  background: #f9fafb;
}

.main-content {
  padding: 24rpx;
  padding-bottom: calc(180rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(180rpx + env(safe-area-inset-bottom));
}

/* 处理信息 */
.info-section {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 24rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16rpx 0;

  & + & {
    border-top: 1rpx solid #f3f4f6;
  }
}

.info-label {
  font-size: 26rpx;
  color: #6b7280;
}
.info-value {
  font-size: 28rpx;
  font-weight: 500;
  color: #1f2937;
}

.ellipsis {
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 参数面板 */
.params-card {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 28rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}

.params-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24rpx;
}

.params-title {
  font-size: 30rpx;
  font-weight: 600;
  color: #1f2937;
}
.params-reset {
  font-size: 24rpx;
  color: #f59e0b;
}

.param-item {
  margin-bottom: 28rpx;
}

.param-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12rpx;
  font-size: 26rpx;
  color: #374151;
}

.param-value {
  color: #f59e0b;
  font-weight: 600;
}

/* 错误卡片 */
.error-card {
  display: flex;
  align-items: center;
  gap: 12rpx;
  background: #fef2f2;
  border: 2rpx solid #fecaca;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 24rpx;
}

.error-msg {
  font-size: 26rpx;
  color: #ef4444;
  flex: 1;
}

/* 结果区域 */
.result-section {
  margin-bottom: 24rpx;
}
.section-label {
  font-size: 28rpx;
  font-weight: 600;
  color: #374151;
  margin-bottom: 16rpx;
  display: block;
}

.result-card {
  background: #ffffff;
  border-radius: 20rpx;
  overflow: hidden;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.08);
  border: 2rpx solid #10b981;
}

.result-image {
  width: 100%;
  display: block;
}

.result-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20rpx 24rpx;
}

.result-time {
  font-size: 26rpx;
  color: #6b7280;
}
.cache-badge {
  font-size: 22rpx;
  color: #10b981;
  background: #ecfdf5;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
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
  justify-content: center;
}

.process-btn {
  width: 100%;
  padding: 24rpx;
  background: linear-gradient(135deg, #f59e0b, #d97706);
  color: #ffffff;
  border: none;
  border-radius: 16rpx;
  font-size: 32rpx;
  font-weight: 700;

  &:disabled {
    background: #d1d5db;
    color: #9ca3af;
  }

  &:active:not(:disabled) {
    opacity: 0.85;
  }
}

.processing-indicator {
  display: flex;
  align-items: center;
  gap: 16rpx;
  padding: 24rpx 48rpx;
  background: linear-gradient(135deg, #f59e0b, #d97706);
  border-radius: 16rpx;
}

.processing-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #ffffff;
}

.compare-btn {
  width: 100%;
  padding: 24rpx;
  background: linear-gradient(135deg, #10b981, #059669);
  color: #ffffff;
  border: none;
  border-radius: 16rpx;
  font-size: 32rpx;
  font-weight: 700;

  &:active {
    opacity: 0.85;
  }
}
</style>
