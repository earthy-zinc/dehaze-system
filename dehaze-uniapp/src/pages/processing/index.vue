<template>
  <PageLayout level="L2" title="去雾处理" class="processing-page">
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

        <!-- 预设选择 -->
        <view v-if="presets.length > 0" class="preset-section">
          <text class="preset-label">参数预设</text>
          <scroll-view scroll-x class="preset-scroll" :show-scrollbar="false">
            <view
              v-for="preset in presets"
              :key="preset.id"
              class="preset-chip"
              @click="handleApplyPreset(preset)"
            >
              <text>{{ preset.name }}</text>
            </view>
          </scroll-view>
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
        <SvgIcon name="error-circle" size="20" color="#ef4444" />
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
            :disabled="store.isProcessing"
            @click="handleProcess"
          >
            {{ store.isProcessing ? "处理中..." : "开始处理" }}
          </button>
        </template>

        <!-- 处理中 -->
        <template v-else-if="store.status === 'processing'">
          <view class="processing-indicator">
            <view class="loading-spinner" />
            <text class="processing-text">去雾处理中...</text>
            <text class="processing-elapsed"
              >已用 {{ formatDuration(store.elapsedTime) }}</text
            >
          </view>
          <button class="cancel-btn" @click="handleCancel">取消处理</button>
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
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import { useProcessingStore, DEFAULT_DEHAZE_PARAMS } from "@/store/processing";
import { ModelAPI } from "dehaze-sdk-js";
import type { PresetVO } from "dehaze-sdk-js";
import type { SliderChangeEvent } from "@/types/uni-events";

// ==================== 状态 ====================

const store = useProcessingStore();
const presets = ref<PresetVO[]>([]);

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

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}秒`;
  return `${Math.floor(s / 60)}分${s % 60}秒`;
};

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
  if (store.isProcessing) return;
  if (!store.selectedAlgorithm || !store.currentImage) {
    uni.showToast({ title: "缺少图片或算法", icon: "none" });
    return;
  }

  // 确认对话框
  const confirmResult = await new Promise<boolean>((resolve) => {
    uni.showModal({
      title: "确认开始去雾处理",
      content: `图片：${store.currentImage!.name}\n算法：${store.selectedAlgorithm!.name}`,
      confirmText: "开始处理",
      cancelText: "取消",
      success: (res) => resolve(res.confirm),
      fail: () => resolve(false),
    });
  });
  if (!confirmResult) return;

  const res = await store.runPrediction({
    algorithmId: store.selectedAlgorithm.id,
    fileId: store.fileId ?? undefined,
    imageUrl: !store.fileId ? store.currentImage?.url : undefined,
    params: buildPredictParams(),
    onQuotaExhausted: ({ used, total }) => {
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
    },
  });

  if (res.ok && res.result) {
    uni.showToast({
      title: `处理完成，耗时${res.result.time ?? 0}s`,
      icon: "success",
      duration: 2000,
    });
  } else if (!res.ok && res.error) {
    uni.showToast({ title: res.error, icon: "none", duration: 2500 });
  }
}

/** 取消处理 */
function handleCancel() {
  store.cancelProcessing();
  uni.showToast({ title: "已取消处理", icon: "none" });
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

/** 应用预设 */
function handleApplyPreset(preset: PresetVO) {
  try {
    const p = JSON.parse(preset.params);
    store.updateParams({
      strength: p.strength ?? DEFAULT_DEHAZE_PARAMS.strength,
      saturation: p.saturation ?? DEFAULT_DEHAZE_PARAMS.saturation,
      contrast: p.contrast ?? DEFAULT_DEHAZE_PARAMS.contrast,
      sharpness: p.sharpen ?? DEFAULT_DEHAZE_PARAMS.sharpness,
    });
  } catch {
    uni.showToast({ title: "预设参数解析失败", icon: "none" });
  }
}

// ==================== 生命周期 ====================

onMounted(() => {
  if (!store.hasAlgorithm) {
    uni.showToast({ title: "请先选择算法", icon: "none", duration: 2000 });
    setTimeout(() => uni.navigateBack(), 2000);
  }
  ModelAPI.getPresets({ pageNum: 1, pageSize: 50 })
    .then((res) => {
      presets.value = res.list || [];
    })
    .catch(() => {
      /* 静默 */
    });
});
</script>

<style lang="scss" scoped>
@import "@/styles/mixins.scss";

.processing-page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: 24rpx;
  @include safe-area-bottom(180rpx);
}

/* 处理信息 */
.info-section {
  background: $color-white;
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
    border-top: 1rpx solid $color-border-light;
  }
}

.info-label {
  font-size: 26rpx;
  color: $color-text-secondary;
}
.info-value {
  font-size: 28rpx;
  font-weight: 500;
  color: $color-text-primary;
}

.ellipsis {
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 参数面板 */
.params-card {
  background: $color-white;
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
  color: $color-text-primary;
}
.params-reset {
  font-size: 24rpx;
  color: $color-warning;
}

.preset-section {
  margin-bottom: 28rpx;

  .preset-label {
    font-size: 24rpx;
    color: $color-text-secondary;
    margin-bottom: 12rpx;
    display: block;
  }

  .preset-scroll {
    white-space: nowrap;
  }

  .preset-chip {
    display: inline-block;
    padding: 10rpx 24rpx;
    margin-right: 12rpx;
    font-size: 24rpx;
    color: $color-warning;
    background: $color-warning-bg;
    border-radius: 24rpx;

    &:active {
      background: #fef3c7;
    }
  }
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
  color: $color-warning;
  font-weight: 600;
}

/* 错误卡片 */
.error-card {
  display: flex;
  align-items: center;
  gap: 12rpx;
  background: $color-danger-bg;
  border: 2rpx solid #fecaca;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 24rpx;
}

.error-msg {
  font-size: 26rpx;
  color: $color-danger;
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
  background: $color-white;
  border-radius: 20rpx;
  overflow: hidden;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.08);
  border: 2rpx solid $color-success;
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
  color: $color-text-secondary;
}
.cache-badge {
  font-size: 22rpx;
  color: $color-success;
  background: $color-success-bg;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}

/* 底部操作栏 */
.bottom-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: $color-white;
  border-top: 1rpx solid $color-border-light;
  padding: 20rpx 32rpx;
  @include safe-area-bottom(20rpx);
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
  background: linear-gradient(135deg, $color-warning, #d97706);
  color: $color-white;
  border: none;
  border-radius: 16rpx;
  font-size: 32rpx;
  font-weight: 700;

  &:disabled {
    background: $color-text-disabled;
    color: $color-text-placeholder;
  }

  &:active:not(:disabled) {
    opacity: 0.85;
  }
}

.loading-spinner {
  width: 32rpx;
  height: 32rpx;
  border: 3rpx solid rgba(255, 255, 255, 0.3);
  border-top-color: $color-white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.processing-indicator {
  display: flex;
  align-items: center;
  gap: 16rpx;
  padding: 24rpx 48rpx;
  background: linear-gradient(135deg, $color-warning, #d97706);
  border-radius: 16rpx;
}

.processing-text {
  font-size: 30rpx;
  font-weight: 600;
  color: $color-white;
}

.processing-elapsed {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.8);
  margin-left: auto;
}

.cancel-btn {
  width: 100%;
  margin-top: 16rpx;
  padding: 20rpx;
  background: $color-bg-secondary;
  color: $color-text-secondary;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;

  &:active {
    background: $color-border;
  }
}

.compare-btn {
  width: 100%;
  padding: 24rpx;
  background: linear-gradient(135deg, $color-success, #059669);
  color: $color-white;
  border: none;
  border-radius: 16rpx;
  font-size: 32rpx;
  font-weight: 700;

  &:active {
    opacity: 0.85;
  }
}
</style>
