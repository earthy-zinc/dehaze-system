<template>
  <PageLayout level="L1" title="去雾">
    <view class="dehaze-page">
      <!-- 步骤指示器 -->
      <view class="dehaze-steps">
        <view
          v-for="(step, i) in steps"
          :key="step.key"
          class="dehaze-step"
          @click="goStep(i)"
        >
          <view
            class="dehaze-step-dot"
            :class="{ done: i < currentStep, active: i === currentStep }"
          >
            <text v-if="i < currentStep">✓</text>
            <text v-else>{{ i + 1 }}</text>
          </view>
          <text
            class="dehaze-step-label"
            :class="{ 'active-label': i === currentStep }"
          >
            {{ step.label }}
          </text>
        </view>
      </view>

      <!-- 步骤内容区 -->
      <view class="dehaze-flow-body">
        <!-- 步骤1：上传图像 -->
        <view v-if="currentStep === 0" class="dehaze-step-content">
          <view class="dehaze-upload-area" @click="handleChooseImage">
            <image
              v-if="currentImage"
              :src="currentImage.url"
              class="dehaze-upload-preview"
              mode="aspectFit"
            />
            <view v-else class="dehaze-upload-placeholder">
              <SvgIcon name="camera" size="48" color="#9ca3af" />
              <text class="dehaze-upload-text">点击选择图片</text>
              <text class="dehaze-upload-hint">支持相册或拍照</text>
            </view>
          </view>
          <view v-if="currentImage" class="dehaze-image-info">
            <text class="dehaze-info-item">{{ currentImage.name }}</text>
            <text class="dehaze-info-item">
              {{ currentImage.width }}×{{ currentImage.height }}
            </text>
            <text class="dehaze-info-item">{{
              formatSize(currentImage.size)
            }}</text>
          </view>
          <view class="dehaze-step-action">
            <view
              class="dehaze-btn dehaze-btn-primary"
              :class="{ disabled: !currentImage }"
              @click="currentImage && goStep(1)"
            >
              <text>下一步：选择算法</text>
              <SvgIcon name="arrow-right" size="16" color="#fff" />
            </view>
          </view>
        </view>

        <!-- 步骤2：选择算法 -->
        <view v-if="currentStep === 1" class="dehaze-step-content">
          <view class="dehaze-algo-search">
            <SvgIcon name="search" size="16" color="#9ca3af" />
            <input
              class="dehaze-algo-search-input"
              placeholder="搜索算法..."
              :value="algoSearch"
              @input="(e: any) => (algoSearch = e.detail.value)"
            />
          </view>
          <view v-if="algoLoading" class="dehaze-loading">
            <text>加载算法中...</text>
          </view>
          <view v-else class="dehaze-algo-list">
            <view
              v-for="algo in filteredAlgorithms"
              :key="algo.id"
              class="dehaze-algo-card"
              :class="{ selected: selectedAlgorithm?.id === algo.id }"
              @click="handleSelectAlgorithm(algo)"
            >
              <text class="dehaze-algo-name">{{ algo.name }}</text>
              <view v-if="algo.type" class="dehaze-algo-type">
                <text>{{ algo.type }}</text>
              </view>
              <text v-if="algo.description" class="dehaze-algo-desc">
                {{ algo.description }}
              </text>
            </view>
          </view>
          <view class="dehaze-step-action">
            <view class="dehaze-btn dehaze-btn-secondary" @click="goStep(0)">
              返回上一步
            </view>
            <view
              class="dehaze-btn dehaze-btn-primary"
              :class="{ disabled: !selectedAlgorithm }"
              @click="selectedAlgorithm && goStep(2)"
            >
              <text>下一步：调节参数</text>
              <SvgIcon name="arrow-right" size="16" color="#fff" />
            </view>
          </view>
        </view>

        <!-- 步骤3：调节参数 -->
        <view v-if="currentStep === 2" class="dehaze-step-content">
          <view class="dehaze-params-section">
            <!-- 预设选择 -->
            <view v-if="presets.length > 0" class="dehaze-preset-section">
              <text class="dehaze-preset-label">参数预设</text>
              <scroll-view scroll-x class="dehaze-preset-scroll" :show-scrollbar="false">
                <view
                  v-for="preset in presets"
                  :key="preset.id"
                  class="dehaze-preset-chip"
                  @click="handleApplyPreset(preset)"
                >
                  <text>{{ preset.name }}</text>
                </view>
              </scroll-view>
            </view>

            <view class="dehaze-param-item">
              <view class="dehaze-param-header">
                <text class="dehaze-param-label">去雾强度</text>
                <text class="dehaze-param-value">{{ params.strength }}</text>
              </view>
              <slider
                :min="0"
                :max="100"
                :value="params.strength"
                @change="
                  (e: { detail: { value: number } }) =>
                    (params.strength = e.detail.value)
                "
              />
            </view>
            <view class="dehaze-param-item">
              <view class="dehaze-param-header">
                <text class="dehaze-param-label">色彩饱和度</text>
                <text class="dehaze-param-value">{{ params.saturation }}</text>
              </view>
              <slider
                :min="0"
                :max="200"
                :value="params.saturation"
                @change="
                  (e: { detail: { value: number } }) =>
                    (params.saturation = e.detail.value)
                "
              />
            </view>
            <view class="dehaze-param-item">
              <view class="dehaze-param-header">
                <text class="dehaze-param-label">对比度</text>
                <text class="dehaze-param-value">{{ params.contrast }}</text>
              </view>
              <slider
                :min="0"
                :max="200"
                :value="params.contrast"
                @change="
                  (e: { detail: { value: number } }) =>
                    (params.contrast = e.detail.value)
                "
              />
            </view>
            <view class="dehaze-param-item">
              <view class="dehaze-param-header">
                <text class="dehaze-param-label">锐化程度</text>
                <text class="dehaze-param-value">{{ params.sharpness }}</text>
              </view>
              <slider
                :min="0"
                :max="100"
                :value="params.sharpness"
                @change="
                  (e: { detail: { value: number } }) =>
                    (params.sharpness = e.detail.value)
                "
              />
            </view>
            <view class="dehaze-param-reset" @click="resetParams">
              <text>恢复默认</text>
            </view>
          </view>
          <view v-if="selectedAlgorithm" class="dehaze-algo-summary">
            <text class="dehaze-summary-label">已选算法：</text>
            <text class="dehaze-summary-value">{{
              selectedAlgorithm.name
            }}</text>
          </view>
          <view class="dehaze-step-action">
            <view class="dehaze-btn dehaze-btn-secondary" @click="goStep(1)">
              返回上一步
            </view>
            <view class="dehaze-btn dehaze-btn-primary" @click="handleProcess">
              <text>开始去雾</text>
              <SvgIcon name="arrow-right" size="16" color="#fff" />
            </view>
          </view>
        </view>

        <!-- 步骤4：处理中 -->
        <view v-if="currentStep === 3" class="dehaze-step-content">
          <view
            v-if="processStatus === 'processing'"
            class="dehaze-processing-status"
          >
            <view class="dehaze-spinner" />
            <text class="dehaze-status-text">正在去雾处理中...</text>
            <text class="dehaze-status-hint"
              >已用 {{ formatDuration(elapsedTime) }}</text
            >
            <view class="dehaze-step-action">
              <view
                class="dehaze-btn dehaze-btn-secondary"
                @click="handleCancelProcess"
              >
                取消处理
              </view>
            </view>
          </view>
          <view
            v-if="processStatus === 'success' && result"
            class="dehaze-success-status"
          >
            <view class="dehaze-success-icon">✓</view>
            <text class="dehaze-status-text">处理完成</text>
            <text class="dehaze-status-hint">
              耗时 {{ formatDuration(result.time ?? 0) }}
              <text v-if="result.fromCache"> · 缓存命中</text>
            </text>
            <view v-if="result.resultUrl" class="dehaze-result-preview">
              <image
                :src="result.resultUrl"
                class="dehaze-result-img"
                mode="aspectFit"
              />
            </view>
            <view class="dehaze-step-action">
              <view
                class="dehaze-btn dehaze-btn-primary"
                @click="handleGoCompare"
              >
                <text>进入效果对比</text>
                <SvgIcon name="arrow-right" size="16" color="#fff" />
              </view>
              <view
                class="dehaze-btn dehaze-btn-secondary"
                @click="handleReset"
              >
                重新开始
              </view>
            </view>
          </view>
          <view v-if="processStatus === 'error'" class="dehaze-error-status">
            <view class="dehaze-error-icon">!</view>
            <text class="dehaze-status-text">处理失败</text>
            <text class="dehaze-status-hint">{{ errorMsg }}</text>
            <view class="dehaze-step-action">
              <view class="dehaze-btn dehaze-btn-primary" @click="handleProcess"
                >重试</view
              >
              <view class="dehaze-btn dehaze-btn-secondary" @click="goStep(2)"
                >调整参数</view
              >
            </view>
          </view>
        </view>

        <!-- 步骤5：对比入口 -->
        <view v-if="currentStep === 4" class="dehaze-step-content">
          <view class="dehaze-compare-entry">
            <view class="dehaze-compare-icon">
              <text>⟷</text>
            </view>
            <text class="dehaze-compare-title">效果对比</text>
            <text class="dehaze-compare-desc">
              查看处理前后的对比效果，支持并排、重叠、放大镜等多种模式
            </text>
            <view class="dehaze-step-action">
              <view
                class="dehaze-btn dehaze-btn-primary"
                @click="handleGoCompare"
              >
                <text>进入效果对比</text>
                <SvgIcon name="arrow-right" size="16" color="#fff" />
              </view>
              <view
                class="dehaze-btn dehaze-btn-secondary"
                @click="handleReset"
              >
                开始新的去雾
              </view>
            </view>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { AlgorithmAPI, ModelAPI } from "dehaze-sdk-js";
import type { Algorithm, PredictionResultVO, PresetVO } from "dehaze-sdk-js";
import { uploadImage } from "@/api/file";
import { getErrorMessage } from "@/utils/error";

interface StepDef {
  key: string;
  label: string;
}

interface ImageData {
  url: string;
  name: string;
  width: number;
  height: number;
  size: number;
}

interface ProcessParams {
  strength: number;
  saturation: number;
  contrast: number;
  sharpness: number;
}

const steps: StepDef[] = [
  { key: "upload", label: "上传图像" },
  { key: "algorithm", label: "选择算法" },
  { key: "params", label: "调节参数" },
  { key: "processing", label: "处理" },
  { key: "compare", label: "效果对比" },
];

const DEFAULT_PARAMS: ProcessParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpness: 30,
};

type ProcessStatus = "idle" | "processing" | "success" | "error";

/** 重试间隔（毫秒）：2s → 5s → 10s */
const RETRY_DELAYS = [2000, 5000, 10000];
const MAX_RETRIES = 3;

const currentStep = ref(0);
const currentImage = ref<ImageData | null>(null);
const algorithms = ref<Algorithm[]>([]);
const algoLoading = ref(false);
const algoSearch = ref("");
const selectedAlgorithm = ref<Algorithm | null>(null);
const params = ref<ProcessParams>({ ...DEFAULT_PARAMS });
const processStatus = ref<ProcessStatus>("idle");
const result = ref<PredictionResultVO | null>(null);
const errorMsg = ref("");
const elapsedTime = ref(0);
const uploadedFileId = ref<number | null>(null);
const cancelled = ref(false);
const presets = ref<PresetVO[]>([]);
let elapsedTimer: ReturnType<typeof setInterval> | null = null;

const clearTimer = () => {
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
};

onUnmounted(clearTimer);

onMounted(() => {
  if (currentStep.value === 1 && algorithms.value.length === 0) {
    loadAlgorithms();
  }
  ModelAPI.getPresets({ pageNum: 1, pageSize: 50 })
    .then((res) => { presets.value = res.list || []; })
    .catch(() => { /* 静默 */ });
});

const loadAlgorithms = async () => {
  algoLoading.value = true;
  try {
    const list = await AlgorithmAPI.getList();
    algorithms.value = list || [];
  } catch {
    uni.showToast({ title: "加载算法失败", icon: "none" });
  } finally {
    algoLoading.value = false;
  }
};

/** 收集叶子算法 */
const collectLeaves = (nodes: Algorithm[]): Algorithm[] => {
  const leaves: Algorithm[] = [];
  const walk = (list: Algorithm[]) => {
    for (const node of list) {
      if (node.children && node.children.length > 0) {
        walk(node.children);
      } else {
        leaves.push(node);
      }
    }
  };
  walk(nodes);
  return leaves;
};

const leafAlgorithms = computed(() =>
  collectLeaves(algorithms.value).filter((a) => a.status === 4)
);

const filteredAlgorithms = computed(() => {
  if (!algoSearch.value) return leafAlgorithms.value;
  const kw = algoSearch.value.toLowerCase();
  return leafAlgorithms.value.filter(
    (a) =>
      a.name?.toLowerCase().includes(kw) ||
      a.description?.toLowerCase().includes(kw)
  );
});

const goStep = (step: number) => {
  if (step < 0 || step >= steps.length) return;
  if (step > currentStep.value) {
    if (step >= 1 && !currentImage.value) {
      uni.showToast({ title: "请先上传图片", icon: "none" });
      return;
    }
    if (step >= 2 && !selectedAlgorithm.value) {
      uni.showToast({ title: "请先选择算法", icon: "none" });
      return;
    }
  }
  // 切换到算法选择步骤时加载算法
  if (step === 1 && algorithms.value.length === 0 && !algoLoading.value) {
    loadAlgorithms();
  }
  currentStep.value = step;
};

/** 选择图片 */
const handleChooseImage = () => {
  uni.chooseMedia({
    count: 1,
    mediaType: ["image"],
    sourceType: ["album", "camera"],
    sizeType: ["original", "compressed"],
    success: (res) => {
      const file = res.tempFiles[0];
      if (!file) return;
      uni.getImageInfo({
        src: file.tempFilePath,
        success: (info) => {
          currentImage.value = {
            url: file.tempFilePath,
            name: `图片_${Date.now()}`,
            width: info.width,
            height: info.height,
            size: file.size,
          };
          uni.showToast({ title: "上传成功", icon: "success" });
        },
        fail: () => {
          uni.showToast({ title: "获取图片信息失败", icon: "none" });
        },
      });
    },
    fail: (err: { errMsg?: string }) => {
      if (!err.errMsg?.includes("cancel")) {
        uni.showToast({ title: "选择图片失败", icon: "none" });
      }
    },
  });
};

const handleSelectAlgorithm = (algorithm: Algorithm) => {
  selectedAlgorithm.value = algorithm;
  goStep(2);
};

const handleProcess = async () => {
  if (!currentImage.value || !selectedAlgorithm.value) return;

  // 配额检查
  try {
    const quota = await ModelAPI.getQuota();
    if (quota.remaining === 0) {
      uni.showModal({
        title: "预测次数不足",
        content: `当前剩余预测次数为 0（已使用 ${quota.used}/${quota.total}），请及时充值。`,
        confirmText: "去充值",
        cancelText: "取消",
        success: (res) => {
          if (res.confirm) {
            uni.navigateTo({ url: "/pages/user-center/index" });
          }
        },
      });
      return;
    }
  } catch {
    // 配额查询失败也允许继续
  }

  // 确认对话框
  const confirmResult = await new Promise<boolean>((resolve) => {
    uni.showModal({
      title: "确认开始去雾处理",
      content: `图片：${currentImage.value!.name}\n尺寸：${currentImage.value!.width}×${currentImage.value!.height}\n算法：${selectedAlgorithm.value!.name}`,
      confirmText: "开始处理",
      cancelText: "取消",
      success: (res) => resolve(res.confirm),
      fail: () => resolve(false),
    });
  });
  if (!confirmResult) return;

  processStatus.value = "processing";
  errorMsg.value = "";
  result.value = null;
  cancelled.value = false;
  elapsedTime.value = 0;
  elapsedTimer = setInterval(() => {
    elapsedTime.value += 100;
  }, 100);

  const attempt = async (attemptNumber: number): Promise<void> => {
    if (cancelled.value) return;

    try {
      if (!uploadedFileId.value) {
        const fileInfo = await uploadImage({ url: currentImage.value!.url });
        uploadedFileId.value = fileInfo.id;
      }

      const res = await ModelAPI.predictAndWait({
        algorithmId: selectedAlgorithm.value!.id,
        fileId: uploadedFileId.value,
        params: JSON.stringify(params.value),
      });

      if (cancelled.value) return;
      clearTimer();

      if (res.status === 3) {
        throw new Error(res.errorMessage || "处理失败");
      }

      result.value = res;
      processStatus.value = "success";
      uni.setStorageSync("prediction_result", JSON.stringify(res));
      uni.showToast({ title: "处理完成", icon: "success" });
      goStep(4);
    } catch (error) {
      if (cancelled.value) return;

      const errMsg = getErrorMessage(error, "处理失败");
      if (attemptNumber < MAX_RETRIES) {
        const delay = RETRY_DELAYS[attemptNumber] || 2000;
        errorMsg.value = `${errMsg}，${delay / 1000}秒后自动重试（${attemptNumber + 1}/${MAX_RETRIES}）`;
        await new Promise((r) => setTimeout(r, delay));
        if (!cancelled.value) {
          return attempt(attemptNumber + 1);
        }
      }

      clearTimer();
      processStatus.value = "error";
      errorMsg.value = errMsg;
    }
  };

  await attempt(0);
};

const handleGoCompare = () => {
  uni.navigateTo({ url: "/pages/side-by-side/index" });
};

const handleApplyPreset = (preset: PresetVO) => {
  try {
    const p = JSON.parse(preset.params);
    params.value = {
      strength: p.strength ?? DEFAULT_PARAMS.strength,
      saturation: p.saturation ?? DEFAULT_PARAMS.saturation,
      contrast: p.contrast ?? DEFAULT_PARAMS.contrast,
      sharpness: p.sharpen ?? DEFAULT_PARAMS.sharpness,
    };
  } catch {
    uni.showToast({ title: "预设参数解析失败", icon: "none" });
  }
};

const resetParams = () => {
  params.value = { ...DEFAULT_PARAMS };
};

const handleCancelProcess = () => {
  cancelled.value = true;
  clearTimer();
  processStatus.value = "idle";
  errorMsg.value = "";
  elapsedTime.value = 0;
  uni.showToast({ title: "已取消处理", icon: "none" });
};

const handleReset = () => {
  currentStep.value = 0;
  currentImage.value = null;
  selectedAlgorithm.value = null;
  params.value = { ...DEFAULT_PARAMS };
  processStatus.value = "idle";
  result.value = null;
  errorMsg.value = "";
  elapsedTime.value = 0;
  uploadedFileId.value = null;
  cancelled.value = false;
  clearTimer();
};

const formatSize = (bytes?: number): string => {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
};

const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}秒`;
  const minutes = Math.floor(seconds / 60);
  const remainSeconds = seconds % 60;
  return `${minutes}分${remainSeconds}秒`;
};
</script>

<style lang="scss" scoped>
.dehaze-page {
  padding: 24rpx;
  background: $color-bg-primary;
  min-height: 100vh;
}

/* 步骤指示器 */
.dehaze-steps {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: $color-white;
  border-radius: 20rpx;
  padding: 24rpx 16rpx;
  margin-bottom: 24rpx;
  box-shadow: $shadow-sm;

  .dehaze-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8rpx;
    flex: 1;
  }

  .dehaze-step-dot {
    width: 48rpx;
    height: 48rpx;
    border-radius: 50%;
    background: $color-bg-secondary;
    color: $color-text-placeholder;
    font-size: $font-xs;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;

    &.done {
      background: $gradient-primary;
      color: $color-white;
    }

    &.active {
      background: $color-primary;
      color: $color-white;
    }
  }

  .dehaze-step-label {
    font-size: 20rpx;
    color: $color-text-secondary;
    white-space: nowrap;

    &.active-label {
      color: $color-primary;
      font-weight: 600;
    }
  }
}

.dehaze-flow-body {
  background: $color-white;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: $shadow-sm;
}

.dehaze-step-content {
  min-height: 400rpx;
}

/* 上传区 */
.dehaze-upload-area {
  width: 100%;
  min-height: 360rpx;
  border: 2rpx dashed $color-border;
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;

  .dehaze-upload-preview {
    width: 100%;
    height: 360rpx;
  }

  .dehaze-upload-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12rpx;

    .dehaze-upload-text {
      font-size: $font-md;
      color: $color-text-secondary;
    }

    .dehaze-upload-hint {
      font-size: $font-xs;
      color: $color-text-placeholder;
    }
  }
}

.dehaze-image-info {
  display: flex;
  gap: 24rpx;
  margin-top: 16rpx;

  .dehaze-info-item {
    font-size: $font-xs;
    color: $color-text-secondary;
  }
}

/* 算法搜索 */
.dehaze-algo-search {
  display: flex;
  align-items: center;
  gap: 12rpx;
  height: 72rpx;
  padding: 0 20rpx;
  background: $color-bg-secondary;
  border-radius: 16rpx;
  margin-bottom: 16rpx;

  .dehaze-algo-search-input {
    flex: 1;
    height: 100%;
    font-size: $font-sm;
    background: transparent;
  }
}

.dehaze-loading {
  text-align: center;
  padding: 48rpx;
  color: $color-text-secondary;
}

.dehaze-algo-list {
  max-height: 500rpx;
  overflow-y: auto;
}

.dehaze-algo-card {
  padding: 20rpx;
  border-radius: 16rpx;
  background: $color-bg-secondary;
  margin-bottom: 12rpx;

  &.selected {
    background: $color-primary-bg;
    border: 2rpx solid $color-primary;
  }

  .dehaze-algo-name {
    font-size: $font-md;
    font-weight: 600;
    color: $color-text-primary;
  }

  .dehaze-algo-type {
    display: inline-block;
    padding: 2rpx 12rpx;
    background: $color-primary-bg;
    border-radius: 8rpx;
    font-size: $font-xs;
    color: $color-primary;
    margin-top: 6rpx;
  }

  .dehaze-algo-desc {
    display: block;
    font-size: $font-xs;
    color: $color-text-secondary;
    margin-top: 6rpx;
  }
}

/* 参数调节 */
.dehaze-params-section {
  .dehaze-preset-section {
    margin-bottom: 24rpx;

    .dehaze-preset-label {
      display: block;
      margin-bottom: 10rpx;
      font-size: 22rpx;
      color: $color-text-placeholder;
    }

    .dehaze-preset-scroll {
      white-space: nowrap;
    }

    .dehaze-preset-chip {
      display: inline-block;
      padding: 8rpx 20rpx;
      margin-right: 10rpx;
      font-size: 22rpx;
      color: $color-primary;
      background: $color-primary-bg;
      border-radius: 20rpx;

      &:active {
        opacity: 0.7;
      }
    }
  }

  .dehaze-param-item {
    margin-bottom: 24rpx;

    .dehaze-param-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 12rpx;

      .dehaze-param-label {
        font-size: $font-sm;
        color: $color-text-primary;
      }

      .dehaze-param-value {
        font-size: $font-sm;
        font-weight: 600;
        color: $color-primary;
      }
    }
  }

  .dehaze-param-reset {
    text-align: right;
    font-size: $font-xs;
    color: $color-primary;
    margin-top: -8rpx;
  }
}

.dehaze-algo-summary {
  padding: 16rpx 20rpx;
  background: $color-primary-bg;
  border-radius: 12rpx;
  margin-top: 16rpx;

  .dehaze-summary-label {
    font-size: $font-xs;
    color: $color-text-secondary;
  }

  .dehaze-summary-value {
    font-size: $font-xs;
    font-weight: 600;
    color: $color-primary;
  }
}

/* 处理状态 */
.dehaze-processing-status,
.dehaze-success-status,
.dehaze-error-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  padding: 48rpx 24rpx;
}

.dehaze-spinner {
  width: 64rpx;
  height: 64rpx;
  border: 4rpx solid $color-border;
  border-top-color: $color-primary;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.dehaze-success-icon {
  width: 80rpx;
  height: 80rpx;
  border-radius: 50%;
  background: $color-success;
  color: $color-white;
  font-size: 36rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dehaze-error-icon {
  width: 80rpx;
  height: 80rpx;
  border-radius: 50%;
  background: $color-danger;
  color: $color-white;
  font-size: 36rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dehaze-status-text {
  font-size: $font-lg;
  font-weight: 600;
  color: $color-text-primary;
}

.dehaze-status-hint {
  font-size: $font-sm;
  color: $color-text-secondary;
}

.dehaze-result-preview {
  width: 100%;
  margin-top: 16rpx;
  border-radius: 16rpx;
  overflow: hidden;

  .dehaze-result-img {
    width: 100%;
    height: 320rpx;
  }
}

/* 对比入口 */
.dehaze-compare-entry {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  padding: 48rpx 24rpx;

  .dehaze-compare-icon {
    width: 96rpx;
    height: 96rpx;
    border-radius: 50%;
    background: $color-primary-bg;
    color: $color-primary;
    font-size: 40rpx;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .dehaze-compare-title {
    font-size: $font-xl;
    font-weight: 700;
    color: $color-text-primary;
  }

  .dehaze-compare-desc {
    font-size: $font-sm;
    color: $color-text-secondary;
    text-align: center;
    line-height: 1.6;
  }
}

/* 操作按钮 */
.dehaze-step-action {
  display: flex;
  gap: 16rpx;
  margin-top: 32rpx;
}

.dehaze-btn {
  flex: 1;
  height: 88rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  border-radius: 16rpx;
  font-size: $font-md;
  font-weight: 600;

  &.dehaze-btn-primary {
    background: $gradient-primary;
    color: $color-white;

    &.disabled {
      opacity: 0.5;
      pointer-events: none;
    }
  }

  &.dehaze-btn-secondary {
    background: $color-bg-secondary;
    color: $color-text-primary;
  }
}
</style>
