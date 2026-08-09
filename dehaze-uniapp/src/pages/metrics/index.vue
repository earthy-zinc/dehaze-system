<template>
  <ImmersiveLayout title="指标评估">
    <scroll-view v-if="hasImages" class="main-content" scroll-y>
      <!-- 算法处理信息 -->
      <view class="info-card">
        <text class="card-title">处理信息</text>
        <view class="info-row">
          <text class="info-label">算法</text>
          <text class="info-value">{{ algorithmName }}</text>
        </view>
        <view v-if="result?.time !== undefined" class="info-row">
          <text class="info-label">耗时</text>
          <text class="info-value">{{ result.time }}s</text>
        </view>
        <view v-if="result?.fromCache" class="cache-tag">缓存命中</view>
      </view>

      <!-- 评估操作 -->
      <view class="eval-actions">
        <button
          class="eval-btn"
          :disabled="evaluating || !canEvaluate"
          @click="handleEvaluate"
        >
          <text v-if="evaluating">评估中...</text>
          <text v-else>开始评估</text>
        </button>
        <view v-if="hasImages && !gtUrl" class="gt-hint">
          <text class="gt-hint-text"
            >当前图片无 GT 参考，仅可计算无参考指标 (NIQE/BRISQUE)</text
          >
        </view>
      </view>

      <!-- 评估结果 -->
      <view v-if="evalResult" class="metrics-panel">
        <text class="section-title">评估结果</text>

        <!-- 有参考指标 -->
        <view v-if="hasReferenceMetrics" class="metric-group">
          <text class="group-title">图像质量指标 (有参考)</text>
          <view class="metrics-grid">
            <view
              v-for="m in referenceMetrics"
              :key="m.key"
              class="metric-card"
            >
              <text class="metric-key">{{ m.label }}</text>
              <text class="metric-value" :style="{ color: m.color }">{{
                m.displayValue
              }}</text>
              <text class="metric-desc">{{ m.desc }}</text>
            </view>
          </view>
        </view>

        <!-- 无参考指标 -->
        <view v-if="hasNoRefMetrics" class="metric-group">
          <text class="group-title">图像质量指标 (无参考)</text>
          <view class="metrics-grid">
            <view v-for="m in noRefMetrics" :key="m.key" class="metric-card">
              <text class="metric-key">{{ m.label }}</text>
              <text class="metric-value" :style="{ color: m.color }">{{
                m.displayValue
              }}</text>
              <text class="metric-desc">{{ m.desc }}</text>
            </view>
          </view>
        </view>

        <!-- 评估耗时 -->
        <view v-if="evalResult.time" class="eval-time">
          <text>评估耗时: {{ (evalResult.time / 1000).toFixed(2) }}s</text>
        </view>
      </view>

      <!-- 未评估提示 -->
      <view v-else class="eval-hint">
        <text class="hint-text">点击"开始评估"计算图像质量指标</text>
      </view>
    </scroll-view>

    <CompareEmptyState v-else text="暂无处理结果" btn-color="#ec4899" />

    <template #toolbar>
      <view class="toolbar-grid">
        <view
          v-for="m in modes"
          :key="m.key"
          class="toolbar-item"
          :class="{ active: m.key === 'metrics' }"
          @click="switchPage(m.path)"
        >
          <SvgIcon :name="m.icon" size="20" color="#ec4899" />
          <text>{{ m.label }}</text>
        </view>
      </view>
      <view class="toolbar-actions">
        <view class="action-item" @click="handleReprocess">
          <SvgIcon name="refresh" size="18" color="rgba(255,255,255,0.7)" />
          <text>重新处理</text>
        </view>
        <view class="action-item" @click="handleChangeAlgorithm">
          <SvgIcon name="swap" size="18" color="rgba(255,255,255,0.7)" />
          <text>换算法</text>
        </view>
      </view>
    </template>
  </ImmersiveLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import ImmersiveLayout from "@/layout/ImmersiveLayout.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvaluationResultVO } from "dehaze-sdk-js";

const store = useProcessingStore();
const evaluating = ref(false);
const evalResult = ref<EvaluationResultVO | null>(null);

const hasImages = computed(() => !!store.result?.resultUrl);
const gtUrl = computed(() => store.currentImage?.sampleInfo?.cleanUrl || "");
const canEvaluate = computed(() => hasImages.value);
const algorithmName = computed(() => store.selectedAlgorithm?.name || "-");
const result = computed(() => store.result);

interface MetricDisplay {
  key: string;
  label: string;
  value: number;
  unit: string;
  desc: string;
  displayValue: string;
  color: string;
}

const GOOD = "#34d399";
const FAIR = "#fbbf24";
const BAD = "#ef4444";
const NEUTRAL = "#9ca3af";

function getColor(key: string, value: number, higherBetter: boolean): string {
  const thresholds: Record<string, [number, number]> = {
    psnr: [30, 25],
    ssim: [0.9, 0.7],
    mse: [100, 500],
    niqe: [5, 8],
    brisque: [20, 40],
    entropy: [7.5, 6.5],
  };
  const range = thresholds[key];
  if (!range) return NEUTRAL;
  if (higherBetter) {
    return value >= range[0] ? GOOD : value >= range[1] ? FAIR : BAD;
  }
  return value <= range[0] ? GOOD : value <= range[1] ? FAIR : BAD;
}

const referenceMetrics = computed<MetricDisplay[]>(() => {
  const metrics = evalResult.value?.metrics || {};
  const defs = [
    { key: "psnr", label: "PSNR", unit: "dB", desc: "峰值信噪比，>30dB 优秀" },
    { key: "ssim", label: "SSIM", unit: "", desc: "结构相似度，>0.9 优秀" },
    { key: "mse", label: "MSE", unit: "", desc: "均方误差，越小越好" },
  ];
  return defs.map((d) => {
    const val = metrics[d.key];
    if (val === undefined)
      return { ...d, value: 0, displayValue: "-", color: NEUTRAL };
    return {
      ...d,
      value: val,
      displayValue: d.unit ? `${val.toFixed(2)} ${d.unit}` : val.toFixed(4),
      color: getColor(d.key, val, d.key !== "mse"),
    };
  });
});

const noRefMetrics = computed<MetricDisplay[]>(() => {
  const metrics = evalResult.value?.metrics || {};
  const defs = [
    { key: "niqe", label: "NIQE", unit: "", desc: "自然图像质量，<5 为好" },
    {
      key: "brisque",
      label: "BRISQUE",
      unit: "",
      desc: "无参考质量，越低越好",
    },
    { key: "entropy", label: "信息熵", unit: "", desc: "图像信息量，7-8 为佳" },
  ];
  return defs.map((d) => {
    const val = metrics[d.key];
    if (val === undefined)
      return { ...d, value: 0, displayValue: "-", color: NEUTRAL };
    return {
      ...d,
      value: val,
      displayValue: val.toFixed(2),
      color: getColor(d.key, val, d.key === "entropy"),
    };
  });
});

const hasReferenceMetrics = computed(() =>
  referenceMetrics.value.some((m) => m.value !== 0)
);
const hasNoRefMetrics = computed(() =>
  noRefMetrics.value.some((m) => m.value !== 0)
);

const modes = [
  {
    key: "side-by-side",
    label: "并排",
    path: "/pages/side-by-side/index",
    icon: "grid",
  },
  {
    key: "overlay",
    label: "重叠",
    path: "/pages/overlay/index",
    icon: "photo",
  },
  {
    key: "magnifier",
    label: "放大镜",
    path: "/pages/magnifier/index",
    icon: "search",
  },
  {
    key: "filter",
    label: "滤镜",
    path: "/pages/filter/index",
    icon: "setting",
  },
  {
    key: "metrics",
    label: "指标",
    path: "/pages/metrics/index",
    icon: "integral",
  },
];

async function handleEvaluate() {
  if (!store.selectedAlgorithm?.id) {
    uni.showToast({ title: "缺少算法信息", icon: "none" });
    return;
  }
  evaluating.value = true;
  try {
    const result = await ModelAPI.evaluateAndWait({
      algorithmId: store.selectedAlgorithm.id,
      predUrl: store.result?.resultUrl,
      gtUrl: gtUrl.value || undefined,
    });
    if (result.status === 3) {
      throw new Error(result.errorMessage || "评估失败");
    }
    evalResult.value = result;
    uni.showToast({ title: "评估完成", icon: "success" });
  } catch (e: any) {
    uni.showToast({ title: e.message || "评估失败", icon: "none" });
  } finally {
    evaluating.value = false;
  }
}

function switchPage(url: string) {
  uni.redirectTo({ url });
}

function handleReprocess() {
  uni.redirectTo({ url: "/pages/processing/index" });
}

function handleChangeAlgorithm() {
  uni.redirectTo({ url: "/pages/algorithm-select/index" });
}

onMounted(() => {
  if (!hasImages.value)
    uni.showToast({ title: "请先完成去雾处理", icon: "none" });
});
</script>

<style lang="scss" scoped>
.main-content {
  height: 100%;
  padding: 24rpx;
  overflow: hidden;
}

.info-card {
  padding: 28rpx 32rpx;
  margin-bottom: 24rpx;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 24rpx;

  .card-title {
    font-size: 28rpx;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    margin-bottom: 16rpx;
    display: block;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 12rpx 0;
    border-bottom: 1rpx solid rgba(255, 255, 255, 0.05);

    &:last-child {
      border-bottom: none;
    }

    .info-label {
      font-size: 26rpx;
      color: rgba(255, 255, 255, 0.5);
    }
    .info-value {
      font-size: 26rpx;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.8);
    }
  }

  .cache-tag {
    display: inline-block;
    margin-top: 12rpx;
    font-size: 22rpx;
    color: #fbbf24;
    background: rgba(251, 191, 36, 0.15);
    padding: 4rpx 12rpx;
    border-radius: 8rpx;
  }
}

.eval-actions {
  margin-bottom: 24rpx;
}

.eval-btn {
  width: 100%;
  padding: 24rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #ec4899, #db2777);
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 30rpx;
  font-weight: 600;

  &:disabled {
    opacity: 0.6;
  }
}

.gt-hint {
  padding: 20rpx 24rpx;
  margin-top: 16rpx;
  background: rgba(251, 191, 36, 0.12);
  border-radius: 16rpx;
}

.gt-hint-text {
  font-size: 24rpx;
  color: #fbbf24;
  line-height: 1.5;
}

.metrics-panel {
  .section-title {
    font-size: 28rpx;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    margin-bottom: 16rpx;
    display: block;
  }
}

.metric-group {
  margin-bottom: 24rpx;

  .group-title {
    display: block;
    font-size: 24rpx;
    color: rgba(255, 255, 255, 0.4);
    margin-bottom: 12rpx;
  }
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20rpx;
}

.metric-card {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 20rpx;
  padding: 28rpx;
  text-align: center;
}

.metric-key {
  display: block;
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.4);
  margin-bottom: 8rpx;
}

.metric-value {
  display: block;
  font-size: 36rpx;
  font-weight: 800;
  margin-bottom: 8rpx;
}

.metric-desc {
  font-size: 20rpx;
  color: rgba(255, 255, 255, 0.3);
}

.eval-time {
  text-align: center;
  padding: 16rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.3);
}

.eval-hint {
  display: flex;
  justify-content: center;
  padding: 48rpx;
}

.hint-text {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.3);
}

.toolbar-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4rpx;
  padding: 16rpx 16rpx 8rpx;
}

.toolbar-actions {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 4rpx;
  padding: 0 16rpx 16rpx;
}

.toolbar-item,
.action-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8rpx;
  padding: 20rpx 8rpx;
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.5);

  &:active {
    background: rgba(236, 72, 153, 0.15);
    border-radius: 12rpx;
  }

  &.active {
    color: #ec4899;
    background: rgba(236, 72, 153, 0.12);
    border-radius: 12rpx;
  }
}
</style>
