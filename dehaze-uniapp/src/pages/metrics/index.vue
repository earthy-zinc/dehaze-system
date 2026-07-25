<template>
  <PageLayout class="page">
    <view class="main-content">
      <PageHeaderCard
        icon="integral"
        icon-color="#ec4899"
        icon-bg="#fce7f3"
        title="指标评估"
        subtitle="图像质量评估指标"
        variant="dark"
      />

      <view v-if="hasImages" class="content-area">
        <!-- 算法处理信息 -->
        <view class="info-card">
          <text class="info-title">处理信息</text>
          <view class="info-row">
            <text class="info-label">算法</text>
            <text class="info-value">{{
              store.selectedAlgorithm?.name || "-"
            }}</text>
          </view>
          <view class="info-row">
            <text class="info-label">耗时</text>
            <text class="info-value">{{ store.result?.time || "-" }}s</text>
          </view>
          <view v-if="store.result?.fromCache" class="cache-tag"
            >⚡ 缓存命中</view
          >
        </view>

        <!-- 操作栏 -->
        <view class="eval-actions">
          <button
            class="eval-btn"
            :disabled="evaluating || !canEvaluate"
            @click="handleEvaluate"
          >
            <u-loading-icon
              v-if="evaluating"
              mode="circle"
              size="18"
              color="#fff"
            />
            <text>{{ evaluating ? "评估中..." : "开始评估" }}</text>
          </button>
          <!-- 无 GT 参考图时提示 -->
          <view v-if="hasImages && !gtUrl" class="gt-hint">
            <u-icon name="info-circle" size="16" color="#fbbf24" />
            <text class="gt-hint-text"
              >当前图片无 GT
              参考，无法评估。请使用数据集样例图片进行评估。</text
            >
          </view>
        </view>

        <!-- 评估结果 -->
        <view v-if="evalResult" class="metrics-panel">
          <text class="section-title">评估结果</text>
          <view class="metrics-grid">
            <view v-for="m in metricsList" :key="m.key" class="metric-card">
              <text class="metric-key">{{ m.label }}</text>
              <text class="metric-value" :style="{ color: m.color }">{{
                m.displayValue
              }}</text>
              <text class="metric-desc">{{ m.desc }}</text>
            </view>
          </view>
        </view>

        <!-- 未评估提示 -->
        <view v-else class="eval-hint">
          <u-icon name="info-circle" size="20" color="#9ca3af" />
          <text class="hint-text">点击"开始评估"计算图像质量指标</text>
        </view>

        <!-- 导航 -->
        <view class="nav-row">
          <view
            class="nav-item"
            @click="switchPage('/pages/side-by-side/index')"
          >
            <u-icon name="grid" size="20" color="#ec4899" /><text
              >并排对比</text
            >
          </view>
          <view class="nav-item" @click="switchPage('/pages/filter/index')">
            <u-icon name="setting" size="20" color="#ec4899" /><text
              >滤镜调节</text
            >
          </view>
        </view>
      </view>

      <view v-else class="empty-state">
        <up-empty mode="image" text="暂无处理结果" />
        <button class="back-btn" @click="handleBack">返回处理页</button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import { useProcessingStore } from "@/store/processing";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvaluationResultVO } from "dehaze-sdk-js";

const store = useProcessingStore();
const evaluating = ref(false);
const evalResult = ref<EvaluationResultVO | null>(null);

const hasImages = computed(() => !!store.result?.resultUrl);

/** GT 参考图 URL：数据集样例的无雾清晰图（cleanUrl） */
const gtUrl = computed(() => store.currentImage?.sampleInfo?.cleanUrl || "");

/** 是否可评估：需要处理结果与 GT 参考图同时存在 */
const canEvaluate = computed(() => hasImages.value && !!gtUrl.value);

interface MetricDisplay {
  key: string;
  label: string;
  value: number;
  unit: string;
  desc: string;
  better: "higher" | "lower";
  displayValue: string;
  color: string;
}

/** 指标阈值颜色配置：good/fair 为阈值，lowerIsBetter 表示数值越低越好 */
const COLOR_THRESHOLDS: Record<
  string,
  { good: number; fair: number; lowerIsBetter?: boolean }
> = {
  psnr: { good: 30, fair: 25 },
  ssim: { good: 0.9, fair: 0.7 },
  mse: { good: 100, fair: 500, lowerIsBetter: true },
};

const COLOR_GOOD = "#10b981";
const COLOR_FAIR = "#f59e0b";
const COLOR_BAD = "#ef4444";
const COLOR_NEUTRAL = "#3b82f6";

function getMetricColor(key: string, value: number): string {
  const cfg = COLOR_THRESHOLDS[key];
  if (!cfg) return COLOR_NEUTRAL;
  const isGood = cfg.lowerIsBetter ? value <= cfg.good : value >= cfg.good;
  const isFair = cfg.lowerIsBetter ? value <= cfg.fair : value >= cfg.fair;
  return isGood ? COLOR_GOOD : isFair ? COLOR_FAIR : COLOR_BAD;
}

const metricsList = computed<MetricDisplay[]>(() => {
  if (!evalResult.value) return getDefaultMetrics();

  const definitions: Omit<MetricDisplay, "value" | "displayValue" | "color">[] =
    [
      {
        key: "psnr",
        label: "PSNR",
        unit: "dB",
        desc: "峰值信噪比",
        better: "higher",
      },
      {
        key: "ssim",
        label: "SSIM",
        unit: "",
        desc: "结构相似度",
        better: "higher",
      },
      { key: "mse", label: "MSE", unit: "", desc: "均方误差", better: "lower" },
      {
        key: "fsim",
        label: "FSIM",
        unit: "",
        desc: "特征相似度",
        better: "higher",
      },
    ];

  const metrics = evalResult.value.metrics || {};
  return definitions.map((d) => {
    const value = metrics[d.key] ?? 0;
    const displayValue = d.unit
      ? `${value.toFixed(2)} ${d.unit}`
      : value.toFixed(4);
    return { ...d, value, displayValue, color: getMetricColor(d.key, value) };
  });
});

function getDefaultMetrics(): MetricDisplay[] {
  return [
    {
      key: "psnr",
      label: "PSNR",
      value: 0,
      unit: "dB",
      desc: "峰值信噪比",
      better: "higher",
      displayValue: "-",
      color: "#9ca3af",
    },
    {
      key: "ssim",
      label: "SSIM",
      value: 0,
      unit: "",
      desc: "结构相似度",
      better: "higher",
      displayValue: "-",
      color: "#9ca3af",
    },
    {
      key: "mse",
      label: "MSE",
      value: 0,
      unit: "",
      desc: "均方误差",
      better: "lower",
      displayValue: "-",
      color: "#9ca3af",
    },
    {
      key: "fsim",
      label: "FSIM",
      value: 0,
      unit: "",
      desc: "特征相似度",
      better: "higher",
      displayValue: "-",
      color: "#9ca3af",
    },
  ];
}

async function handleEvaluate() {
  if (!store.selectedAlgorithm?.id) {
    uni.showToast({ title: "缺少算法信息", icon: "none" });
    return;
  }
  if (!gtUrl.value) {
    uni.showToast({ title: "当前图片无 GT 参考，无法评估", icon: "none" });
    return;
  }
  evaluating.value = true;
  try {
    const result = await ModelAPI.evaluateAndWait({
      algorithmId: store.selectedAlgorithm.id,
      predUrl: store.result?.resultUrl,
      gtUrl: gtUrl.value,
    });
    if (result.status === "failed") {
      throw new Error(result.errorMessage || "评估失败");
    }
    evalResult.value = result;
    uni.showToast({ title: "评估完成", icon: "success" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "评估失败，请检查后端服务";
    uni.showToast({ title: msg, icon: "none" });
  } finally {
    evaluating.value = false;
  }
}

function switchPage(url: string) {
  uni.redirectTo({ url });
}
function handleBack() {
  uni.navigateBack();
}

onMounted(() => {
  if (!hasImages.value)
    uni.showToast({ title: "请先完成去雾处理", icon: "none" });
});
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: #0f172a;
}
.main-content {
  padding: 24rpx;
}

.info-card {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 20rpx;
  padding: 28rpx;
  margin-bottom: 24rpx;
}
.info-title {
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
  & + & {
    border-top: 1rpx solid rgba(255, 255, 255, 0.05);
  }
}
.info-label {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
}
.info-value {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.8);
  font-weight: 500;
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

.eval-actions {
  margin-bottom: 24rpx;
}
.eval-btn {
  width: 100%;
  padding: 24rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12rpx;
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
  display: flex;
  align-items: center;
  gap: 12rpx;
  margin-top: 16rpx;
  padding: 20rpx 24rpx;
  background: rgba(251, 191, 36, 0.12);
  border-radius: 16rpx;
}
.gt-hint-text {
  font-size: 24rpx;
  color: #fbbf24;
  flex: 1;
  line-height: 1.5;
}

.section-title {
  font-size: 28rpx;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 16rpx;
  display: block;
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

.eval-hint {
  display: flex;
  align-items: center;
  gap: 12rpx;
  justify-content: center;
  padding: 48rpx;
}
.hint-text {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.3);
}

.nav-row {
  display: flex;
  gap: 20rpx;
  margin-top: 32rpx;
}
.nav-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12rpx;
  padding: 28rpx;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
  &:active {
    background: rgba(236, 72, 153, 0.15);
  }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.back-btn {
  margin-top: 32rpx;
  padding: 16rpx 48rpx;
  background: #ec4899;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
