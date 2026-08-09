<template>
  <PageLayout level="L2" title="指标管理" class="page">
    <view class="main-content">
      <!-- 页面标题 -->
      <PageHeaderCard
        icon="bar-chart"
        icon-color="#8b5cf6"
        icon-bg="linear-gradient(135deg, #ede9fe, #ddd6fe)"
        title="指标管理"
        subtitle="查看评估记录与指标对比"
      />

      <!-- 工具栏 -->
      <view class="toolbar">
        <view v-if="!compareMode" class="toolbar-row">
          <text class="toolbar-hint">选择记录进行指标对比（最多3条）</text>
          <button
            v-if="selectedIds.size >= 2"
            class="toolbar-btn"
            @click="startCompare"
          >
            对比 ({{ selectedIds.size }})
          </button>
        </view>
        <view v-else class="toolbar-row">
          <text class="toolbar-hint"
            >对比 {{ compareRecords.length }} 条记录</text
          >
          <button class="toolbar-btn outline" @click="exitCompare">
            退出对比
          </button>
        </view>
      </view>

      <!-- 对比表格 -->
      <view
        v-if="compareMode && compareRecords.length > 0"
        class="compare-section"
      >
        <scroll-view scroll-x class="compare-table-wrapper">
          <view class="compare-table">
            <view class="table-row header">
              <view class="table-cell metric-label-cell">
                <text class="cell-text">指标</text>
              </view>
              <view
                v-for="r in compareRecords"
                :key="r.id"
                class="table-cell algo-cell"
              >
                <text class="cell-text">{{
                  r.algorithmName || `算法${r.algorithmId}`
                }}</text>
              </view>
            </view>
            <view v-for="key in metricKeys" :key="key" class="table-row">
              <view class="table-cell metric-label-cell">
                <text class="cell-text">{{ METRIC_LABELS[key] || key }}</text>
              </view>
              <view
                v-for="r in compareRecords"
                :key="r.id"
                class="table-cell value-cell"
              >
                <text class="cell-text value">
                  {{
                    r.metrics?.[key] != null
                      ? formatValue(key, r.metrics[key])
                      : "-"
                  }}
                </text>
              </view>
            </view>
          </view>
        </scroll-view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-state">
        <view class="loading-spinner" />
        <text class="loading-text">加载评估记录...</text>
      </view>

      <!-- 错误状态 -->
      <view v-else-if="error" class="error-state">
        <text class="error-text">{{ error }}</text>
        <button class="retry-btn" @click="fetchRecords">重新加载</button>
      </view>

      <!-- 空状态 -->
      <view v-else-if="records.length === 0" class="empty-state">
        <view class="empty-tip">暂无评估记录</view>
        <text class="empty-hint">完成去雾处理后可在对比页生成评估指标</text>
      </view>

      <!-- 记录列表 -->
      <view v-else class="record-list">
        <view
          v-for="record in records"
          :key="record.id"
          class="record-card"
          :class="{ selected: selectedIds.has(record.id) }"
          @click="!compareMode && toggleSelect(record.id)"
        >
          <view class="card-header">
            <text class="card-algo">
              {{ record.algorithmName || `算法${record.algorithmId}` }}
            </text>
            <view class="card-meta">
              <view
                class="status-badge"
                :class="'status-' + (record.status ?? 1)"
              >
                {{ statusLabel(record.status) }}
              </view>
              <view
                v-if="!compareMode && selectedIds.has(record.id)"
                class="select-mark"
              >
                ✓
              </view>
            </view>
          </view>

          <view
            v-if="record.metrics && Object.keys(record.metrics).length > 0"
            class="card-metrics"
          >
            <view
              v-for="(value, key) in record.metrics"
              :key="key"
              class="mini-metric"
            >
              <text class="mini-label">{{ METRIC_LABELS[key] || key }}</text>
              <text class="mini-value">{{ formatValue(key, value) }}</text>
            </view>
          </view>

          <view class="card-footer">
            <text v-if="record.time != null" class="card-time">
              耗时 {{ (record.time / 1000).toFixed(1) }}s
            </text>
            <text v-if="record.createTime" class="card-date">{{
              record.createTime
            }}</text>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import { ModelAPI } from "dehaze-sdk-js";
import type { EvalMetricsVO } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

// ==================== 常量 ====================

const METRIC_LABELS: Record<string, string> = {
  psnr: "PSNR",
  ssim: "SSIM",
  lpips: "LPIPS",
  niqe: "NIQE",
  entropy: "信息熵",
  mse: "MSE",
};

// ==================== 状态 ====================

const records = ref<EvalMetricsVO[]>([]);
const loading = ref(true);
const error = ref("");
const selectedIds = ref<Set<number>>(new Set());
const compareMode = ref(false);

// ==================== 计算属性 ====================

const compareRecords = computed(() => {
  if (!compareMode.value) return [];
  return records.value.filter((r) => selectedIds.value.has(r.id));
});

const metricKeys = computed(() => {
  const keys = new Set<string>();
  compareRecords.value.forEach((r) => {
    if (r.metrics) Object.keys(r.metrics).forEach((k) => keys.add(k));
  });
  return Array.from(keys);
});

// ==================== 方法 ====================

function statusLabel(status?: number): string {
  if (status === 2) return "已完成";
  if (status === 3) return "失败";
  if (status === 1) return "处理中";
  return "未知";
}

function formatValue(key: string, value: number): string {
  if (key === "psnr") return value.toFixed(2) + " dB";
  if (key === "ssim" || key === "lpips") return value.toFixed(4);
  return value.toFixed(2);
}

async function fetchRecords() {
  loading.value = true;
  error.value = "";
  try {
    const res = await ModelAPI.getEvalMetrics({ pageNum: 1, pageSize: 50 });
    records.value = (res.list || []) as EvalMetricsVO[];
  } catch (err: unknown) {
    error.value = getErrorMessage(err, "加载评估记录失败");
  } finally {
    loading.value = false;
  }
}

function toggleSelect(id: number) {
  const next = new Set(selectedIds.value);
  if (next.has(id)) {
    next.delete(id);
  } else {
    if (next.size >= 3) {
      uni.showToast({ title: "最多选择3条记录对比", icon: "none" });
      return;
    }
    next.add(id);
  }
  selectedIds.value = next;
}

function startCompare() {
  if (selectedIds.value.size < 2) {
    uni.showToast({ title: "请至少选择2条记录", icon: "none" });
    return;
  }
  compareMode.value = true;
}

function exitCompare() {
  compareMode.value = false;
  selectedIds.value = new Set();
}

onMounted(() => {
  fetchRecords();
});
</script>

<style lang="scss" scoped>
@import "@/styles/mixins.scss";

.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: $spacing-md;
  @include safe-area-bottom(120rpx);
}

/* 工具栏 */
.toolbar {
  background: $color-white;
  border-radius: $radius-lg;
  padding: $spacing-md;
  margin-bottom: $spacing-md;
  box-shadow: $shadow-sm;
}

.toolbar-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: $spacing-sm;
}

.toolbar-hint {
  font-size: $font-sm;
  color: $color-text-secondary;
  flex: 1;
}

.toolbar-btn {
  padding: 12rpx 32rpx;
  background: $color-accent;
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-sm;
  font-weight: 600;
  white-space: nowrap;

  &.outline {
    background: transparent;
    color: $color-accent;
    border: 2rpx solid $color-accent;
  }

  &:active {
    opacity: 0.85;
  }
}

/* 对比表格 */
.compare-section {
  margin-bottom: $spacing-md;
  background: $color-white;
  border-radius: $radius-xl;
  overflow: hidden;
  box-shadow: $shadow-md;
}

.compare-table-wrapper {
  overflow-x: auto;
}

.compare-table {
  min-width: 600rpx;
}

.table-row {
  display: flex;
  border-bottom: 1rpx solid $color-border-light;

  &.header {
    background: #ede9fe;
    border-bottom: 2rpx solid #c4b5fd;
  }
}

.table-cell {
  padding: 20rpx 24rpx;
  min-width: 160rpx;
  flex: 1;
}

.metric-label-cell {
  min-width: 140rpx;
  flex-shrink: 0;
  background: rgba(0, 0, 0, 0.01);
}

.algo-cell {
  min-width: 200rpx;
}

.cell-text {
  font-size: $font-sm;
  color: $color-text-primary;
  font-weight: 500;

  &.value {
    font-weight: 600;
    color: $color-accent;
  }
}

.header .cell-text {
  font-weight: 700;
  color: #5b21b6;
}

/* 加载/错误/空 */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}

.loading-spinner {
  border-top-color: $color-accent;
}

.loading-text {
  margin-top: $spacing-md;
  font-size: $font-md;
  color: $color-text-placeholder;
}

.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.error-text {
  font-size: $font-md;
  color: $color-danger;
  margin-bottom: $spacing-md;
}

.retry-btn {
  padding: $spacing-sm 48rpx;
  background: $color-accent;
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-md;
}

.empty-state {
  padding: 80rpx 0;
}

.empty-tip {
  font-size: $font-md;
}

.empty-hint {
  font-size: $font-sm;
  color: $color-text-placeholder;
  display: block;
  text-align: center;
  margin-top: $spacing-sm;
}

/* 记录列表 */
.record-list {
  display: flex;
  flex-direction: column;
  gap: $spacing-sm;
}

.record-card {
  background: $color-white;
  border-radius: $radius-lg;
  padding: 24rpx;
  box-shadow: $shadow-sm;
  border: 2rpx solid transparent;

  &.selected {
    border-color: $color-accent;
    background: #faf5ff;
  }

  &:active {
    transform: scale(0.98);
  }
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16rpx;
}

.card-algo {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
}

.card-meta {
  display: flex;
  align-items: center;
  gap: 12rpx;
}

.status-badge {
  font-size: $font-xs;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;

  &.status-2 {
    color: $color-success;
    background: $color-success-bg;
  }
  &.status-3 {
    color: $color-danger;
    background: $color-danger-bg;
  }
  &.status-1 {
    color: $color-warning;
    background: $color-warning-bg;
  }
}

.select-mark {
  width: 40rpx;
  height: 40rpx;
  background: $color-accent;
  color: $color-white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22rpx;
  font-weight: 700;
}

.card-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
  margin-bottom: 12rpx;
}

.mini-metric {
  display: flex;
  align-items: center;
  gap: 6rpx;
  background: $color-bg-secondary;
  padding: 6rpx 16rpx;
  border-radius: $radius-sm;
}

.mini-label {
  font-size: $font-xs;
  color: $color-text-secondary;
}

.mini-value {
  font-size: $font-xs;
  color: $color-accent;
  font-weight: 600;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.card-date {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
</style>
