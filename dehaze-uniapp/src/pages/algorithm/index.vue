<template>
  <PageLayout level="L2" title="算法库" class="page">
    <view class="main-content">
      <!-- 页面标题 -->
      <PageHeaderCard
        icon="server-fill"
        icon-color="#6366f1"
        icon-bg="linear-gradient(135deg, #e0e7ff, #c7d2fe)"
        title="算法库"
        subtitle="浏览算法、查看详情、智能推荐"
      />

      <!-- 搜索框 -->
      <view class="search-bar">
        <SvgIcon name="search" size="18" color="#9ca3af" />
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
          <SvgIcon name="close-circle-fill" size="16" color="#9ca3af" />
        </view>
      </view>

      <!-- 状态筛选 -->
      <view class="filter-bar">
        <view
          class="filter-item"
          :class="{ active: statusFilter === '' }"
          @click="statusFilter = ''"
        >
          全部
        </view>
        <view
          class="filter-item"
          :class="{ active: statusFilter === 4 }"
          @click="statusFilter = 4"
        >
          已发布
        </view>
        <view v-if="recommendLoading" class="filter-item">
          <text class="filter-loading-text">推荐加载中...</text>
        </view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-state">
        <view class="loading-spinner" />
        <text class="loading-text">加载算法列表...</text>
      </view>

      <!-- 错误状态 -->
      <view v-else-if="error" class="error-state">
        <text class="error-text">{{ error }}</text>
        <button class="retry-btn" @click="fetchAlgorithms">重新加载</button>
      </view>

      <!-- 算法列表 -->
      <view v-else class="algorithm-section">
        <text class="section-label">
          可用算法 ({{ filteredList.length
          }}{{ searchKeyword ? "/" + algorithmList.length : "" }})
        </text>

        <view v-if="filteredList.length === 0" class="empty-state">
          <view class="empty-tip">暂无算法数据</view>
        </view>

        <view v-else class="algorithm-list">
          <view
            v-for="algorithm in filteredList"
            :key="algorithm.id"
            class="algorithm-card"
            @click="handleDetail(algorithm)"
          >
            <view class="card-header">
              <view class="card-name-row">
                <text class="card-name">{{ algorithm.name }}</text>
                <text class="card-type">{{ algorithm.type || "未知" }}</text>
              </view>
              <view class="card-tags">
                <view
                  v-if="recommendedIds.has(algorithm.id)"
                  class="tag-recommend"
                >
                  推荐
                </view>
                <view
                  class="tag-status"
                  :class="'status-' + (algorithm.status ?? 0)"
                >
                  {{ statusLabel(algorithm.status) }}
                </view>
              </view>
            </view>

            <text v-if="algorithm.description" class="card-desc">
              {{ algorithm.description }}
            </text>

            <view class="card-meta">
              <text v-if="algorithm.version" class="meta-item">
                v{{ algorithm.version }}
              </text>
              <text v-if="algorithm.flops" class="meta-item">
                {{ algorithm.flops }}
              </text>
              <text v-if="algorithm.size" class="meta-item">
                {{ algorithm.size }}
              </text>
            </view>

            <view class="card-actions" @click.stop>
              <button class="use-btn" @click="handleUseAlgorithm(algorithm)">
                使用该算法
              </button>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 算法详情弹层 -->
    <Popup :show="detailVisible" mode="bottom" round @close="handleCloseDetail">
      <view class="detail-popup">
        <view class="detail-header">
          <text class="detail-title">{{ detailAlgo?.name || "算法详情" }}</text>
          <view class="detail-close" @click="handleCloseDetail">
            <SvgIcon name="close" size="20" color="#6b7280" />
          </view>
        </view>

        <scroll-view v-if="detailAlgo" class="detail-scroll" scroll-y>
          <!-- 基本信息 -->
          <view class="detail-section">
            <text class="detail-section-title">基本信息</text>
            <view class="detail-info-card">
              <view class="detail-row">
                <text class="detail-label">名称</text>
                <text class="detail-value">{{ detailAlgo.name }}</text>
              </view>
              <view class="detail-row">
                <text class="detail-label">类型</text>
                <text class="detail-value detail-type-tag">
                  {{ detailAlgo.type || "未知" }}
                </text>
              </view>
              <view class="detail-row">
                <text class="detail-label">版本</text>
                <text class="detail-value">
                  v{{ detailAlgo.version || "-" }}
                </text>
              </view>
              <view class="detail-row">
                <text class="detail-label">状态</text>
                <text class="detail-value">
                  {{ statusLabel(detailAlgo.status) }}
                </text>
              </view>
            </view>
          </view>

          <!-- 描述 -->
          <view v-if="detailAlgo.description" class="detail-section">
            <text class="detail-section-title">算法描述</text>
            <view class="detail-desc-card">
              <text class="detail-desc-text">{{ detailAlgo.description }}</text>
            </view>
          </view>

          <!-- 技术指标 -->
          <view class="detail-section">
            <text class="detail-section-title">技术指标</text>
            <view class="detail-specs-grid">
              <view class="detail-spec-card">
                <text class="detail-spec-label">计算量</text>
                <text class="detail-spec-value">
                  {{ detailAlgo.flops || "-" }}
                </text>
              </view>
              <view class="detail-spec-card">
                <text class="detail-spec-label">模型大小</text>
                <text class="detail-spec-value">
                  {{ detailAlgo.size || "-" }}
                </text>
              </view>
              <view class="detail-spec-card">
                <text class="detail-spec-label">路径</text>
                <text class="detail-spec-value detail-spec-small">
                  {{ detailAlgo.path || "-" }}
                </text>
              </view>
              <view class="detail-spec-card">
                <text class="detail-spec-label">创建时间</text>
                <text class="detail-spec-value detail-spec-small">
                  {{ formatRelativeTime(detailAlgo.createTime || "") }}
                </text>
              </view>
            </view>
          </view>

          <!-- 参数说明 -->
          <view v-if="detailAlgo.params" class="detail-section">
            <text class="detail-section-title">参数说明</text>
            <view class="detail-params-card">
              <text class="detail-params-text">{{ detailAlgo.params }}</text>
            </view>
          </view>
        </scroll-view>

        <view class="detail-bottom-bar">
          <button
            class="detail-use-btn"
            @click="detailAlgo && handleUseAlgorithm(detailAlgo)"
          >
            使用该算法
          </button>
        </view>
      </view>
    </Popup>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import Popup from "@/components/common/Popup.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { AlgorithmAPI, RecommendationAPI } from "dehaze-sdk-js";
import type { Algorithm } from "dehaze-sdk-js";
import { formatRelativeTime } from "@/utils/format";
import { getErrorMessage } from "@/utils/error";

// ==================== 状态 ====================

const algorithmList = ref<Algorithm[]>([]);
const loading = ref(false);
const error = ref("");
const searchKeyword = ref("");
const statusFilter = ref<number | "">(4);

// 智能推荐
const recommendLoading = ref(false);
const recommendedIds = ref<Set<number>>(new Set());

// 详情弹层
const detailAlgo = ref<Algorithm | null>(null);
const detailVisible = ref(false);

// ==================== 计算属性 ====================

/** 过滤算法列表：仅叶子节点，按搜索关键词和状态筛选 */
const filteredList = computed<Algorithm[]>(() => {
  let list = algorithmList.value;

  if (statusFilter.value !== "") {
    list = list.filter((a) => a.status === statusFilter.value);
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

const STATUS_MAP: Record<number, string> = {
  0: "待审核",
  1: "已通过",
  2: "已驳回",
  3: "已下架",
  4: "已发布",
  5: "已归档",
};

function statusLabel(status?: number): string {
  if (status == null) return "未知";
  return STATUS_MAP[status] || `状态${status}`;
}

/** 加载算法列表 */
async function fetchAlgorithms() {
  if (loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    const data = await AlgorithmAPI.getList();
    algorithmList.value = data || [];
  } catch (e) {
    error.value = getErrorMessage(e, "加载失败，请重试");
  } finally {
    loading.value = false;
  }
}

/** 加载智能推荐（无图像上下文时获取默认推荐） */
async function fetchRecommendations() {
  recommendLoading.value = true;
  try {
    const recs = await RecommendationAPI.getAlgorithmRecommendations({});
    const ids = new Set<number>();
    (recs || []).forEach((r) => ids.add(r.algorithmId));
    recommendedIds.value = ids;
  } catch {
    // 推荐失败不影响主列表
  } finally {
    recommendLoading.value = false;
  }
}

/** 查看算法详情 */
async function handleDetail(algo: Algorithm) {
  detailAlgo.value = algo;
  detailVisible.value = true;
  try {
    const detail = await AlgorithmAPI.getAlgorithmInfoById(algo.id);
    detailAlgo.value = detail;
  } catch {
    // 使用列表数据
  }
}

/** 关闭详情弹层 */
function handleCloseDetail() {
  detailVisible.value = false;
  detailAlgo.value = null;
}

/** 使用该算法：带入去雾流程 */
function handleUseAlgorithm(algo: Algorithm) {
  // 存储选中算法到本地
  uni.setStorageSync("selected_algorithm", JSON.stringify(algo));
  // 跳转到 algorithm-select 页面
  uni.navigateTo({
    url: `/pages/algorithm-select/index?algorithmId=${algo.id}`,
    fail: () => {
      uni.showToast({ title: "页面跳转失败", icon: "none" });
    },
  });
}

// ==================== 生命周期 ====================

onMounted(() => {
  fetchAlgorithms();
  fetchRecommendations();
});
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: $spacing-md;
  padding-bottom: calc(120rpx + $safe-area-bottom-env);
}

/* 搜索框 */
.search-bar {
  display: flex;
  align-items: center;
  gap: $spacing-sm;
  background: $color-white;
  border-radius: $radius-lg;
  padding: 20rpx 24rpx;
  margin-bottom: $spacing-md;
  box-shadow: $shadow-sm;
}

.search-input {
  flex: 1;
  font-size: $font-md;
  color: $color-text-primary;
}

.search-placeholder {
  color: $color-text-placeholder;
  font-size: $font-md;
}

.search-clear {
  padding: $spacing-sm 24rpx;
  min-width: 88rpx;
  min-height: 88rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 筛选 */
.filter-bar {
  display: flex;
  gap: $spacing-sm;
  margin-bottom: $spacing-md;
}

.filter-item {
  padding: 14rpx 28rpx;
  background: $color-white;
  border-radius: 32rpx;
  font-size: $font-sm;
  color: $color-text-secondary;
  box-shadow: $shadow-sm;

  &.active {
    background: #e0e7ff;
    color: $color-secondary;
    font-weight: 600;
  }

  &:active {
    opacity: 0.85;
  }
}

.filter-loading-text {
  color: $color-text-placeholder;
}

/* 加载/错误/空 */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
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
  background: $color-secondary;
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-md;
}

.empty-state {
  padding: 80rpx 0;
}

/* 算法列表 */
.algorithm-section {
  margin-bottom: $spacing-md;
}

.section-label {
  font-size: $font-md;
  font-weight: 600;
  color: #374151;
  margin-bottom: $spacing-sm;
  display: block;
}

.algorithm-list {
  display: flex;
  flex-direction: column;
  gap: 20rpx;
}

.algorithm-card {
  background: $color-white;
  border-radius: 20rpx;
  padding: 28rpx;
  box-shadow: $shadow-md;
  border: 2rpx solid transparent;

  &:active {
    transform: scale(0.98);
  }
}

.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 12rpx;
}

.card-name-row {
  display: flex;
  align-items: center;
  gap: 12rpx;
  flex: 1;
  min-width: 0;
}

.card-name {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
}

.card-type {
  font-size: $font-xs;
  color: $color-secondary;
  background: #e0e7ff;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  flex-shrink: 0;
}

.card-tags {
  display: flex;
  align-items: center;
  gap: 8rpx;
  flex-shrink: 0;
}

.tag-recommend {
  font-size: $font-xs;
  color: #d97706;
  background: #fef3c7;
  padding: 2rpx 10rpx;
  border-radius: $radius-sm;
  font-weight: 600;
}

.tag-status {
  font-size: $font-xs;
  padding: 2rpx 10rpx;
  border-radius: $radius-sm;

  &.status-4 {
    color: $color-success;
    background: $color-success-bg;
  }

  &.status-0 {
    color: $color-warning;
    background: $color-warning-bg;
  }

  &.status-3 {
    color: $color-info;
    background: $color-bg-secondary;
  }
}

.card-desc {
  display: block;
  font-size: $font-sm;
  color: $color-text-secondary;
  line-height: 1.5;
  margin-bottom: 12rpx;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  display: flex;
  gap: $spacing-sm;
  margin-bottom: $spacing-sm;
}

.meta-item {
  font-size: $font-xs;
  color: $color-text-placeholder;
  background: $color-bg-secondary;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
}

.card-actions {
  display: flex;
  justify-content: flex-end;
}

.use-btn {
  padding: 14rpx 32rpx;
  background: $gradient-primary;
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-sm;
  font-weight: 600;

  &:active {
    opacity: 0.85;
  }
}

/* ==================== 详情弹层 ==================== */

.detail-popup {
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  border-radius: $radius-xl $radius-xl 0 0;
  overflow: hidden;
}

.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: $spacing-lg;
  border-bottom: 1rpx solid $color-border-light;
  flex-shrink: 0;
}

.detail-title {
  font-size: $font-xl;
  font-weight: 700;
  color: $color-text-primary;
}

.detail-close {
  width: 56rpx;
  height: 56rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: $color-bg-secondary;
}

.detail-scroll {
  flex: 1;
  overflow-y: auto;
  padding: $spacing-md;
}

.detail-section {
  margin-bottom: $spacing-md;
}

.detail-section-title {
  font-size: $font-md;
  font-weight: 600;
  color: #374151;
  margin-bottom: 12rpx;
  display: block;
}

.detail-info-card,
.detail-desc-card,
.detail-params-card {
  background: $color-bg-secondary;
  border-radius: $radius-lg;
  padding: 28rpx;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14rpx 0;

  & + & {
    border-top: 1rpx solid $color-border;
  }
}

.detail-label {
  font-size: $font-sm;
  color: $color-text-secondary;
}

.detail-value {
  font-size: $font-sm;
  font-weight: 500;
  color: $color-text-primary;
}

.detail-type-tag {
  color: $color-secondary;
  background: #e0e7ff;
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}

.detail-desc-text {
  font-size: $font-sm;
  color: #4b5563;
  line-height: 1.6;
}

.detail-params-text {
  font-size: 24rpx;
  color: #4b5563;
  line-height: 1.6;
  font-family: monospace;
  white-space: pre-wrap;
}

.detail-specs-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: $spacing-sm;
}

.detail-spec-card {
  background: $color-bg-secondary;
  border-radius: $radius-lg;
  padding: 24rpx;
  text-align: center;
}

.detail-spec-label {
  display: block;
  font-size: $font-xs;
  color: $color-text-placeholder;
  margin-bottom: $spacing-xs;
}

.detail-spec-value {
  display: block;
  font-size: $font-md;
  font-weight: 700;
  color: $color-text-primary;
}

.detail-spec-small {
  font-size: $font-xs;
  font-weight: 500;
}

.detail-bottom-bar {
  padding: $spacing-md $spacing-lg;
  padding-bottom: calc($spacing-md + $safe-area-bottom-env);
  border-top: 1rpx solid $color-border-light;
  flex-shrink: 0;
}

.detail-use-btn {
  width: 100%;
  padding: 24rpx;
  background: $gradient-primary;
  color: $color-white;
  border: none;
  border-radius: $radius-lg;
  font-size: $font-lg;
  font-weight: 700;

  &:active {
    opacity: 0.85;
  }
}
</style>
