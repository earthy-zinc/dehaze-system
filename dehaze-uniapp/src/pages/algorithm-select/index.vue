<template>
  <PageLayout level="L2" title="选择算法" class="algorithm-select-page">
    <view class="main-content">
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

      <!-- 智能推荐 -->
      <view
        v-if="
          processingStore.hasImage &&
          (recommendLoading || recommendList.length > 0 || imageAnalysis)
        "
        class="recommend-section"
      >
        <text class="section-label">智能推荐</text>
        <view v-if="imageAnalysis" class="analysis-tags">
          <text class="analysis-tag">雾霾: {{ imageAnalysis.hazeLevel }}</text>
          <text class="analysis-tag">场景: {{ imageAnalysis.sceneType }}</text>
        </view>
        <view v-if="recommendLoading" class="recommend-loading">
          <view class="loading-spinner" style="border-top-color: #8b5cf6" />
          <text class="recommend-loading-text">分析中...</text>
        </view>
        <view v-else class="recommend-list">
          <view
            v-for="(item, index) in recommendList"
            :key="item.algorithmId"
            class="recommend-card"
            @click="handleSelectRecommend(item)"
          >
            <view class="recommend-rank">#{{ index + 1 }}</view>
            <view class="recommend-info">
              <view class="recommend-header">
                <text class="recommend-name">{{ item.algorithmName }}</text>
                <text class="recommend-score"
                  >{{ Math.round(item.matchScore) }}%</text
                >
              </view>
              <text v-if="item.reason" class="recommend-reason">{{
                item.reason
              }}</text>
              <view class="match-score-bar">
                <view
                  class="match-score-fill"
                  :style="{ width: Math.round(item.matchScore) + '%' }"
                />
              </view>
            </view>
            <view
              class="fav-btn"
              @click.stop="handleToggleFavorite(item.algorithmId)"
            >
              <SvgIcon
                :name="favoriteIds.has(item.algorithmId) ? 'star-fill' : 'star'"
                size="18"
                :color="
                  favoriteIds.has(item.algorithmId) ? '#f59e0b' : '#9ca3af'
                "
              />
            </view>
          </view>
        </view>
      </view>

      <!-- 搜索栏 -->
      <view class="search-bar">
        <SvgIcon name="search" size="18" color="#9ca3af" />
        <input
          v-model="searchKeyword"
          class="search-input"
          type="text"
          placeholder="搜索算法名称、类型或描述"
          placeholder-class="search-placeholder"
          @focus="showHistory = true"
          @blur="onSearchBlur"
          @confirm="handleSearchSubmit"
        />
        <view v-if="searchKeyword" class="search-clear" @click="clearSearch">
          <SvgIcon name="close-circle-fill" size="16" color="#9ca3af" />
        </view>
      </view>

      <!-- 搜索历史 -->
      <view
        v-if="showHistory && !searchKeyword && searchHistory.length > 0"
        class="search-history-panel"
      >
        <view class="history-header">
          <text class="history-title">搜索历史</text>
          <view class="history-clear" @click="clearHistory">
            <text>清空</text>
          </view>
        </view>
        <view class="history-tags">
          <view
            v-for="kw in searchHistory"
            :key="kw"
            class="history-tag"
            @click="useHistory(kw)"
          >
            <text>{{ kw }}</text>
          </view>
        </view>
      </view>

      <!-- 对比栏 -->
      <view v-if="compareList.length > 0" class="compare-bar">
        <view class="compare-bar-tags">
          <view v-for="c in compareList" :key="c.id" class="compare-bar-tag">
            <text>{{ c.name }}</text>
            <view class="compare-bar-remove" @click="toggleCompare(c)">
              <SvgIcon name="close-circle-fill" size="12" color="#9ca3af" />
            </view>
          </view>
        </view>
        <view class="compare-bar-btn" @click="handleCompare">
          <text>对比 ({{ compareList.length }}/{{ COMPARE_MAX }})</text>
        </view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
        <text class="loading-text">加载算法列表...</text>
      </view>

      <!-- 算法树 -->
      <template v-else-if="!error">
        <view class="algorithm-section">
          <text class="section-label">算法列表 ({{ leafCount }})</text>
          <view class="algorithm-tree">
            <template v-for="node in displayTree" :key="node.id">
              <AlgorithmTreeNode
                :node="node"
                :level="0"
                :expanded-keys="expandedKeys"
                :favorite-ids="favoriteIds"
                :compare-list="compareList"
                @toggle-expand="toggleExpand"
                @select="handleSelect"
                @toggle-favorite="handleToggleFavorite"
                @show-detail="handleShowDetail"
                @toggle-compare="toggleCompare"
              />
            </template>
          </view>

          <!-- 空状态 -->
          <view v-if="displayTree.length === 0" class="empty-state">
            <view class="empty-tip">{{
              searchKeyword ? "未找到匹配的算法" : "暂无可用算法"
            }}</view>
          </view>
        </view>
      </template>

      <!-- 错误状态 -->
      <view v-if="error" class="error-state">
        <text class="error-text">{{ error }}</text>
        <button class="retry-btn" @click="loadData">重新加载</button>
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

    <!-- 算法详情弹窗 -->
    <view v-if="showDetail" class="detail-overlay" @click="showDetail = false">
      <view class="detail-panel" @click.stop>
        <view class="detail-header">
          <text class="detail-title">算法详情</text>
          <view class="detail-close" @click="showDetail = false">
            <SvgIcon name="close" size="20" color="#6b7280" />
          </view>
        </view>
        <scroll-view v-if="detailLoading" class="detail-body" scroll-y>
          <view class="loading-container">
            <text>加载中...</text>
          </view>
        </scroll-view>
        <scroll-view v-else-if="detailData" class="detail-body" scroll-y>
          <view class="detail-section">
            <text class="detail-section-title">基本信息</text>
            <view class="detail-item">
              <text class="detail-label">算法名称</text>
              <text class="detail-value">{{ detailData.name }}</text>
            </view>
            <view v-if="detailData.type" class="detail-item">
              <text class="detail-label">算法类型</text>
              <text class="detail-value">{{ detailData.type }}</text>
            </view>
            <view v-if="detailData.version" class="detail-item">
              <text class="detail-label">版本</text>
              <text class="detail-value">{{ detailData.version }}</text>
            </view>
            <view v-if="detailData.avgRating !== undefined" class="detail-item">
              <text class="detail-label">评分</text>
              <text class="detail-value"
                >{{ detailData.avgRating }} / 5 ({{
                  detailData.ratingCount || 0
                }}
                评价)</text
              >
            </view>
            <view
              v-if="detailData.usageCount !== undefined"
              class="detail-item"
            >
              <text class="detail-label">使用次数</text>
              <text class="detail-value">{{ detailData.usageCount }}</text>
            </view>
          </view>
          <view v-if="detailData.description" class="detail-section">
            <text class="detail-section-title">算法描述</text>
            <text class="detail-desc">{{ detailData.description }}</text>
          </view>
          <view
            v-if="detailData.size || detailData.params || detailData.flops"
            class="detail-section"
          >
            <text class="detail-section-title">性能指标</text>
            <view v-if="detailData.size" class="detail-item">
              <text class="detail-label">模型大小</text>
              <text class="detail-value">{{ detailData.size }}</text>
            </view>
            <view v-if="detailData.params" class="detail-item">
              <text class="detail-label">参数量</text>
              <text class="detail-value">{{ detailData.params }}</text>
            </view>
            <view v-if="detailData.flops" class="detail-item">
              <text class="detail-label">FLOPs</text>
              <text class="detail-value">{{ detailData.flops }}</text>
            </view>
          </view>
          <view
            v-if="detailData.sampleImages && detailData.sampleImages.length > 0"
            class="detail-section"
          >
            <text class="detail-section-title">效果样例</text>
            <scroll-view class="sample-scroll" scroll-x>
              <view class="sample-list">
                <image
                  v-for="(url, idx) in detailData.sampleImages"
                  :key="idx"
                  :src="url"
                  class="sample-image"
                  mode="aspectFill"
                />
              </view>
            </scroll-view>
          </view>
          <view v-if="processingStore.hasImage" class="detail-section">
            <text class="detail-section-title">自定义测试</text>
            <image
              v-if="testResult"
              :src="testResult"
              class="test-result-image"
              mode="widthFix"
            />
            <button
              v-else
              class="test-btn"
              :loading="testLoading"
              @click="handleCustomTest"
            >
              使用当前图片测试效果
            </button>
          </view>
        </scroll-view>
        <view class="detail-footer">
          <button
            class="footer-btn outline"
            @click="detailData && handleToggleFavorite(detailData.id)"
          >
            {{
              detailData && favoriteIds.has(detailData.id) ? "取消收藏" : "收藏"
            }}
          </button>
          <button class="footer-btn primary" @click="handleUseFromDetail">
            立即使用
          </button>
        </view>
      </view>
    </view>

    <!-- 算法对比弹窗 -->
    <view
      v-if="showCompare"
      class="detail-overlay"
      @click="showCompare = false"
    >
      <view class="compare-panel" @click.stop>
        <view class="detail-header">
          <text class="detail-title">算法对比</text>
          <view class="detail-close" @click="showCompare = false">
            <SvgIcon name="close" size="20" color="#6b7280" />
          </view>
        </view>
        <scroll-view class="detail-body" scroll-y>
          <view v-if="compareLoading" class="loading-container">
            <text>对比中...</text>
          </view>
          <view
            v-else-if="compareResult && compareResult.length > 0"
            class="compare-table"
          >
            <view class="compare-row header-row">
              <view class="compare-cell label-cell"><text>指标</text></view>
              <view
                v-for="c in compareResult"
                :key="c.algorithmId"
                class="compare-cell"
              >
                <text class="compare-alg-name">{{ c.algorithmName }}</text>
              </view>
            </view>
            <view class="compare-row">
              <view class="compare-cell label-cell"><text>处理耗时</text></view>
              <view
                v-for="c in compareResult"
                :key="c.algorithmId"
                class="compare-cell"
              >
                <text>{{ c.time ? c.time + "ms" : "-" }}</text>
              </view>
            </view>
            <view
              v-if="compareResult.some((c) => c.resultUrl)"
              class="compare-row"
            >
              <view class="compare-cell label-cell"><text>效果预览</text></view>
              <view
                v-for="c in compareResult"
                :key="c.algorithmId"
                class="compare-cell"
              >
                <image
                  v-if="c.resultUrl"
                  :src="c.resultUrl"
                  class="compare-preview-img"
                  mode="aspectFill"
                />
                <text v-else>-</text>
              </view>
            </view>
          </view>
          <view v-else class="empty-state">
            <view class="empty-tip">暂无对比数据</view>
          </view>
        </scroll-view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import AlgorithmTreeNode from "./components/AlgorithmTreeNode.vue";
import { useProcessingStore } from "@/store/processing";
import { AlgorithmAPI, FavoriteAPI, RecommendationAPI } from "dehaze-sdk-js";
import type {
  AlgorithmSelectNodeVO,
  AlgorithmDetailVO,
  RecommendedAlgorithm,
  AlgorithmCompareVO,
} from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

// ==================== 常量 ====================
const SEARCH_HISTORY_KEY = "alg_select_search_history";
const SEARCH_HISTORY_MAX = 10;
const COMPARE_MAX = 3;

// ==================== 状态 ====================
const processingStore = useProcessingStore();
const loading = ref(false);
const error = ref("");
const tree = ref<AlgorithmSelectNodeVO[]>([]);
const expandedKeys = ref<Set<number>>(new Set());
const selectedAlgorithm = ref<AlgorithmSelectNodeVO | null>(null);

// 收藏
const favoriteIds = ref<Set<number>>(new Set());
const favoriteMap = ref<Map<number, number>>(new Map());
const togglingIds = ref<Set<number>>(new Set());

// 搜索
const searchKeyword = ref("");
const searchResults = ref<AlgorithmSelectNodeVO[] | null>(null);
const searchHistory = ref<string[]>([]);
const showHistory = ref(false);
let searchTimer: ReturnType<typeof setTimeout> | null = null;

// 推荐
const recommendList = ref<RecommendedAlgorithm[]>([]);
const recommendLoading = ref(false);
const imageAnalysis = ref<{ hazeLevel: string; sceneType: string } | null>(
  null
);

// 详情
const showDetail = ref(false);
const detailData = ref<AlgorithmDetailVO | null>(null);
const detailLoading = ref(false);
const testResult = ref<string | null>(null);
const testLoading = ref(false);

// 对比
const compareList = ref<AlgorithmSelectNodeVO[]>([]);
const compareResult = ref<AlgorithmCompareVO[] | null>(null);
const compareLoading = ref(false);
const showCompare = ref(false);

// ==================== 计算属性 ====================
const leafCount = computed(() => {
  const walk = (nodes: AlgorithmSelectNodeVO[]): number => {
    let count = 0;
    for (const n of nodes) {
      if (n.children && n.children.length > 0) count += walk(n.children);
      else if (n.leaf) count++;
    }
    return count;
  };
  return walk(tree.value);
});

const displayTree = computed(() => {
  if (searchResults.value !== null) return searchResults.value;
  return tree.value;
});

// ==================== 方法 ====================
async function loadData() {
  loading.value = true;
  error.value = "";
  try {
    const [data, favPage] = await Promise.all([
      AlgorithmAPI.tree(),
      FavoriteAPI.getPage({
        targetType: "algorithm",
        pageNum: 1,
        pageSize: 200,
      }).catch(() => null),
    ]);
    tree.value = data || [];
    expandedKeys.value = new Set((data || []).map((n) => n.id));
    favoriteIds.value = new Set();
    favoriteMap.value = new Map();
    if (favPage?.list) {
      for (const fav of favPage.list) {
        favoriteIds.value.add(fav.targetId);
        favoriteMap.value.set(fav.targetId, fav.id);
      }
    }
  } catch (e) {
    error.value = getErrorMessage(e, "加载失败");
  } finally {
    loading.value = false;
  }
}

async function loadRecommendations() {
  if (!processingStore.originUrl) return;
  recommendLoading.value = true;
  try {
    const analysis = await RecommendationAPI.analyze({
      imageUrl: processingStore.originUrl,
    });
    imageAnalysis.value = {
      hazeLevel: analysis.hazeLevel || "",
      sceneType: analysis.sceneType || "",
    };
    if (!analysis.imageMd5) {
      recommendList.value = [];
      return;
    }
    const list = await RecommendationAPI.getAlgorithmRecommendations({
      imageMd5: analysis.imageMd5,
    });
    recommendList.value = list || [];
  } catch {
    recommendList.value = [];
  } finally {
    recommendLoading.value = false;
  }
}

function toggleExpand(id: number) {
  const next = new Set(expandedKeys.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  expandedKeys.value = next;
}

function handleSelect(node: AlgorithmSelectNodeVO) {
  selectedAlgorithm.value = node;
  processingStore.setAlgorithm({
    id: node.id,
    name: node.name,
    parentId: node.parentId,
    type: node.type || "",
    description: "",
  } as any);
}

function collectLeaf(nodes: AlgorithmSelectNodeVO[]): AlgorithmSelectNodeVO[] {
  const result: AlgorithmSelectNodeVO[] = [];
  for (const n of nodes) {
    if (n.children?.length) result.push(...collectLeaf(n.children));
    else if (n.leaf) result.push(n);
  }
  return result;
}

function handleSelectRecommend(item: RecommendedAlgorithm) {
  const allLeaf = collectLeaf(tree.value);
  const node = allLeaf.find((n) => n.id === item.algorithmId);
  if (node) handleSelect(node);
  else uni.showToast({ title: "算法不在列表中", icon: "none" });
}

async function handleToggleFavorite(algorithmId: number) {
  if (togglingIds.value.has(algorithmId)) return;
  togglingIds.value = new Set(togglingIds.value).add(algorithmId);
  try {
    const existed = favoriteIds.value.has(algorithmId);
    if (existed) {
      const favId = favoriteMap.value.get(algorithmId);
      if (favId) await FavoriteAPI.deleteByIds([favId]);
      const nextIds = new Set(favoriteIds.value);
      nextIds.delete(algorithmId);
      favoriteIds.value = nextIds;
      const nextMap = new Map(favoriteMap.value);
      nextMap.delete(algorithmId);
      favoriteMap.value = nextMap;
      uni.showToast({ title: "已取消收藏", icon: "none" });
    } else {
      const favId = await FavoriteAPI.add({
        targetType: "algorithm",
        targetId: algorithmId,
      });
      favoriteIds.value = new Set(favoriteIds.value).add(algorithmId);
      favoriteMap.value = new Map(favoriteMap.value).set(algorithmId, favId);
      uni.showToast({ title: "已收藏", icon: "none" });
    }
  } catch (e) {
    uni.showToast({ title: getErrorMessage(e, "操作失败"), icon: "none" });
  } finally {
    const next = new Set(togglingIds.value);
    next.delete(algorithmId);
    togglingIds.value = next;
  }
}

function onSearchInput() {
  if (searchTimer) clearTimeout(searchTimer);
  const kw = searchKeyword.value.trim();
  if (!kw) {
    searchResults.value = null;
    return;
  }
  searchTimer = setTimeout(async () => {
    try {
      const results = await AlgorithmAPI.search(kw);
      searchResults.value = results || [];
    } catch {
      searchResults.value = [];
    }
  }, 300);
}

function handleSearchSubmit() {
  const kw = searchKeyword.value.trim();
  if (!kw) return;
  const list = searchHistory.value.filter((k) => k !== kw);
  list.unshift(kw);
  const trimmed = list.slice(0, SEARCH_HISTORY_MAX);
  searchHistory.value = trimmed;
  try {
    uni.setStorageSync(SEARCH_HISTORY_KEY, JSON.stringify(trimmed));
  } catch {}
  showHistory.value = false;
  onSearchInput();
}

function useHistory(kw: string) {
  searchKeyword.value = kw;
  handleSearchSubmit();
}

function clearSearch() {
  searchKeyword.value = "";
  searchResults.value = null;
}

function clearHistory() {
  searchHistory.value = [];
  try {
    uni.removeStorageSync(SEARCH_HISTORY_KEY);
  } catch {}
}

function onSearchBlur() {
  setTimeout(() => {
    showHistory.value = false;
  }, 200);
}

async function handleShowDetail(node: AlgorithmSelectNodeVO) {
  detailLoading.value = true;
  showDetail.value = true;
  testResult.value = null;
  try {
    const detail = await AlgorithmAPI.getSelectDetail(node.id);
    detailData.value = detail;
  } catch (e) {
    uni.showToast({ title: getErrorMessage(e, "加载详情失败"), icon: "none" });
    showDetail.value = false;
  } finally {
    detailLoading.value = false;
  }
}

async function handleCustomTest() {
  if (!detailData.value || !processingStore.originUrl) return;
  testLoading.value = true;
  try {
    const result = await AlgorithmAPI.test(detailData.value.id, {
      imageUrl: processingStore.originUrl,
    });
    testResult.value = result?.resultUrl || "";
  } catch (e) {
    uni.showToast({ title: getErrorMessage(e, "测试失败"), icon: "none" });
  } finally {
    testLoading.value = false;
  }
}

function handleUseFromDetail() {
  if (!detailData.value) return;
  selectedAlgorithm.value = {
    id: detailData.value.id,
    parentId: 0,
    name: detailData.value.name,
    type: detailData.value.type || "",
    leaf: true,
  };
  showDetail.value = false;
}

function toggleCompare(node: AlgorithmSelectNodeVO) {
  const idx = compareList.value.findIndex((c) => c.id === node.id);
  if (idx >= 0) {
    compareList.value = compareList.value.filter((c) => c.id !== node.id);
  } else {
    if (compareList.value.length >= COMPARE_MAX) {
      uni.showToast({ title: `最多对比 ${COMPARE_MAX} 个算法`, icon: "none" });
      return;
    }
    compareList.value = [...compareList.value, node];
  }
}

async function handleCompare() {
  if (compareList.value.length < 2) {
    uni.showToast({ title: "至少选择 2 个算法", icon: "none" });
    return;
  }
  compareLoading.value = true;
  showCompare.value = true;
  try {
    const result = await AlgorithmAPI.compare({
      algorithmIds: compareList.value.map((c) => c.id),
      imageUrl: processingStore.originUrl || undefined,
    });
    compareResult.value = result || [];
  } catch (e) {
    uni.showToast({ title: getErrorMessage(e, "对比失败"), icon: "none" });
    showCompare.value = false;
  } finally {
    compareLoading.value = false;
  }
}

function handleNext() {
  if (!selectedAlgorithm.value) {
    uni.showToast({ title: "请选择算法", icon: "none" });
    return;
  }
  if (!processingStore.hasImage) {
    uni.showToast({ title: "请先选择图片", icon: "none" });
    return;
  }
  uni.navigateTo({ url: "/pages/processing/index" });
}

// ==================== 生命周期 ====================
onMounted(() => {
  loadData();
  if (processingStore.hasImage) loadRecommendations();
  try {
    const raw = uni.getStorageSync(SEARCH_HISTORY_KEY);
    if (raw) searchHistory.value = JSON.parse(raw);
  } catch {}
});
</script>

<style lang="scss" scoped>
@import "@/styles/mixins.scss";

.algorithm-select-page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: 24rpx;
  @include safe-area-bottom(180rpx);
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
  background: $color-white;
  border-radius: 20rpx;
  padding: 20rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.preview-image {
  width: 120rpx;
  height: 120rpx;
  border-radius: 16rpx;
  background: $color-bg-secondary;
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
  color: $color-text-primary;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 8rpx;
}
.preview-size {
  font-size: 24rpx;
  color: $color-text-placeholder;
}

/* 智能推荐 */
.recommend-section {
  margin-bottom: 24rpx;
}
.analysis-tags {
  display: flex;
  gap: 12rpx;
  margin-bottom: 16rpx;
}
.analysis-tag {
  padding: 6rpx 16rpx;
  font-size: 22rpx;
  color: $color-secondary;
  background: #eef2ff;
  border-radius: 8rpx;
}
.recommend-loading {
  display: flex;
  align-items: center;
  gap: 16rpx;
  padding: 40rpx 0;
  justify-content: center;
}
.recommend-loading-text {
  font-size: 28rpx;
  color: $color-text-placeholder;
}
.recommend-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.recommend-card {
  display: flex;
  gap: 20rpx;
  align-items: center;
  padding: 24rpx;
  background: $color-white;
  border-radius: 20rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.recommend-rank {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 56rpx;
  height: 56rpx;
  font-size: 24rpx;
  font-weight: 700;
  color: $color-white;
  background: linear-gradient(135deg, $color-primary, $color-secondary);
  border-radius: 50%;
}
.recommend-info {
  flex: 1;
  min-width: 0;
}
.recommend-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8rpx;
}
.recommend-name {
  font-size: 28rpx;
  font-weight: 600;
  color: $color-text-primary;
}
.recommend-score {
  font-size: 24rpx;
  color: $color-accent;
  font-weight: 600;
  flex-shrink: 0;
}
.recommend-reason {
  display: block;
  font-size: 22rpx;
  color: $color-primary;
  margin-bottom: 8rpx;
}
.match-score-bar {
  height: 8rpx;
  background: $color-border;
  border-radius: 4rpx;
  overflow: hidden;
}
.match-score-fill {
  height: 100%;
  background: linear-gradient(90deg, $color-primary, $color-accent);
  border-radius: 4rpx;
}
.fav-btn {
  flex-shrink: 0;
  padding: 16rpx;
}

/* 搜索 */
.search-bar {
  display: flex;
  align-items: center;
  gap: 16rpx;
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.search-input {
  flex: 1;
  font-size: 28rpx;
  color: $color-text-primary;
}
.search-placeholder {
  color: $color-text-placeholder;
  font-size: 28rpx;
}
.search-clear {
  padding: 16rpx 24rpx;
  min-width: 88rpx;
  min-height: 88rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 搜索历史 */
.search-history-panel {
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12rpx;
}
.history-title {
  font-size: 26rpx;
  color: $color-text-secondary;
}
.history-clear {
  font-size: 24rpx;
  color: $color-primary;
}
.history-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
}
.history-tag {
  padding: 8rpx 20rpx;
  font-size: 24rpx;
  color: #4b5563;
  background: $color-bg-secondary;
  border-radius: 20rpx;
}

/* 对比栏 */
.compare-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16rpx 24rpx;
  background: $color-white;
  border-radius: 16rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.04);
}
.compare-bar-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8rpx;
  flex: 1;
  min-width: 0;
}
.compare-bar-tag {
  display: flex;
  align-items: center;
  gap: 4rpx;
  padding: 6rpx 16rpx;
  font-size: 22rpx;
  color: $color-accent;
  background: #ede9fe;
  border-radius: 16rpx;
}
.compare-bar-remove {
  padding: 4rpx;
}
.compare-bar-btn {
  flex-shrink: 0;
  padding: 12rpx 28rpx;
  font-size: 24rpx;
  font-weight: 600;
  color: $color-white;
  background: $color-accent;
  border-radius: 24rpx;
}

/* 算法树 */
.algorithm-section {
  margin-bottom: 24rpx;
}
.algorithm-tree {
  background: $color-white;
  border-radius: 20rpx;
  overflow: hidden;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
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
  color: $color-text-placeholder;
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
  color: $color-danger;
  margin-bottom: 24rpx;
}
.retry-btn {
  padding: 16rpx 48rpx;
  background: $color-accent;
  color: $color-white;
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
  color: $color-accent;
}
.no-selection {
  font-size: 26rpx;
  color: $color-text-placeholder;
}
.next-btn {
  padding: 20rpx 48rpx;
  background: linear-gradient(135deg, $color-accent, $color-secondary);
  color: $color-white;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
  font-weight: 600;
  white-space: nowrap;
}
.next-btn:disabled {
  background: $color-text-disabled;
  color: $color-text-placeholder;
}

/* 详情/对比弹窗 */
.detail-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.4);
  z-index: 200;
  display: flex;
  align-items: flex-end;
}
.detail-panel {
  width: 100%;
  height: 80%;
  background: $color-white;
  border-radius: 32rpx 32rpx 0 0;
  display: flex;
  flex-direction: column;
}
.compare-panel {
  width: 100%;
  height: 75%;
  background: $color-white;
  border-radius: 32rpx 32rpx 0 0;
  display: flex;
  flex-direction: column;
}
.detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 32rpx 40rpx;
  border-bottom: 2rpx solid #f1f5f9;
  flex-shrink: 0;
}
.detail-title {
  font-size: 34rpx;
  font-weight: 600;
  color: $color-text-primary;
}
.detail-close {
  padding: 8rpx;
}
.detail-body {
  flex: 1;
  padding: 24rpx 40rpx 32rpx;
}
.detail-section {
  margin-bottom: 28rpx;
}
.detail-section-title {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: $color-text-primary;
  margin-bottom: 16rpx;
  padding-bottom: 8rpx;
  border-bottom: 2rpx solid $color-border-light;
}
.detail-desc {
  display: block;
  font-size: 26rpx;
  line-height: 1.6;
  color: #4b5563;
}
.detail-item {
  display: flex;
  align-items: center;
  padding: 16rpx 0;
}
.detail-label {
  flex-shrink: 0;
  width: 160rpx;
  font-size: 26rpx;
  color: $color-text-secondary;
}
.detail-value {
  flex: 1;
  font-size: 28rpx;
  color: $color-text-primary;
}
.sample-scroll {
  white-space: nowrap;
}
.sample-list {
  display: inline-flex;
  gap: 16rpx;
}
.sample-image {
  width: 240rpx;
  height: 180rpx;
  border-radius: 12rpx;
  background: $color-bg-secondary;
  flex-shrink: 0;
}
.test-result-image {
  width: 100%;
  border-radius: 12rpx;
}
.test-btn {
  padding: 16rpx 32rpx;
  font-size: 26rpx;
  color: $color-accent;
  background: #ede9fe;
  border: none;
  border-radius: 12rpx;
}
.detail-footer {
  display: flex;
  gap: 24rpx;
  padding: 24rpx 40rpx;
  border-top: 2rpx solid #f1f5f9;
  flex-shrink: 0;
}
.footer-btn {
  flex: 1;
  padding: 20rpx 0;
  font-size: 28rpx;
  border-radius: 16rpx;
  border: none;
  text-align: center;
}
.footer-btn.outline {
  color: $color-accent;
  background: #ede9fe;
}
.footer-btn.primary {
  color: $color-white;
  background: linear-gradient(135deg, $color-accent, $color-secondary);
}

/* 对比表格 */
.compare-table {
}
.compare-row {
  display: flex;
  border-bottom: 2rpx solid $color-border-light;
}
.compare-row.header-row {
  background: $color-bg-primary;
  font-weight: 600;
}
.compare-cell {
  flex: 1;
  padding: 20rpx 12rpx;
  font-size: 26rpx;
  color: #4b5563;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
}
.compare-cell.label-cell {
  flex-shrink: 0;
  width: 140rpx;
  color: $color-text-secondary;
  font-size: 24rpx;
}
.compare-alg-name {
  font-weight: 600;
  color: $color-text-primary;
}
.compare-preview-img {
  width: 120rpx;
  height: 90rpx;
  border-radius: 8rpx;
  background: $color-bg-secondary;
}
</style>
