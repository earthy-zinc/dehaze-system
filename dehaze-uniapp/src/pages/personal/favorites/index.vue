<template>
  <PageLayout level="L2" title="我的收藏" class="page">
    <view class="main-content">
      <!-- 搜索栏 -->
      <view class="search-bar">
        <SvgIcon name="search" size="18" color="#9ca3af" />
        <input
          v-model="keyword"
          class="search-input"
          type="text"
          placeholder="搜索收藏内容..."
          placeholder-class="search-placeholder"
          @confirm="handleSearch"
        />
        <view v-if="keyword" class="search-clear" @click="clearSearch">
          <SvgIcon name="close-circle-fill" size="16" color="#9ca3af" />
        </view>
      </view>

      <!-- 类型筛选 Tab -->
      <scroll-view scroll-x class="type-bar" :show-scrollbar="false">
        <view
          v-for="t in typeTabs"
          :key="t.key"
          class="type-item"
          :class="{ active: typeFilter === t.key }"
          @click="switchType(t.key)"
        >
          <text>{{ t.label }}</text>
        </view>
      </scroll-view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
        <text class="loading-text">加载中...</text>
      </view>

      <!-- 收藏列表 -->
      <view v-else-if="favorites.length > 0" class="fav-list">
        <view v-for="fav in favorites" :key="fav.id" class="fav-card">
          <view class="fav-info">
            <text class="fav-name">{{ fav.targetName || "未命名" }}</text>
            <view class="fav-meta">
              <view
                class="fav-tag"
                :style="{
                  backgroundColor: getTypeColor(fav.targetType) + '20',
                }"
              >
                <text
                  class="fav-tag-text"
                  :style="{ color: getTypeColor(fav.targetType) }"
                >
                  {{ getTypeLabel(fav.targetType) }}
                </text>
              </view>
              <text class="fav-time">{{
                formatRelativeTime(fav.createTime)
              }}</text>
            </view>
          </view>
          <view class="fav-action" @click.stop="removeFavorite(fav.id)">
            <SvgIcon name="star-fill" size="20" color="#f59e0b" />
          </view>
        </view>
        <view v-if="!hasMore" class="end-text">— 没有更多了 —</view>
        <view v-else class="load-more" @click="loadMore">加载更多</view>
      </view>

      <!-- 空状态 -->
      <view v-else class="empty-state">
        <view class="empty-tip">暂无收藏</view>
        <text class="empty-hint">收藏的处理结果或算法会显示在这里</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { FavoriteAPI } from "dehaze-sdk-js";
import type { FavoriteVO, FavoriteTargetType } from "dehaze-sdk-js";
import { formatRelativeTime } from "@/utils/format";

const loading = ref(false);
const favorites = ref<FavoriteVO[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);
const keyword = ref("");
const typeFilter = ref<"" | FavoriteTargetType>("");

const typeTabs = [
  { key: "" as const, label: "全部" },
  { key: "algorithm" as const, label: "算法" },
  { key: "result" as const, label: "结果" },
  { key: "dataset" as const, label: "数据集" },
  { key: "image" as const, label: "图片" },
  { key: "preset" as const, label: "预设" },
];

function getTypeColor(type: string): string {
  const map: Record<string, string> = {
    algorithm: "#3b82f6",
    result: "#10b981",
    dataset: "#f59e0b",
    image: "#ec4899",
    preset: "#7c3aed",
  };
  return map[type] || "#9ca3af";
}

function getTypeLabel(type: string): string {
  const tab = typeTabs.find((t) => t.key === type);
  return tab?.label || type;
}

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;
  try {
    const params: Record<string, any> = { pageNum: page, pageSize: 20 };
    if (typeFilter.value) params.targetType = typeFilter.value;
    if (keyword.value.trim()) params.keywords = keyword.value.trim();
    const result = await FavoriteAPI.getPage(params);
    if (page === 1) favorites.value = result.list;
    else favorites.value = [...favorites.value, ...result.list];
    hasMore.value = favorites.value.length < result.total;
    currentPage.value = page;
  } catch {
    uni.showToast({ title: "加载失败", icon: "none" });
  } finally {
    loading.value = false;
  }
}

function loadMore() {
  if (hasMore.value) loadData(currentPage.value + 1);
}

function switchType(key: "" | FavoriteTargetType) {
  typeFilter.value = key;
  currentPage.value = 1;
  loadData(1);
}

function handleSearch() {
  currentPage.value = 1;
  loadData(1);
}

function clearSearch() {
  keyword.value = "";
  currentPage.value = 1;
  loadData(1);
}

async function removeFavorite(id: number) {
  try {
    await FavoriteAPI.deleteByIds([id]);
    favorites.value = favorites.value.filter((f) => f.id !== id);
    uni.showToast({ title: "已取消收藏", icon: "success" });
  } catch {
    uni.showToast({ title: "操作失败", icon: "none" });
  }
}

onMounted(() => loadData());
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
  @include safe-area-bottom(80rpx);
}

/* 搜索栏 */
.search-bar {
  display: flex;
  align-items: center;
  gap: 16rpx;
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx 24rpx;
  margin-bottom: 16rpx;
  box-shadow: $shadow-sm;
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
  padding: 4rpx;
}

/* 类型筛选 */
.type-bar {
  display: flex;
  white-space: nowrap;
  margin-bottom: 16rpx;
  padding-bottom: 8rpx;
}
.type-item {
  display: inline-flex;
  flex-shrink: 0;
  padding: 10rpx 28rpx;
  margin-right: 16rpx;
  background: $color-bg-secondary;
  border-radius: 32rpx;
  font-size: 26rpx;
  color: $color-text-secondary;
}
.type-item.active {
  background: #fef3c7;
  font-weight: 500;
  color: #d97706;
}

/* 列表 */
.fav-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.fav-card {
  display: flex;
  align-items: center;
  gap: 16rpx;
  background: $color-white;
  border-radius: $radius-lg;
  padding: 24rpx;
  box-shadow: $shadow-sm;
}
.fav-info {
  flex: 1;
  min-width: 0;
}
.fav-name {
  font-size: $font-md;
  font-weight: 500;
  color: $color-text-primary;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 8rpx;
}
.fav-meta {
  display: flex;
  align-items: center;
  gap: 16rpx;
}
.fav-tag {
  padding: 2rpx 12rpx;
  border-radius: 8rpx;
}
.fav-tag-text {
  font-size: 20rpx;
  font-weight: 500;
}
.fav-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.fav-action {
  flex-shrink: 0;
  padding: 8rpx;
}

.end-text {
  text-align: center;
  font-size: $font-sm;
  color: $color-text-disabled;
  padding: 32rpx 0;
}
.load-more {
  text-align: center;
  font-size: $font-sm;
  color: $color-secondary;
  padding: 24rpx 0;
}
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.loading-text {
  margin-top: 24rpx;
  font-size: $font-md;
  color: $color-text-placeholder;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}
.empty-tip {
  font-size: $font-md;
}
.empty-hint {
  font-size: $font-sm;
  color: $color-text-placeholder;
  margin-top: 16rpx;
}
</style>
