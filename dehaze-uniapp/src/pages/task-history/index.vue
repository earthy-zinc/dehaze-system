<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="clock" size="28" color="#6366f1" />
        </view>
        <view class="header-text">
          <text class="header-title">处理历史</text>
          <text class="header-subtitle">查看所有去雾处理记录</text>
        </view>
      </view>

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <up-loading-icon mode="circle" size="40" color="#6366f1" />
        <text class="loading-text">加载中...</text>
      </view>

      <!-- 列表 -->
      <view v-else-if="records.length > 0" class="record-list">
        <view
          v-for="record in records"
          :key="record.id"
          class="record-card"
          @click="handleClick(record)"
        >
          <view class="record-thumb" v-if="record.predUrl">
            <image :src="record.predUrl" class="thumb-img" mode="aspectFill" />
          </view>
          <view class="record-body">
            <text class="record-algo">{{
              record.algorithmName || "未知算法"
            }}</text>
            <text class="record-time">
              耗时 {{ record.time ? record.time + "s" : "-" }}
            </text>
            <text class="record-date">{{ formatDate(record.createTime) }}</text>
          </view>
          <view class="record-arrow">
            <u-icon name="arrow-right" size="16" color="#d1d5db" />
          </view>
        </view>

        <!-- 底部提示 -->
        <view v-if="!hasMore" class="end-text">— 没有更多了 —</view>
        <view v-else class="load-more" @click="loadMore">
          <text>加载更多</text>
        </view>
      </view>

      <!-- 空状态 -->
      <view v-else class="empty-state">
        <up-empty mode="list" text="暂无处理记录" />
        <text class="empty-hint">去雾处理过的图片会显示在这里</text>
        <button class="start-btn" @click="handleStart">开始去雾处理</button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import {
  getPredictionLogs,
  type PredLogVO,
  type PageResult,
} from "@/api/prediction";

const loading = ref(false);
const records = ref<PredLogVO[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;

  try {
    const result = await getPredictionLogs({ pageNum: page, pageSize: 15 });
    if (page === 1) {
      records.value = result.list;
    } else {
      records.value = [...records.value, ...result.list];
    }
    hasMore.value = records.value.length < result.total;
    currentPage.value = page;
  } catch (e) {
    console.warn("加载历史失败:", e);
    uni.showToast({ title: "加载失败", icon: "none" });
  } finally {
    loading.value = false;
  }
}

function loadMore() {
  if (hasMore.value) {
    loadData(currentPage.value + 1);
  }
}

function handleClick(record: PredLogVO) {
  if (record.predUrl) {
    uni.previewImage({ urls: [record.predUrl], current: record.predUrl });
  }
}

function handleStart() {
  uni.switchTab({ url: "/pages/image-input/index" });
}

function formatDate(time?: string): string {
  if (!time) return "-";
  const d = new Date(time);
  const now = Date.now();
  const diff = now - d.getTime();
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
  if (diff < 172800000) return "昨天";
  return d.toLocaleDateString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

onMounted(() => loadData());
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: #f9fafb;
}
.main-content {
  padding: 24rpx;
  padding-bottom: calc(80rpx + constant(safe-area-inset-bottom));
}

.page-header-card {
  display: flex;
  align-items: center;
  gap: 24rpx;
  background: #fff;
  border-radius: 24rpx;
  padding: 32rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
}
.header-icon {
  width: 80rpx;
  height: 80rpx;
  background: #e0e7ff;
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}
.header-title {
  font-size: 36rpx;
  font-weight: 700;
  color: #1f2937;
  display: block;
  margin-bottom: 8rpx;
}
.header-subtitle {
  font-size: 26rpx;
  color: #6b7280;
  display: block;
}

.record-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.record-card {
  display: flex;
  align-items: center;
  gap: 20rpx;
  background: #fff;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.03);
  &:active {
    background: #f9fafb;
  }
}
.record-thumb {
  width: 100rpx;
  height: 100rpx;
  border-radius: 16rpx;
  overflow: hidden;
  flex-shrink: 0;
  background: #f3f4f6;
}
.thumb-img {
  width: 100%;
  height: 100%;
}
.record-body {
  flex: 1;
  min-width: 0;
}
.record-algo {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 6rpx;
}
.record-time {
  display: block;
  font-size: 24rpx;
  color: #6b7280;
  margin-bottom: 4rpx;
}
.record-date {
  font-size: 22rpx;
  color: #9ca3af;
}
.record-arrow {
  flex-shrink: 0;
}

.end-text {
  text-align: center;
  font-size: 24rpx;
  color: #d1d5db;
  padding: 32rpx 0;
}
.load-more {
  text-align: center;
  font-size: 26rpx;
  color: #6366f1;
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
  font-size: 28rpx;
  color: #9ca3af;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}
.empty-hint {
  font-size: 26rpx;
  color: #9ca3af;
  margin: 16rpx 0 32rpx;
}
.start-btn {
  padding: 16rpx 48rpx;
  background: #6366f1;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
