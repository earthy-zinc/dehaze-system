<template>
  <PageLayout level="L2" title="处理历史" class="page">
    <view class="main-content">
      <PageHeaderCard
        icon="clock"
        icon-color="#6366f1"
        icon-bg="#e0e7ff"
        title="处理历史"
        subtitle="查看所有去雾处理记录"
      />

      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
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
            <text class="record-date">{{
              formatRelativeTime(record.createTime || "")
            }}</text>
            <view class="record-actions" @click.stop>
              <view
                class="action-btn compare-btn"
                :class="{ disabled: !record.predUrl || !record.originUrl }"
                @click="handleCompare(record)"
              >
                <SvgIcon name="grid" size="14" color="#3b82f6" />
                <text>对比</text>
              </view>
              <view
                class="action-btn reprocess-btn"
                :class="{ disabled: !record.originUrl }"
                @click="handleReprocess(record)"
              >
                <SvgIcon name="reload" size="14" color="#f59e0b" />
                <text>重新处理</text>
              </view>
            </view>
          </view>
          <view class="record-arrow">
            <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
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
        <view class="empty-tip">暂无处理记录</view>
        <text class="empty-hint">去雾处理过的图片会显示在这里</text>
        <button class="start-btn" @click="handleStart">开始去雾处理</button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import { AlgorithmAPI, ModelAPI } from "dehaze-sdk-js";
import type { PredLogVO } from "dehaze-sdk-js";
import { useProcessingStore } from "@/store/processing";
import type { ImageData } from "@/pages/image-input/data/imageInputData";
import { formatRelativeTime } from "@/utils/format";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { getErrorMessage } from "@/utils/error";

const processingStore = useProcessingStore();
const loading = ref(false);
const records = ref<PredLogVO[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;

  try {
    const result = await ModelAPI.getPredLogs({ pageNum: page, pageSize: 15 });
    if (page === 1) {
      records.value = result.list;
    } else {
      records.value = [...records.value, ...result.list];
    }
    hasMore.value = records.value.length < result.total;
    currentPage.value = page;
  } catch {
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

/** 从历史记录构造 ImageData（无尺寸信息时省略，由展示层条件渲染） */
function buildImageData(originUrl: string): ImageData {
  return {
    url: originUrl,
    name: originUrl.split("/").pop() || "历史图片",
  };
}

/** 对比：跳转到并排对比页，注入原图与结果图 */
function handleCompare(record: PredLogVO) {
  if (!record.predUrl || !record.originUrl) {
    uni.showToast({ title: "缺少原图或结果图，无法对比", icon: "none" });
    return;
  }
  processingStore.reset();
  processingStore.setImage(buildImageData(record.originUrl));
  processingStore.complete({
    // 有结果图即已完成（后端 LogStatusEnum 序列化为整数：2 = COMPLETED）
    status: 2,
    resultUrl: record.predUrl,
    time: record.time || 0,
  });
  uni.navigateTo({ url: "/pages/side-by-side/index" });
}

/** 重新处理：拉取算法详情后跳转到处理页 */
async function handleReprocess(record: PredLogVO) {
  if (!record.originUrl) {
    uni.showToast({ title: "缺少原图，无法重新处理", icon: "none" });
    return;
  }
  if (!record.algorithmId) {
    uni.showToast({ title: "缺少算法信息", icon: "none" });
    return;
  }
  uni.showLoading({ title: "准备中...", mask: true });
  try {
    const algorithm = await AlgorithmAPI.getAlgorithmInfoById(
      record.algorithmId
    );
    processingStore.reset();
    processingStore.setImage(buildImageData(record.originUrl));
    processingStore.setAlgorithm(algorithm);
    uni.hideLoading();
    uni.navigateTo({ url: "/pages/processing/index" });
  } catch (e) {
    uni.hideLoading();
    const msg = getErrorMessage(e, "算法信息加载失败");
    uni.showToast({ title: msg, icon: "none" });
  }
}

function handleStart() {
  uni.reLaunch({ url: "/pages/image-input/index" });
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
  padding: 24rpx;
  @include safe-area-bottom(80rpx);
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
  background: $color-white;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.03);
  &:active {
    background: $color-bg-primary;
  }
}
.record-thumb {
  width: 100rpx;
  height: 100rpx;
  border-radius: 16rpx;
  overflow: hidden;
  flex-shrink: 0;
  background: $color-bg-secondary;
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
  color: $color-text-primary;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 6rpx;
}
.record-time {
  display: block;
  font-size: 24rpx;
  color: $color-text-secondary;
  margin-bottom: 4rpx;
}
.record-date {
  display: block;
  font-size: 22rpx;
  color: $color-text-placeholder;
  margin-bottom: 12rpx;
}
.record-actions {
  display: flex;
  gap: 16rpx;
}
.action-btn {
  display: flex;
  align-items: center;
  gap: 6rpx;
  padding: 16rpx 32rpx;
  border-radius: 24rpx;
  font-size: 22rpx;
  font-weight: 500;

  &.disabled {
    opacity: 0.4;
    pointer-events: none;
  }

  &:active:not(.disabled) {
    opacity: 0.7;
  }
}
.compare-btn {
  background: #dbeafe;
  color: $color-primary;
}
.reprocess-btn {
  background: #fef3c7;
  color: $color-warning;
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
  color: $color-text-placeholder;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}
.empty-tip {
  font-size: 28rpx;
}
.empty-hint {
  font-size: 26rpx;
  color: $color-text-placeholder;
  margin: 16rpx 0 32rpx;
}
.start-btn {
  padding: 16rpx 48rpx;
  background: #6366f1;
  color: $color-white;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
