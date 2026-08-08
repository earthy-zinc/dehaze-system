<template>
  <PageLayout level="L2" title="我的文件" class="page">
    <view class="main-content">
      <!-- 加载状态 -->
      <view v-if="loading" class="loading-container">
        <up-loading-icon mode="circle" size="40" color="#10b981" />
        <text class="loading-text">加载中...</text>
      </view>

      <!-- 文件列表 -->
      <view v-else-if="files.length > 0" class="file-list">
        <view
          v-for="file in files"
          :key="file.id"
          class="file-card"
          @click="handleClick(file)"
        >
          <view class="file-icon">
            <SvgIcon :name="getIcon(file.type)" size="24" color="#10b981" />
          </view>
          <view class="file-info">
            <text class="file-name">{{ file.name }}</text>
            <text class="file-meta">
              {{ formatFileSize(file.size || "") }} ·
              {{ formatRelativeTime(file.createTime || "") }}
            </text>
          </view>
          <view class="file-arrow">
            <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
          </view>
        </view>

        <view v-if="!hasMore" class="end-text">— 没有更多了 —</view>
        <view v-else class="load-more" @click="loadMore">
          <text class="load-more-text">加载更多</text>
        </view>
      </view>

      <!-- 空状态 -->
      <view v-else class="empty-state">
        <up-empty mode="list" text="暂无文件" />
        <text class="empty-hint">上传图片后文件会显示在这里</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { FileAPI } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import { formatFileSize, formatRelativeTime } from "@/utils/format";

const loading = ref(false);
const files = ref<FileInfo[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;
  try {
    const result = await FileAPI.getPage({ pageNum: page, pageSize: 20 });
    if (page === 1) {
      files.value = result.list;
    } else {
      files.value = [...files.value, ...result.list];
    }
    hasMore.value = files.value.length < result.total;
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

function handleClick(file: FileInfo) {
  if (file.url) {
    uni.setClipboardData({
      data: file.url,
      success: () => uni.showToast({ title: "URL 已复制", icon: "success" }),
    });
  }
}

function getIcon(type?: string): string {
  if (!type) return "file";
  const t = type.toLowerCase();
  if (t.includes("image")) return "photo";
  if (t.includes("video")) return "play-circle";
  if (t.includes("audio")) return "mic";
  if (t.includes("pdf") || t.includes("doc")) return "file-text";
  return "file";
}

onMounted(() => loadData());
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
  padding-bottom: calc(80rpx + constant(safe-area-inset-bottom));
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.file-card {
  display: flex;
  align-items: center;
  gap: 20rpx;
  background: #fff;
  border-radius: 20rpx;
  padding: 24rpx;
  box-shadow: $shadow-sm;
  &:active {
    background: #f9fafb;
  }
}
.file-icon {
  width: 80rpx;
  height: 80rpx;
  background: #ecfdf5;
  border-radius: 16rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.file-info {
  flex: 1;
  min-width: 0;
}
.file-name {
  display: block;
  font-size: $font-md;
  font-weight: 500;
  color: $color-text-primary;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 6rpx;
}
.file-meta {
  font-size: $font-sm;
  color: $color-text-placeholder;
}
.file-arrow {
  flex-shrink: 0;
}

.end-text {
  text-align: center;
  font-size: $font-sm;
  color: $color-text-disabled;
  padding: 32rpx 0;
}
.load-more {
  text-align: center;
  padding: 24rpx 0;
}
.load-more-text {
  font-size: $font-sm;
  color: #10b981;
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
.empty-hint {
  font-size: $font-sm;
  color: $color-text-placeholder;
  margin-top: 16rpx;
}
</style>
