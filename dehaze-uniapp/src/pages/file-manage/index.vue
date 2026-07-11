<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="folder" size="28" color="#10b981" />
        </view>
        <view class="header-text">
          <text class="header-title">文件管理</text>
          <text class="header-subtitle">查看已上传的文件列表</text>
        </view>
      </view>

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
            <u-icon :name="getIcon(file.type)" size="24" color="#10b981" />
          </view>
          <view class="file-info">
            <text class="file-name">{{ file.name }}</text>
            <text class="file-meta">
              {{ formatSize(file.size) }} · {{ formatDate(file.createTime) }}
            </text>
          </view>
          <view class="file-arrow">
            <u-icon name="arrow-right" size="16" color="#d1d5db" />
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
import PageLayout from "@/layout/index.vue";
import { getFileList, type SysFile } from "@/api/file";

const loading = ref(false);
const files = ref<SysFile[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;

  try {
    const result = await getFileList(page, 20);
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

function handleClick(file: SysFile) {
  if (file.url) {
    // 复制文件 URL
    uni.setClipboardData({
      data: file.url,
      success: () => {
        uni.showToast({ title: "URL 已复制", icon: "success" });
      },
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

function formatSize(size?: string): string {
  if (!size || size === "0") return "0 B";
  const bytes = parseInt(size, 10);
  if (isNaN(bytes)) return size;
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatDate(time?: string): string {
  if (!time) return "-";
  const d = new Date(time);
  const now = Date.now();
  const diff = now - d.getTime();
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
  if (diff < 172800000) return "昨天";
  return d.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}

onMounted(() => loadData());
</script>

<style lang="scss" scoped>
.page { width: 100%; min-height: 100vh; background: #f9fafb; }
.main-content { padding: 24rpx; padding-bottom: calc(80rpx + constant(safe-area-inset-bottom)); }

.page-header-card {
  display: flex; align-items: center; gap: 24rpx;
  background: #fff; border-radius: 24rpx; padding: 32rpx; margin-bottom: 24rpx;
  box-shadow: 0 4rpx 16rpx rgba(0,0,0,0.06);
}
.header-icon { width: 80rpx; height: 80rpx; background: #d1fae5; border-radius: 20rpx; display: flex; align-items: center; justify-content: center; }
.header-title { font-size: 36rpx; font-weight: 700; color: #1f2937; display: block; margin-bottom: 8rpx; }
.header-subtitle { font-size: 26rpx; color: #6b7280; display: block; }

.file-list { display: flex; flex-direction: column; gap: 16rpx; }
.file-card {
  display: flex; align-items: center; gap: 20rpx;
  background: #fff; border-radius: 20rpx; padding: 24rpx;
  box-shadow: 0 2rpx 8rpx rgba(0,0,0,0.03);
  &:active { background: #f9fafb; }
}
.file-icon {
  width: 80rpx; height: 80rpx; background: #ecfdf5; border-radius: 16rpx;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.file-info { flex: 1; min-width: 0; }
.file-name { display: block; font-size: 28rpx; font-weight: 500; color: #1f2937; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 6rpx; }
.file-meta { font-size: 24rpx; color: #9ca3af; }
.file-arrow { flex-shrink: 0; }

.end-text { text-align: center; font-size: 24rpx; color: #d1d5db; padding: 32rpx 0; }
.load-more { text-align: center; padding: 24rpx 0; }
.load-more-text { font-size: 26rpx; color: #10b981; }

.loading-container { display: flex; flex-direction: column; align-items: center; padding: 120rpx 0; }
.loading-text { margin-top: 24rpx; font-size: 28rpx; color: #9ca3af; }

.empty-state { display: flex; flex-direction: column; align-items: center; padding: 80rpx 0; }
.empty-hint { font-size: 26rpx; color: #9ca3af; margin-top: 16rpx; }
</style>
