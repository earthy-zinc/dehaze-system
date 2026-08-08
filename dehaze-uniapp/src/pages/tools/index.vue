<template>
  <PageLayout level="L1" title="工具">
    <view class="tools-page">
      <!-- 全局搜索栏 -->
      <view class="tools-search">
        <SvgIcon name="search" size="18" color="#9ca3af" />
        <input
          class="tools-search-input"
          placeholder="搜索算法、功能、文档..."
          :value="searchKeyword"
          @input="onSearchInput"
          @confirm="onSearchConfirm"
        />
        <view
          v-if="searchKeyword"
          class="tools-search-clear"
          @click="clearSearch"
        >
          <SvgIcon name="close-circle-fill" size="18" color="#9ca3af" />
        </view>
      </view>

      <!-- 快捷入口横滑区 -->
      <view class="tools-quick-section">
        <text class="tools-section-label">快捷入口</text>
        <scroll-view scroll-x class="tools-quick-scroll">
          <view class="tools-quick-row">
            <view
              v-for="entry in quickEntries"
              :key="entry.label"
              class="tools-quick-item"
              @click="handleQuickClick(entry)"
            >
              <view class="tools-quick-icon">
                <SvgIcon :name="entry.icon" size="22" color="#3b82f6" />
              </view>
              <text class="tools-quick-label">{{ entry.label }}</text>
            </view>
          </view>
        </scroll-view>
      </view>

      <!-- 功能网格 -->
      <view class="tools-grid-section">
        <text class="tools-section-label">全部功能</text>
        <view class="tools-grid">
          <view
            v-for="entry in gridEntries"
            :key="entry.label"
            class="tools-grid-item"
            @click="handleGridClick(entry)"
          >
            <view class="tools-grid-icon">
              <SvgIcon :name="entry.icon" size="24" color="#3b82f6" />
            </view>
            <text class="tools-grid-label">{{ entry.label }}</text>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";

interface QuickEntry {
  label: string;
  icon: string;
  target: string;
}

interface GridEntry {
  label: string;
  icon: string;
  target: string;
  isTab?: boolean;
}

const searchKeyword = ref("");

/** 快捷入口（高频功能横滑直达） */
const quickEntries: QuickEntry[] = [
  { label: "处理历史", icon: "clock", target: "/pages/task-history/index" },
  { label: "我的收藏", icon: "star", target: "/pages/task-history/index" },
  { label: "批量处理", icon: "car", target: "/pages/task-history/index" },
  { label: "算法选择", icon: "grid", target: "/pages/algorithm-select/index" },
];

/** 功能网格（工具/浏览类功能，管理类归「我的」） */
const gridEntries: GridEntry[] = [
  { label: "图像输入", icon: "camera", target: "/pages/image-input/index" },
  { label: "算法库", icon: "gift", target: "/pages/algorithm/index" },
  { label: "数据集", icon: "server-fill", target: "/pages/dataset/index" },
  { label: "批量处理", icon: "car", target: "/pages/task-history/index" },
  { label: "指标管理", icon: "integral", target: "/pages/metrics/index" },
  { label: "API 文档", icon: "file-text", target: "" },
];

const onSearchInput = (e: any) => {
  searchKeyword.value = e.detail.value;
};

const onSearchConfirm = () => {
  if (!searchKeyword.value) return;
  uni.navigateTo({
    url: `/pages/algorithm-select/index?keyword=${encodeURIComponent(searchKeyword.value)}`,
  });
};

const clearSearch = () => {
  searchKeyword.value = "";
};

const handleQuickClick = (entry: QuickEntry) => {
  uni.navigateTo({ url: entry.target });
};

const handleGridClick = (entry: GridEntry) => {
  if (entry.label === "API 文档") {
    uni.showToast({ title: "API 文档功能开发中，敬请期待", icon: "none" });
    return;
  }
  if (entry.isTab) {
    uni.switchTab({ url: entry.target });
  } else {
    uni.navigateTo({ url: entry.target });
  }
};
</script>

<style lang="scss" scoped>
.tools-page {
  padding: 24rpx;
  background: $color-bg-primary;
  min-height: 100vh;
}

.tools-search {
  display: flex;
  align-items: center;
  gap: 12rpx;
  height: 80rpx;
  padding: 0 24rpx;
  border-radius: 999rpx;
  background: $color-bg-secondary;
  margin-bottom: 24rpx;

  .tools-search-input {
    flex: 1;
    height: 100%;
    font-size: $font-sm;
    color: $color-text-primary;
    background: transparent;
  }

  .tools-search-clear {
    width: 36rpx;
    height: 36rpx;
    display: flex;
    align-items: center;
    justify-content: center;
  }
}

.tools-section-label {
  display: block;
  font-size: $font-xs;
  color: $color-text-secondary;
  margin-bottom: 16rpx;
  padding: 0 8rpx;
}

.tools-quick-section {
  margin-bottom: 32rpx;
}

.tools-quick-scroll {
  white-space: nowrap;

  .tools-quick-row {
    display: inline-flex;
    gap: 24rpx;
    padding: 0 8rpx;
  }

  .tools-quick-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10rpx;
    width: 120rpx;
  }

  .tools-quick-icon {
    width: 88rpx;
    height: 88rpx;
    border-radius: 24rpx;
    background: $color-primary-bg;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .tools-quick-label {
    font-size: $font-xs;
    color: $color-text-primary;
  }
}

.tools-grid-section {
  .tools-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20rpx;
  }

  .tools-grid-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12rpx;
    padding: 24rpx 0;
    background: $color-white;
    border-radius: 20rpx;
    box-shadow: $shadow-sm;
  }

  .tools-grid-icon {
    width: 72rpx;
    height: 72rpx;
    border-radius: 20rpx;
    background: $color-primary-bg;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .tools-grid-label {
    font-size: $font-xs;
    color: $color-text-primary;
  }
}
</style>
