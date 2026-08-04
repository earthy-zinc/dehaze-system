<template>
  <PageLayout level="L1" title="工具">
    <view class="tools-page">
      <!-- 搜索栏（规划：全局搜索算法/功能/文档） -->
      <view class="search-bar">
        <u-icon name="search" size="18" color="#9ca3af" />
        <text class="search-placeholder">搜索算法、功能...</text>
      </view>

      <!-- 快捷入口横滑区（规划：处理历史/我的收藏/批量处理/算法选择） -->
      <scroll-view scroll-x class="quick-scroll">
        <view class="quick-row">
          <view class="quick-item" v-for="item in quickEntries" :key="item">
            <view class="quick-icon">
              <u-icon name="clock" size="20" color="#3b82f6" />
            </view>
            <text class="quick-label">{{ item }}</text>
          </view>
        </view>
      </scroll-view>

      <!-- 功能网格（规划：数据集/算法库/指标管理/批量处理/图像输入/API文档） -->
      <view class="grid-section">
        <text class="section-label">全部功能</text>
        <view class="grid">
          <view class="grid-item" v-for="item in gridEntries" :key="item">
            <view class="grid-icon">
              <u-icon name="grid" size="22" color="#3b82f6" />
            </view>
            <text class="grid-label">{{ item }}</text>
          </view>
        </view>
      </view>

      <!-- 开发中提示 -->
      <view class="dev-tip">
        <u-icon name="info-circle" size="14" color="#9ca3af" />
        <text>工具聚合中心建设中，功能入口将按规划接入</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import PageLayout from "@/layout/index.vue";

/** 快捷入口（规划：高频功能横滑直达） */
const quickEntries = ["处理历史", "我的收藏", "批量处理", "算法选择"];

/** 功能网格（规划：工具/浏览类功能，管理类归「我的」） */
const gridEntries = [
  "数据集",
  "算法库",
  "指标管理",
  "批量处理",
  "图像输入",
  "API文档",
];
</script>

<style lang="scss" scoped>
.tools-page {
  padding: 24rpx;
  background: #f9fafb;
  min-height: 100vh;
}

.search-bar {
  display: flex;
  align-items: center;
  gap: 12rpx;
  height: 80rpx;
  padding: 0 24rpx;
  border-radius: 999rpx;
  background: #f3f4f6;
  margin-bottom: 24rpx;

  .search-placeholder {
    font-size: 26rpx;
    color: #9ca3af;
  }
}

.quick-scroll {
  white-space: nowrap;
  margin-bottom: 32rpx;

  .quick-row {
    display: inline-flex;
    gap: 24rpx;
    padding: 0 8rpx;
  }

  .quick-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10rpx;
    width: 120rpx;
  }

  .quick-icon {
    width: 88rpx;
    height: 88rpx;
    border-radius: 24rpx;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .quick-label {
    font-size: 22rpx;
    color: #374151;
  }
}

.grid-section {
  .section-label {
    display: block;
    font-size: 24rpx;
    color: #6b7280;
    margin-bottom: 16rpx;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20rpx;
  }

  .grid-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12rpx;
    padding: 24rpx 0;
    background: #ffffff;
    border-radius: 20rpx;
    box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
  }

  .grid-icon {
    width: 72rpx;
    height: 72rpx;
    border-radius: 20rpx;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .grid-label {
    font-size: 22rpx;
    color: #374151;
  }
}

.dev-tip {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  margin-top: 48rpx;
  font-size: 22rpx;
  color: #9ca3af;
}
</style>
