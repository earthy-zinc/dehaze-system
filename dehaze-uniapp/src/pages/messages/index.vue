<template>
  <PageLayout level="L1" title="消息">
    <view class="messages-page">
      <!-- 分类筛选占位（规划：系统通知/处理完成/活动） -->
      <view class="tabs">
        <view
          class="tab"
          v-for="tab in tabs"
          :key="tab"
          :class="{ active: tab === '全部' }"
        >
          {{ tab }}
        </view>
      </view>

      <!-- 消息列表占位 -->
      <view class="empty-state">
        <u-icon name="bell" size="56" color="#d1d5db" />
        <text class="empty-text">暂无消息</text>
        <text class="empty-sub">处理完成、系统通知等将在这里展示</text>
      </view>

      <!-- 开发中提示 -->
      <view class="dev-tip">
        <u-icon name="info-circle" size="14" color="#9ca3af" />
        <text>消息通知模块建设中，将对接消息通知 API</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import PageLayout from "@/layout/index.vue";

/** 消息分类（05 规划） */
const tabs = ["全部", "系统通知", "处理完成", "活动"];
</script>

<style lang="scss" scoped>
.messages-page {
  padding: 24rpx;
  background: #f9fafb;
  min-height: 100vh;
}

.tabs {
  display: flex;
  gap: 16rpx;
  margin-bottom: 24rpx;

  .tab {
    padding: 12rpx 28rpx;
    border-radius: 999rpx;
    font-size: 24rpx;
    background: #f3f4f6;
    color: #6b7280;

    &.active {
      background: #3b82f6;
      color: #ffffff;
    }
  }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  background: #ffffff;
  border-radius: 20rpx;
  padding: 96rpx 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);

  .empty-text {
    font-size: 30rpx;
    font-weight: 600;
    color: #374151;
  }

  .empty-sub {
    font-size: 24rpx;
    color: #9ca3af;
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
