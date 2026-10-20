<template>
  <PageLayout level="L2" title="我的订单" class="page">
    <view class="main-content">
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
        <text class="loading-text">加载中...</text>
      </view>

      <view v-else-if="orders.length > 0" class="order-list">
        <view
          v-for="order in orders"
          :key="order.orderNo"
          class="order-card"
          @click="handleDetail(order)"
        >
          <view class="order-header">
            <text class="order-no">{{ order.orderNo }}</text>
            <text :class="['order-status', statusClass(order.status)]">{{
              statusText(order.status)
            }}</text>
          </view>
          <view class="order-body">
            <text class="order-package">{{ order.packageName }}</text>
            <text class="order-amount"
              >¥{{ fenToYuan(order.payableAmount) }}</text
            >
          </view>
          <view class="order-footer">
            <text class="order-time">{{
              formatRelativeTime(order.createTime)
            }}</text>
          </view>
        </view>
        <view v-if="!hasMore" class="end-text">— 没有更多了 —</view>
        <view v-else class="load-more" @click="loadMore">加载更多</view>
      </view>

      <view v-else class="empty-state">
        <view class="empty-tip">暂无订单</view>
        <text class="empty-hint">购买套餐后订单会显示在这里</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { OrderAPI } from "clarity-sdk-js";
import type { MyOrderVO, OrderStatus } from "clarity-sdk-js";
import { fenToYuan, formatRelativeTime } from "@/utils/format";

const loading = ref(false);
const orders = ref<MyOrderVO[]>([]);
const currentPage = ref(1);
const hasMore = ref(true);

async function loadData(page = 1) {
  if (loading.value) return;
  loading.value = true;
  try {
    const result = await OrderAPI.listMy({ pageNum: page, pageSize: 20 });
    if (page === 1) {
      orders.value = result.list;
    } else {
      orders.value = [...orders.value, ...result.list];
    }
    hasMore.value = orders.value.length < result.total;
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

function handleDetail(order: MyOrderVO) {
  uni.showToast({ title: `订单 ${order.orderNo} 详情`, icon: "none" });
}

function statusText(status: OrderStatus): string {
  const map: Record<string, string> = {
    pending: "待支付",
    paid: "已支付",
    completed: "已完成",
    cancelled: "已取消",
    refunding: "退款中",
    refunded: "已退款",
  };
  return map[status] || status;
}

function statusClass(status: OrderStatus): string {
  const map: Record<string, string> = {
    pending: "status-pending",
    paid: "status-paid",
    completed: "status-done",
    cancelled: "status-cancel",
    refunding: "status-warn",
    refunded: "status-done",
  };
  return map[status] || "";
}

onMounted(() => loadData());
</script>

<style lang="scss" scoped>
@import "@/styles/mixins";

.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}

.main-content {
  padding: $spacing-md;

  @include safe-area-bottom(80rpx);
}

.order-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.order-card {
  padding: 24rpx;
  background: $color-white;
  border-radius: 20rpx;
  box-shadow: $shadow-sm;

  &:active {
    background: $color-bg-primary;
  }
}

.order-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12rpx;
}

.order-no {
  font-family: monospace;
  font-size: $font-sm;
  color: $color-text-secondary;
}

.order-status {
  padding: 4rpx 12rpx;
  font-size: $font-xs;
  font-weight: 500;
  border-radius: 8rpx;
}

.status-pending {
  color: $color-warning;
  background: #fef3c7;
}

.status-paid {
  color: $color-primary;
  background: #dbeafe;
}

.status-done {
  color: $color-success;
  background: #ecfdf5;
}

.status-cancel {
  color: $color-text-placeholder;
  background: $color-bg-secondary;
}

.status-warn {
  color: $color-danger;
  background: #fef2f2;
}

.order-body {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12rpx;
}

.order-package {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
}

.order-amount {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-warning;
}

.order-footer {
}

.order-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.end-text {
  padding: 32rpx 0;
  font-size: $font-sm;
  color: $color-text-disabled;
  text-align: center;
}

.load-more {
  padding: 24rpx 0;
  font-size: $font-sm;
  color: $color-secondary;
  text-align: center;
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
  margin-top: 16rpx;
  font-size: $font-sm;
  color: $color-text-placeholder;
}
</style>
