<template>
  <PageLayout level="L2" title="订单详情">
    <view class="page-body">
      <view class="info-card">
        <view class="info-row"
          ><text class="label">订单号</text
          ><text>{{ order.orderNo }}</text></view
        >
        <view class="info-row"
          ><text class="label">套餐</text
          ><text>{{ order.packageName }}</text></view
        >
        <view class="info-row"
          ><text class="label">用户</text
          ><text>{{ order.username }} (ID:{{ order.userId }})</text></view
        >
        <view class="info-row"
          ><text class="label">应付金额</text
          ><text>¥{{ order.payableAmount }}</text></view
        >
        <view class="info-row"
          ><text class="label">实付金额</text
          ><text>¥{{ order.paidAmount }}</text></view
        >
        <view class="info-row"
          ><text class="label">状态</text
          ><view class="tag tag-sm" :class="'tag-' + tagType(order.status)">
            {{ statusMap[order.status] || order.status }}
          </view></view
        >
        <view class="info-row"
          ><text class="label">创建时间</text
          ><text>{{ order.createTime }}</text></view
        >
        <view class="info-row"
          ><text class="label">支付时间</text
          ><text>{{ order.paidTime || "-" }}</text></view
        >
        <view class="info-row" v-if="order.expireTime"
          ><text class="label">到期时间</text
          ><text>{{ order.expireTime }}</text></view
        >
      </view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { reactive } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import { OrderAPI } from "dehaze-sdk-js";

const statusMap: Record<string, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};
const order = reactive<any>({});

onLoad((options: any) => {
  const orderNo = options?.orderNo || options?.id;
  if (orderNo) fetchDetail(String(orderNo));
});

const fetchDetail = async (orderNo: string) => {
  try {
    const res = await OrderAPI.getDetail(orderNo);
    Object.assign(order, res);
  } catch {}
};
const tagType = (s: string) => {
  switch (s) {
    case "completed":
      return "success";
    case "cancelled":
      return "danger";
    case "paid":
      return "primary";
    case "refunding":
      return "warning";
    case "refunded":
      return "info";
    default:
      return "warning";
  }
};
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.info-card {
  background: $color-white;
  border-radius: $radius-lg;
  padding: 20rpx;
}
.info-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16rpx 0;
  border-bottom: 1rpx solid $color-border;
}
.info-row:last-child {
  border-bottom: none;
}
.label {
  color: $color-text-secondary;
}
.tag {
  padding: 4rpx 12rpx;
  border-radius: $radius-sm;
  font-size: $font-xs;
}
.tag-sm {
  padding: 2rpx 10rpx;
}
.tag-primary {
  color: $color-primary;
  background: $color-primary-bg;
}
.tag-success {
  color: $color-success;
  background: $color-success-bg;
}
.tag-warning {
  color: $color-warning;
  background: $color-warning-bg;
}
.tag-danger {
  color: $color-danger;
  background: $color-danger-bg;
}
.tag-info {
  color: $color-text-secondary;
  background: $color-bg-secondary;
}
</style>
