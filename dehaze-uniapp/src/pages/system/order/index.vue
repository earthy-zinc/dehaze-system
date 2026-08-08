<template>
  <PageLayout level="L2" title="订单管理">
    <view class="page-body">
      <view class="stats-row" v-if="stats">
        <view class="stat-item"
          ><text class="stat-val">{{ stats.totalOrders || 0 }}</text
          ><text class="stat-label">总订单</text></view
        >
        <view class="stat-item"
          ><text class="stat-val">¥{{ stats.totalRevenue || 0 }}</text
          ><text class="stat-label">总收入</text></view
        >
        <view class="stat-item"
          ><text class="stat-val">{{ stats.totalRefund || 0 }}</text
          ><text class="stat-label">总退款</text></view
        >
      </view>
      <u-tabs
        :list="tabs"
        :current="currentTab"
        @change="
          (i: any) => {
            currentTab = i.index;
            if (i.index === 0) fetchList(true);
            else fetchRefunds();
          }
        "
      />
      <u-table v-if="currentTab === 0">
        <u-tr v-for="item in list" :key="item.orderNo" @click="goDetail(item.orderNo)">
          <u-td>{{ item.orderNo }}</u-td>
          <u-td>¥{{ item.payableAmount }}</u-td>
          <u-td>
            <u-tag
              :text="orderStatusMap[item.status] || item.status"
              :type="orderTagType(item.status)"
              size="mini"
            />
          </u-td>
          <u-td><SvgIcon name="arrow-right" /></u-td>
        </u-tr>
      </u-table>
      <u-table v-if="currentTab === 1">
        <u-tr v-for="item in refunds" :key="item.id">
          <u-td>{{ item.orderNo }}</u-td>
          <u-td>¥{{ item.refundAmount }}</u-td>
          <u-td>{{ item.reason }}</u-td>
          <u-td>
            <u-button
              v-if="item.status === 'refunding'"
              size="mini"
              type="success"
              @click="approveRefund(item.id)"
              >通过</u-button
            >
            <u-button
              v-if="item.status === 'refunding'"
              size="mini"
              type="error"
              @click="rejectRefund(item.id)"
              >拒绝</u-button
            >
          </u-td>
        </u-tr>
      </u-table>
      <u-empty
        v-if="!loading && currentTab === 0 && list.length === 0"
        text="暂无订单"
      />
      <u-empty
        v-if="!loading && currentTab === 1 && refunds.length === 0"
        text="暂无退款申请"
      />
      <view
        class="load-more"
        v-if="currentTab === 0 && hasMore"
        @click="loadMore"
        >加载更多</view
      >
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { OrderAPI } from "dehaze-sdk-js";

const tabs = [{ name: "订单列表" }, { name: "退款审核" }];
const orderStatusMap: Record<string, string> = {
  pending: "待支付",
  paid: "已支付",
  completed: "已完成",
  cancelled: "已取消",
  refunding: "退款中",
  refunded: "已退款",
};
const currentTab = ref(0);
const list = ref<any[]>([]);
const refunds = ref<any[]>([]);
const stats = ref<any>(null);
const pageNum = ref(1);
const hasMore = ref(false);
const loading = ref(false);

const fetchStats = async () => {
  try {
    stats.value = await OrderAPI.getStats();
  } catch {}
};
const fetchList = async (reset = false) => {
  if (reset) {
    pageNum.value = 1;
    list.value = [];
  }
  loading.value = true;
  try {
    const res = await OrderAPI.getPage({
      pageNum: pageNum.value,
      pageSize: 20,
    });
    const records = res.list || [];
    if (reset) list.value = records;
    else list.value.push(...records);
    hasMore.value = records.length === 20;
    pageNum.value++;
  } finally {
    loading.value = false;
  }
};
const fetchRefunds = async () => {
  try {
    const res = await OrderAPI.listRefunds({ pageNum: 1, pageSize: 100 });
    refunds.value = res.list || [];
  } catch {}
};

const loadMore = () => fetchList();
const orderTagType = (s: string) => {
  switch (s) {
    case "completed": return "success";
    case "cancelled": return "error";
    case "paid": return "primary";
    case "refunding": return "warning";
    case "refunded": return "info";
    default: return "warning";
  }
};
const goDetail = (orderNo: string) =>
  uni.navigateTo({ url: `/pages/system/order/detail?orderNo=${orderNo}` });
const approveRefund = async (id: number) => {
  try {
    await OrderAPI.approveRefund(id, { approved: true, remark: "审核通过" });
    fetchRefunds();
    fetchStats();
    uni.showToast({ title: "已通过", icon: "success" });
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};
const rejectRefund = async (id: number) => {
  try {
    await OrderAPI.rejectRefund(id, { approved: false, remark: "审核拒绝" });
    fetchRefunds();
    fetchStats();
    uni.showToast({ title: "已拒绝", icon: "success" });
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
};

fetchStats();
fetchList(true);
fetchRefunds();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.stats-row {
  display: flex;
  gap: 16rpx;
  margin-bottom: 20rpx;
}
.stat-item {
  flex: 1;
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx;
  text-align: center;
}
.stat-val {
  font-size: 36rpx;
  font-weight: bold;
  color: $color-primary;
  display: block;
}
.stat-label {
  font-size: 24rpx;
  color: $color-text-secondary;
  margin-top: 8rpx;
}
.load-more {
  text-align: center;
  padding: 20rpx;
  color: $color-text-secondary;
}
</style>
