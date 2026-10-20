<template>
  <PageLayout level="L2" title="订单管理">
    <view class="page-body">
      <view class="stats-row" v-if="stats">
        <view class="stat-item"
          ><text class="stat-val">{{ stats.totalOrders || 0 }}</text
          ><text class="stat-label">总订单</text></view
        >
        <view class="stat-item"
          ><text class="stat-val">¥{{ fenToYuan(stats.totalRevenue) }}</text
          ><text class="stat-label">总收入</text></view
        >
        <view class="stat-item"
          ><text class="stat-val">{{ fenToYuan(stats.totalRefund) }}</text
          ><text class="stat-label">总退款</text></view
        >
      </view>
      <view class="tabs">
        <view
          v-for="(t, i) in tabs"
          :key="i"
          class="tab-item"
          :class="{ active: currentTab === i }"
          @click="onTabChange(i)"
        >
          {{ t.name }}
        </view>
      </view>
      <template v-if="currentTab === 0">
        <view class="list-row list-row-head">
          <text class="cell">订单号</text>
          <text class="cell">金额</text>
          <text class="cell">状态</text>
          <text class="cell"></text>
        </view>
        <view
          v-for="item in list"
          :key="item.orderNo"
          class="list-row"
          @click="goDetail(item.orderNo)"
        >
          <text class="cell">{{ item.orderNo }}</text>
          <text class="cell">¥{{ fenToYuan(item.payableAmount) }}</text>
          <view class="cell">
            <view
              class="tag tag-sm"
              :class="'tag-' + orderTagType(item.status)"
            >
              {{ orderStatusMap[item.status] || item.status }}
            </view>
          </view>
          <view class="cell"><SvgIcon name="arrow-right" /></view>
        </view>
        <view v-if="!loading && list.length === 0" class="empty-tip"
          >暂无订单</view
        >
        <view class="load-more" v-if="hasMore" @click="loadMore">加载更多</view>
      </template>
      <template v-if="currentTab === 1">
        <view class="list-row list-row-head">
          <text class="cell">订单号</text>
          <text class="cell">退款金额</text>
          <text class="cell">原因</text>
          <text class="cell">操作</text>
        </view>
        <view v-for="item in refunds" :key="item.id" class="list-row">
          <text class="cell">{{ item.orderNo }}</text>
          <text class="cell">¥{{ fenToYuan(item.refundAmount) }}</text>
          <text class="cell">{{ item.reason }}</text>
          <view class="cell cell-actions">
            <button
              v-if="item.status === 'refunding'"
              class="btn btn-success btn-sm"
              @click="approveRefund(item.id)"
            >
              通过
            </button>
            <button
              v-if="item.status === 'refunding'"
              class="btn btn-danger btn-sm"
              @click="rejectRefund(item.id)"
            >
              拒绝
            </button>
          </view>
        </view>
        <view v-if="!refundsLoading && refunds.length === 0" class="empty-tip"
          >暂无退款申请</view
        >
      </template>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { usePagedList } from "@/composables/usePagedList";
import { OrderAPI } from "clarity-sdk-js";
import { fenToYuan } from "@/utils/format";

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

const { list, hasMore, loading, fetchList, loadMore } = usePagedList<any>({
  fetcher: (p) =>
    OrderAPI.getPage({
      pageNum: p.pageNum,
      pageSize: 20,
    }).then((r) => r.list || []),
});

const refunds = ref<any[]>([]);
const refundsLoading = ref(false);
const stats = ref<any>(null);

const fetchStats = async () => {
  try {
    stats.value = await OrderAPI.getStats();
  } catch {}
};
const fetchRefunds = async () => {
  refundsLoading.value = true;
  try {
    const res = await OrderAPI.listRefunds({ pageNum: 1, pageSize: 100 });
    refunds.value = res.list || [];
  } catch {
    refunds.value = [];
  } finally {
    refundsLoading.value = false;
  }
};

const onTabChange = (i: number) => {
  currentTab.value = i;
  if (i === 0) fetchList(true);
  else fetchRefunds();
};

const orderTagType = (s: string) => {
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
  padding: 20rpx;
  text-align: center;
  background: $color-white;
  border-radius: $radius-lg;
}

.stat-val {
  display: block;
  font-size: 36rpx;
  font-weight: bold;
  color: $color-primary;
}

.stat-label {
  margin-top: 8rpx;
  font-size: 24rpx;
  color: $color-text-secondary;
}

.tabs {
  display: flex;
  margin-bottom: 20rpx;
  overflow: hidden;
  background: $color-white;
  border-radius: $radius-lg;
}

.tab-item {
  flex: 1;
  padding: 20rpx;
  font-size: 28rpx;
  color: $color-text-secondary;
  text-align: center;
}

.tab-item.active {
  font-weight: 600;
  color: $color-primary;
  border-bottom: 4rpx solid $color-primary;
}

.list-row {
  display: flex;
  align-items: center;
  padding: 20rpx 16rpx;
  font-size: 26rpx;
  border-bottom: 1rpx solid $color-border;

  .cell {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cell-actions {
    display: flex;
    gap: 8rpx;
  }
}

.list-row-head {
  font-weight: 600;
  color: $color-text-secondary;
  background: $color-bg-secondary;
}

.tag {
  padding: 4rpx 12rpx;
  font-size: $font-xs;
  border-radius: $radius-sm;
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

.btn {
  padding: 8rpx 20rpx;
  font-size: $font-sm;
  line-height: 1.6;
  border-radius: $radius-sm;

  &::after {
    border: none;
  }
}

.btn-sm {
  padding: 4rpx 16rpx;
  font-size: $font-xs;
}

.btn-success {
  color: $color-white;
  background: $color-success;
}

.btn-danger {
  color: $color-white;
  background: $color-danger;
}

.load-more {
  padding: 20rpx;
  color: $color-text-secondary;
  text-align: center;
}
</style>
