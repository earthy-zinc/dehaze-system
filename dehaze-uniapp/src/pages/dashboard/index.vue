<template>
  <PageLayout level="L1" title="工作台" :isHome="true" :showSearch="false">
    <view class="page-body">
      <view class="stats-section">
        <view class="section-title">数据概览</view>
        <view class="stats-grid">
          <view
            class="stat-card"
            v-for="item in statsItems"
            :key="item.key"
            @click="goPage(item.path)"
          >
            <text class="stat-val">{{ item.value }}</text>
            <text class="stat-label">{{ item.label }}</text>
          </view>
        </view>
      </view>
      <view class="shortcut-section">
        <view class="section-title">管理功能</view>
        <view class="shortcut-grid">
          <view
            class="shortcut-item"
            v-for="item in shortcuts"
            :key="item.path"
            @click="goPage(item.path)"
          >
            <SvgIcon :name="item.icon" size="28" color="$color-primary" />
            <text class="shortcut-label">{{ item.label }}</text>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { OrderAPI } from "dehaze-sdk-js";

const statsItems = ref([
  {
    key: "users",
    label: "用户数",
    value: "-",
    path: "/pages/system/user/index",
  },
  {
    key: "orders",
    label: "订单数",
    value: "-",
    path: "/pages/system/order/index",
  },
  {
    key: "algorithms",
    label: "算法数",
    value: "-",
    path: "/pages/system/algorithm/index",
  },
  {
    key: "feedbacks",
    label: "反馈数",
    value: "-",
    path: "/pages/system/feedback/index",
  },
]);

const shortcuts = ref([
  { label: "用户管理", icon: "account", path: "/pages/system/user/index" },
  { label: "角色管理", icon: "lock", path: "/pages/system/role/index" },
  { label: "菜单管理", icon: "list", path: "/pages/system/menu/index" },
  { label: "部门管理", icon: "grid", path: "/pages/system/dept/index" },
  { label: "字典管理", icon: "file-text", path: "/pages/system/dict/index" },
  { label: "算法管理", icon: "setting", path: "/pages/system/algorithm/index" },
  { label: "数据集管理", icon: "folder", path: "/pages/system/dataset/index" },
  { label: "任务管理", icon: "hourglass", path: "/pages/system/task/index" },
  { label: "会员管理", icon: "man", path: "/pages/system/member/index" },
  { label: "套餐管理", icon: "rmb", path: "/pages/system/package/index" },
  { label: "订单管理", icon: "order", path: "/pages/system/order/index" },
  { label: "反馈管理", icon: "chat", path: "/pages/system/feedback/index" },
  { label: "消息管理", icon: "bell", path: "/pages/system/message/index" },
  { label: "推荐管理", icon: "star", path: "/pages/system/recommend/index" },
]);

const goPage = (path: string) => {
  uni.navigateTo({ url: path });
};

const fetchStats = async () => {
  try {
    const d = await OrderAPI.getStats();
    if (d) {
      const s = statsItems.value;
      const orderItem = s.find((i) => i.key === "orders");
      if (orderItem && d.totalOrders != null)
        orderItem.value = String(d.totalOrders);
    }
  } catch {}
};

fetchStats();
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.section-title {
  font-size: 30rpx;
  font-weight: bold;
  padding: 20rpx 0;
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16rpx;
}
.stat-card {
  background: $color-white;
  border-radius: 16rpx;
  padding: 30rpx 20rpx;
  text-align: center;
}
.stat-val {
  font-size: 40rpx;
  font-weight: bold;
  color: $color-primary;
  display: block;
}
.stat-label {
  font-size: 24rpx;
  color: $color-text-secondary;
  margin-top: 8rpx;
  display: block;
}
.shortcut-section {
  margin-top: 30rpx;
}
.shortcut-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16rpx;
}
.shortcut-item {
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10rpx;
}
.shortcut-label {
  font-size: 22rpx;
  color: $color-text-primary;
}
</style>
