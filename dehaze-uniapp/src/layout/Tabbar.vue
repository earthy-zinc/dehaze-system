<template>
  <view class="app-tabbar">
    <view
      v-for="(item, index) in tabBarItems"
      :key="item.route"
      class="tabbar-item"
      :class="{ active: currentIndex === index }"
      @click="switchTab(item, index)"
    >
      <u-icon
        :name="
          currentIndex === index ? item.activeIcon || item.icon : item.icon
        "
        :size="22"
        :color="currentIndex === index ? activeColor : inactiveColor"
      />
      <text
        class="tabbar-label"
        :style="{ color: currentIndex === index ? activeColor : inactiveColor }"
      >
        {{ item.title }}
      </text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, watch, onMounted } from "vue";
import { tabBarItems } from "@/config/menu";
import type { MenuItem } from "@/config/menu";

interface Props {
  /** 当前选中索引 */
  current?: number;
  /** 当前路由（用于自动计算索引） */
  currentRoute?: string;
  /** 激活颜色 */
  activeColor?: string;
  /** 未激活颜色 */
  inactiveColor?: string;
}

const props = withDefaults(defineProps<Props>(), {
  current: 0,
  currentRoute: "/pages/home/index",
  activeColor: "#3b82f6",
  inactiveColor: "#9ca3af",
});

const currentIndex = ref(props.current);

// 监听 current 属性变化
watch(
  () => props.current,
  (val) => {
    currentIndex.value = val;
  }
);

// 监听 currentRoute 变化，自动更新索引（非 TabBar 页面无高亮）
watch(
  () => props.currentRoute,
  (route) => {
    if (route) {
      currentIndex.value = tabBarItems.findIndex(
        (item) => item.route === route
      );
    }
  },
  { immediate: true }
);

/** 切换 Tab */
const switchTab = (item: MenuItem, index: number) => {
  if (currentIndex.value === index) return;

  currentIndex.value = index;

  // tabBar 页面使用 reLaunch（清空页面栈，模拟 Tab 切换语义），其他页面使用 navigateTo
  if (tabBarItems.some((tab) => tab.route === item.route)) {
    uni.reLaunch({ url: item.route });
  } else {
    uni.navigateTo({
      url: item.route,
      fail: () => {
        uni.showToast({ title: "页面开发中", icon: "none" });
      },
    });
  }
};

// 初始化时根据当前页面更新索引
onMounted(() => {
  const pages = getCurrentPages();
  if (pages.length > 0) {
    const currentPage = pages[pages.length - 1];
    const route = "/" + currentPage?.route;
    const index = tabBarItems.findIndex((item) => item.route === route);
    if (index !== -1) {
      currentIndex.value = index;
    }
  }
});
</script>

<style lang="scss" scoped>
.app-tabbar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-around;
  height: 100rpx;
  background: #ffffff;
  box-shadow: 0 -4rpx 16rpx rgba(0, 0, 0, 0.06);
  border-top: 1rpx solid #f3f4f6;
  // 适配底部安全区
  padding-bottom: constant(safe-area-inset-bottom);
  padding-bottom: env(safe-area-inset-bottom);
}

.tabbar-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6rpx;
  padding: 12rpx 0;
  transition: all 0.2s;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    .tabbar-label {
      font-weight: 600;
    }
  }
}

.tabbar-label {
  font-size: 22rpx;
  transition: color 0.2s;
}
</style>
