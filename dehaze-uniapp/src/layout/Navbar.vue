<template>
  <view class="app-navbar">
    <!-- 状态栏占位 -->
    <view class="status-bar" :style="{ height: statusBarHeight + 'px' }" />

    <!-- 导航栏内容 -->
    <view class="navbar-content">
      <!-- Logo + 标题 -->
      <view class="navbar-brand" @click="goHome">
        <view class="logo-wrapper">
          <u-icon name="play-circle-fill" color="#fff" size="18" />
        </view>
        <text class="app-title">图像去雾系统</text>
      </view>

      <!-- 右侧操作区 -->
      <view class="navbar-actions">
        <view class="action-btn" @click="handleSearch">
          <u-icon name="search" size="22" color="#374151" />
        </view>
        <view class="action-btn menu-btn" @click="toggleMenu">
          <u-icon name="list" size="22" color="#374151" />
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";

interface Emits {
  (e: "toggle-menu"): void;
}

const emit = defineEmits<Emits>();

/** 状态栏高度 */
const statusBarHeight = ref(0);

onMounted(() => {
  // 获取状态栏高度
  try {
    const sysInfo = uni.getSystemInfoSync();
    statusBarHeight.value = sysInfo.statusBarHeight || 0;
  } catch (error) {
    console.warn("[AppNavbar] Failed to get statusBarHeight:", error);
  }
});

/** 跳转首页 */
const goHome = () => {
  uni.reLaunch({ url: "/pages/home/index" });
};

/** 搜索按钮点击 - 跳转到算法选择页（该页面已实现算法搜索） */
const handleSearch = () => {
  uni.navigateTo({ url: "/pages/algorithm-select/index" });
};

/** 切换菜单 */
const toggleMenu = () => {
  emit("toggle-menu");
};
</script>

<style lang="scss" scoped>
.app-navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: #ffffff;
  box-shadow: 0 2rpx 16rpx rgba(0, 0, 0, 0.08);
}

.status-bar {
  width: 100%;
  background: #ffffff;
}

.navbar-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16rpx 32rpx;
  height: 88rpx;
}

.navbar-brand {
  display: flex;
  align-items: center;
  gap: 16rpx;
  cursor: pointer;

  &:active {
    opacity: 0.8;
  }
}

.logo-wrapper {
  width: 64rpx;
  height: 64rpx;
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
  border-radius: 16rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4rpx 12rpx rgba(59, 130, 246, 0.3);
}

.app-title {
  font-size: 32rpx;
  font-weight: 700;
  color: #1f2937;
}

.navbar-actions {
  display: flex;
  align-items: center;
  gap: 8rpx;
}

.action-btn {
  width: 72rpx;
  height: 72rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 16rpx;
  transition: background 0.2s;

  &:active {
    background: #f3f4f6;
  }
}
</style>
