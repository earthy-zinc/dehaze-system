<template>
  <view class="app-navbar">
    <!-- 状态栏占位 -->
    <view class="status-bar" :style="{ height: statusBarHeight + 'px' }" />

    <!-- 导航栏内容 -->
    <view class="navbar-content">
      <!-- L2：返回 -->
      <view v-if="level === 'L2'" class="navbar-back" @click="goBack">
        <u-icon name="arrow-left" size="22" color="#374151" />
      </view>

      <!-- L1：品牌标识（点击回首页）+ Tab 标题 -->
      <view v-else class="navbar-brand" @click="goHome">
        <view class="logo-wrapper">
          <u-icon name="photo-fill" color="#fff" size="16" />
        </view>
        <text class="app-title">{{ title }}</text>
      </view>

      <!-- L2 居中页面标题 -->
      <text v-if="level === 'L2' && title" class="navbar-title">{{
        title
      }}</text>

      <!-- 右侧操作区：L1 显示全局搜索入口 -->
      <view class="navbar-actions">
        <view v-if="level === 'L1'" class="action-btn" @click="handleSearch">
          <u-icon name="search" size="22" color="#374151" />
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import { HOME_PATH } from "@/routers/guard";

interface Props {
  /** 导航形态：L1 Tab 根页（品牌+标题+搜索）/ L2 二级功能页（返回+标题） */
  level: "L1" | "L2";
  /** 页面标题：L1 为 Tab 标题，L2 为页面功能名 */
  title: string;
}

withDefaults(defineProps<Props>(), {
  level: "L1",
  title: "",
});

/** 状态栏高度 */
const statusBarHeight = ref(0);

onMounted(() => {
  // 获取状态栏高度
  try {
    const sysInfo = uni.getSystemInfoSync();
    statusBarHeight.value = sysInfo.statusBarHeight || 0;
  } catch {
    // 获取失败时保持默认值 0
  }
});

/** 返回上一页；无历史则回首页 */
const goBack = () => {
  const pages = getCurrentPages();
  if (pages.length > 1) {
    uni.navigateBack();
  } else {
    uni.switchTab({ url: HOME_PATH });
  }
};

/** 跳转首页 */
const goHome = () => {
  uni.switchTab({ url: HOME_PATH });
};

/** 搜索按钮点击 - 跳转到算法选择页（该页面已实现算法搜索） */
const handleSearch = () => {
  uni.navigateTo({ url: "/pages/algorithm-select/index" });
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

.navbar-back {
  width: 72rpx;
  height: 72rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: -16rpx;
  border-radius: 16rpx;

  &:active {
    background: #f3f4f6;
  }
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

/* L2 居中标题 */
.navbar-title {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  font-size: 32rpx;
  font-weight: 600;
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
