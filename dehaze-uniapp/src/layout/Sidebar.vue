<template>
  <!-- 遮罩层 -->
  <view
    v-if="visible"
    class="sidebar-overlay"
    :class="{ active: visible }"
    @click="closeSidebar"
  />

  <!-- 侧边栏面板 -->
  <view class="sidebar-panel" :class="{ active: visible }">
    <!-- 头部 -->
    <view class="sidebar-header">
      <view class="header-content">
        <view class="logo-wrapper">
          <u-icon name="play-circle-fill" color="#fff" size="24" />
        </view>
        <view class="header-text">
          <text class="app-name">图像去雾系统</text>
          <text class="app-desc">功能菜单</text>
        </view>
      </view>
      <view class="close-btn" @click="closeSidebar">
        <u-icon name="close" size="20" color="#fff" />
      </view>
    </view>

    <!-- 菜单内容 -->
    <scroll-view class="sidebar-content" scroll-y>
      <!-- 用户信息区 -->
      <view v-if="isLoggedIn" class="user-info-card" @click="goUserCenter">
        <view class="user-avatar">
          <text class="avatar-letter">{{ userInitial }}</text>
        </view>
        <view class="user-detail">
          <text class="user-name">{{ authStore.nickname || authStore.username }}</text>
          <text class="user-role">{{ userRoleText }}</text>
        </view>
        <u-icon name="arrow-right" size="16" color="#d1d5db" />
      </view>

      <!-- 未登录提示 -->
      <view v-else class="login-prompt" @click="goLogin">
        <u-icon name="account" size="20" color="#6b7280" />
        <text class="login-text">点击登录</text>
        <u-icon name="arrow-right" size="16" color="#d1d5db" />
      </view>

      <view class="menu-divider" />

      <!-- 首页 -->
      <view
        class="menu-item"
        :class="{ active: isActive(homeItem.route) }"
        @click="navigateTo(homeItem.route)"
      >
        <u-icon
          :name="
            isActive(homeItem.route)
              ? homeItem.activeIcon || homeItem.icon
              : homeItem.icon
          "
          size="20"
          :color="isActive(homeItem.route) ? '#3b82f6' : '#6b7280'"
        />
        <text class="menu-title">{{ homeItem.title }}</text>
      </view>

      <view class="menu-divider" />

      <!-- 分组菜单 -->
      <view
        v-for="section in menuSections"
        :key="section.title"
        class="menu-section"
      >
        <view class="section-header">
          <u-icon
            v-if="section.icon"
            :name="section.icon"
            size="14"
            color="#9ca3af"
          />
          <text class="section-title">{{ section.title }}</text>
        </view>
        <view
          v-for="item in section.items"
          :key="item.route"
          class="menu-item"
          :class="{ active: isActive(item.route) }"
          @click="navigateTo(item.route)"
        >
          <u-icon
            :name="
              isActive(item.route) ? item.activeIcon || item.icon : item.icon
            "
            size="20"
            :color="isActive(item.route) ? '#3b82f6' : '#6b7280'"
          />
          <text class="menu-title">{{ item.title }}</text>
          <view v-if="item.badge" class="menu-badge">{{ item.badge }}</view>
          <view v-if="item.isNew" class="menu-new">NEW</view>
        </view>
      </view>
    </scroll-view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import { useAuthStore } from "@/store/auth";
import { homeItem, menuSections, isTabBarPage } from "@/config/menu";

const authStore = useAuthStore();

const isLoggedIn = computed(() => authStore.isLoggedIn);
const userInitial = computed(() => (authStore.nickname || authStore.username || "U").charAt(0).toUpperCase());
const userRoleText = computed(() => {
  const roles = authStore.roles;
  if (roles.length === 0) return "未登录";
  return roles.map((r) => r.replace("ROLE_", "")).join(" | ") || "普通用户";
});

interface Props {
  /** 侧边栏是否可见 */
  visible: boolean;
  /** 当前路由 */
  currentRoute?: string;
}

interface Emits {
  (e: "close"): void;
  (e: "navigate", route: string): void;
}

const props = withDefaults(defineProps<Props>(), {
  currentRoute: "/pages/home/index",
});

const emit = defineEmits<Emits>();

/** 判断是否为当前激活路由 */
const isActive = (route: string) => props.currentRoute === route;

/** 关闭侧边栏 */
const closeSidebar = () => {
  emit("close");
};

/** 导航到指定路由 */
const navigateTo = (route: string) => {
  emit("navigate", route);
  closeSidebar();

  // 判断是否为 tabBar 页面
  if (isTabBarPage(route)) {
    uni.switchTab({ url: route });
  } else {
    uni.navigateTo({
      url: route,
      fail: () => {
        uni.showToast({ title: "页面开发中", icon: "none" });
      },
    });
  }
};

/** 跳转用户中心 */
const goUserCenter = () => {
  navigateTo("/pages/user-center/index");
};

/** 跳转登录 */
const goLogin = () => {
  uni.reLaunch({ url: "/pages/login/index" });
};
</script>

<style lang="scss" scoped>
.sidebar-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 998;
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s ease;

  &.active {
    opacity: 1;
    visibility: visible;
  }
}

.sidebar-panel {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 560rpx;
  max-width: 85vw;
  background: #ffffff;
  z-index: 999;
  transform: translateX(100%);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: -8rpx 0 32rpx rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;

  &.active {
    transform: translateX(0);
  }
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 48rpx 32rpx 32rpx;
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
  // 适配状态栏
  padding-top: calc(48rpx + constant(safe-area-inset-top));
  padding-top: calc(48rpx + env(safe-area-inset-top));
}

.header-content {
  display: flex;
  align-items: center;
  gap: 20rpx;
}

.logo-wrapper {
  width: 80rpx;
  height: 80rpx;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.header-text {
  display: flex;
  flex-direction: column;
  gap: 4rpx;
}

.app-name {
  font-size: 36rpx;
  font-weight: 700;
  color: #ffffff;
}

.app-desc {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.8);
}

.close-btn {
  width: 64rpx;
  height: 64rpx;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 16rpx;
  display: flex;
  align-items: center;
  justify-content: center;

  &:active {
    background: rgba(255, 255, 255, 0.3);
  }
}

.sidebar-content {
  flex: 1;
  padding: 24rpx 0;
  // 适配底部安全区
  padding-bottom: calc(24rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(24rpx + env(safe-area-inset-bottom));
}

/* 用户信息卡片 */
.user-info-card {
  display: flex;
  align-items: center;
  gap: 20rpx;
  margin: 0 16rpx 8rpx;
  padding: 24rpx;
  background: linear-gradient(135deg, #eff6ff, #dbeafe);
  border-radius: 20rpx;
  &:active { opacity: 0.8; }
}

.login-prompt {
  display: flex;
  align-items: center;
  gap: 16rpx;
  margin: 0 16rpx 8rpx;
  padding: 24rpx;
  background: #f3f4f6;
  border-radius: 20rpx;
  &:active { opacity: 0.8; }
}

.login-text {
  flex: 1;
  font-size: 28rpx;
  color: #6b7280;
}

.user-avatar {
  width: 72rpx; height: 72rpx; border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}

.avatar-letter { font-size: 32rpx; font-weight: 700; color: #fff; }

.user-detail { flex: 1; min-width: 0; }
.user-name { display: block; font-size: 28rpx; font-weight: 600; color: #1f2937; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.user-role { display: block; font-size: 22rpx; color: #3b82f6; margin-top: 4rpx; }

.menu-divider {
  height: 2rpx;
  background: #e5e7eb;
  margin: 16rpx 24rpx;
}

.menu-section {
  margin-bottom: 16rpx;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12rpx;
  padding: 16rpx 32rpx;
}

.section-title {
  font-size: 24rpx;
  font-weight: 600;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 1rpx;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 24rpx;
  padding: 24rpx 32rpx;
  margin: 4rpx 16rpx;
  border-radius: 20rpx;
  transition: all 0.2s;

  &:active {
    background: #f3f4f6;
  }

  &.active {
    background: rgba(59, 130, 246, 0.1);

    .menu-title {
      color: #3b82f6;
      font-weight: 600;
    }
  }
}

.menu-title {
  flex: 1;
  font-size: 30rpx;
  color: #374151;
}

.menu-badge {
  padding: 4rpx 16rpx;
  background: #3b82f6;
  border-radius: 20rpx;
  font-size: 22rpx;
  color: #ffffff;
  font-weight: 600;
}

.menu-new {
  padding: 4rpx 12rpx;
  background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
  border-radius: 8rpx;
  font-size: 18rpx;
  color: #ffffff;
  font-weight: 700;
}
</style>
