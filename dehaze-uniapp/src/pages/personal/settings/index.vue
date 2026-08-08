<template>
  <PageLayout level="L2" title="系统设置" class="page">
    <view class="main-content">
      <view class="setting-group">
        <text class="group-title">通用</text>
        <view class="setting-card">
          <view class="setting-item" @click="handleCache">
            <text class="setting-label">清理缓存</text>
            <text class="setting-value">{{ cacheSize }}</text>
          </view>
        </view>
      </view>

      <view class="setting-group">
        <text class="group-title">通知</text>
        <view class="setting-card">
          <view class="setting-item" @click="goNotify">
            <text class="setting-label">消息设置</text>
            <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
          </view>
        </view>
      </view>

      <view class="setting-group">
        <text class="group-title">关于</text>
        <view class="setting-card">
          <view class="setting-item">
            <text class="setting-label">版本号</text>
            <text class="setting-value">1.0.0</text>
          </view>
        </view>
      </view>

      <view class="logout-btn" @click="handleLogout">
        <text class="logout-text">退出登录</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { useAuthStore } from "@/store/auth";
import { LOGIN_PATH } from "@/routers/guard";

const auth = useAuthStore();
const cacheSize = ref("0 KB");

function handleCache() {
  uni.showModal({
    title: "清理缓存",
    content: "确认清理应用缓存？",
    confirmColor: "#3b82f6",
    success: (res) => {
      if (res.confirm) {
        cacheSize.value = "0 KB";
        uni.showToast({ title: "缓存已清理", icon: "success" });
      }
    },
  });
}

function goNotify() {
  uni.navigateTo({ url: "/pages/notify/index" });
}

async function handleLogout() {
  uni.showModal({
    title: "确认退出",
    content: "退出登录后需要重新登录",
    confirmColor: "#ef4444",
    success: async (res) => {
      if (res.confirm) {
        try {
          await auth.logout();
          uni.showToast({ title: "已退出", icon: "success" });
          setTimeout(() => uni.reLaunch({ url: LOGIN_PATH }), 800);
        } catch {
          uni.reLaunch({ url: LOGIN_PATH });
        }
      }
    },
  });
}
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
}

.setting-group {
  margin-bottom: $spacing-md;
}
.group-title {
  display: block;
  font-size: $font-xs;
  font-weight: 500;
  color: $color-text-placeholder;
  padding: 0 4rpx 12rpx;
}
.setting-card {
  background: #fff;
  border-radius: $radius-xl;
  overflow: hidden;
  box-shadow: $shadow-sm;
}
.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 28rpx;
  &:active {
    background: #f9fafb;
  }
  & + & {
    border-top: 1rpx solid $color-border-light;
  }
}
.setting-label {
  font-size: $font-md;
  color: $color-text-primary;
}
.setting-value {
  font-size: $font-sm;
  color: $color-text-placeholder;
}

.logout-btn {
  margin-top: 48rpx;
  text-align: center;
  background: #fff;
  padding: 28rpx;
  border-radius: $radius-xl;
  box-shadow: $shadow-sm;
  &:active {
    background: #f9fafb;
  }
}
.logout-text {
  font-size: $font-md;
  color: $color-danger;
}
</style>
