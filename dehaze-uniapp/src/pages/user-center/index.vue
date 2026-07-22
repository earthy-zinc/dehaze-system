<template>
  <PageLayout class="page">
    <view class="main-content">
      <!-- 用户头像区 -->
      <view class="profile-card">
        <view class="avatar-wrapper">
          <image
            v-if="auth.userInfo?.avatar"
            :src="auth.userInfo.avatar"
            class="avatar-img"
          />
          <view v-else class="avatar-placeholder">
            <text class="avatar-text">{{ initial }}</text>
          </view>
        </view>
        <text class="nickname">{{ auth.nickname || auth.username }}</text>
        <text class="username">@{{ auth.username }}</text>
        <text class="join-text">用户ID: {{ auth.userId }}</text>
      </view>

      <!-- 角色信息 -->
      <view class="section-card">
        <text class="section-title">角色信息</text>
        <view class="role-list">
          <view v-for="role in auth.roles" :key="role" class="role-tag">
            {{ formatRole(role) }}
          </view>
          <text v-if="auth.roles.length === 0" class="empty-text"
            >暂无角色</text
          >
        </view>
      </view>

      <!-- 权限概览 -->
      <view class="section-card">
        <view class="section-header">
          <text class="section-title">权限概览</text>
          <text class="section-count">{{ auth.perms.length }} 项</text>
        </view>
        <view v-if="auth.perms.length > 0" class="perm-grid">
          <text
            v-for="perm in auth.perms.slice(0, 12)"
            :key="perm"
            class="perm-item"
          >
            {{ formatPerm(perm) }}
          </text>
        </view>
        <text v-else class="empty-text">暂无权限</text>
      </view>

      <!-- 操作区 -->
      <view class="actions">
        <view class="action-item" @click="handleTaskHistory">
          <u-icon name="clock" size="22" color="#3b82f6" />
          <text class="action-label">处理历史</text>
          <u-icon name="arrow-right" size="16" color="#d1d5db" />
        </view>
        <view class="action-item" @click="handleLogout">
          <u-icon name="close-circle" size="22" color="#ef4444" />
          <text class="action-label danger">退出登录</text>
          <u-icon name="arrow-right" size="16" color="#d1d5db" />
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import PageLayout from "@/layout/index.vue";
import { useAuthStore } from "@/store/auth";
import { navigateToLogin, navigateToHome } from "@/routers/guard";

const auth = useAuthStore();

const initial = computed(() =>
  (auth.nickname || auth.username || "U").charAt(0).toUpperCase()
);

function formatRole(role: string): string {
  return role.replace("ROLE_", "");
}

function formatPerm(perm: string): string {
  const parts = perm.split(":");
  return parts[parts.length - 1] || perm;
}

function handleTaskHistory() {
  uni.navigateTo({ url: "/pages/task-history/index" });
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
          setTimeout(() => navigateToLogin(), 800);
        } catch {
          navigateToLogin();
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
  background: #f9fafb;
}
.main-content {
  padding: 24rpx;
  padding-bottom: calc(80rpx + constant(safe-area-inset-bottom));
}

.profile-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  border-radius: 24rpx;
  padding: 48rpx 32rpx;
  margin-bottom: 24rpx;
}
.avatar-wrapper {
  margin-bottom: 20rpx;
}
.avatar-placeholder {
  width: 128rpx;
  height: 128rpx;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
  border: 4rpx solid rgba(255, 255, 255, 0.4);
}
.avatar-text {
  font-size: 48rpx;
  font-weight: 700;
  color: #fff;
}
.avatar-img {
  width: 128rpx;
  height: 128rpx;
  border-radius: 50%;
  border: 4rpx solid rgba(255, 255, 255, 0.4);
}
.nickname {
  font-size: 36rpx;
  font-weight: 700;
  color: #fff;
  margin-bottom: 8rpx;
}
.username {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.8);
  margin-bottom: 8rpx;
}
.join-text {
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.6);
}

.section-card {
  background: #fff;
  border-radius: 20rpx;
  padding: 28rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16rpx;
}
.section-title {
  font-size: 28rpx;
  font-weight: 600;
  color: #374151;
  display: block;
}
.section-count {
  font-size: 24rpx;
  color: #9ca3af;
}

.role-list {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
  margin-top: 16rpx;
}
.role-tag {
  font-size: 24rpx;
  color: #6366f1;
  background: #e0e7ff;
  padding: 8rpx 20rpx;
  border-radius: 16rpx;
  font-weight: 500;
}

.perm-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8rpx;
  margin-top: 16rpx;
}
.perm-item {
  font-size: 22rpx;
  color: #6b7280;
  background: #f3f4f6;
  padding: 6rpx 14rpx;
  border-radius: 10rpx;
}

.actions {
  background: #fff;
  border-radius: 20rpx;
  overflow: hidden;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);
}
.action-item {
  display: flex;
  align-items: center;
  gap: 20rpx;
  padding: 28rpx;
  &:active {
    background: #f9fafb;
  }
  & + & {
    border-top: 1rpx solid #f3f4f6;
  }
}
.action-label {
  flex: 1;
  font-size: 28rpx;
  color: #374151;
  &.danger {
    color: #ef4444;
  }
}
.empty-text {
  font-size: 24rpx;
  color: #9ca3af;
  margin-top: 12rpx;
  display: block;
}
</style>
