<template>
  <PageLayout level="L1" title="我的" class="page">
    <view class="main-content">
      <!-- 用户卡 -->
      <view class="user-card">
        <view class="user-info">
          <image
            v-if="auth.userInfo?.avatar"
            :src="auth.userInfo.avatar"
            class="avatar-img"
          />
          <view v-else class="avatar-placeholder">
            <text class="avatar-text">{{ initial }}</text>
          </view>
          <view class="user-text">
            <view class="user-name-row">
              <text class="nickname">{{ auth.nickname || auth.username }}</text>
              <view v-if="auth.roles.length > 0" class="role-tags">
                <text v-for="role in auth.roles" :key="role" class="role-tag">{{
                  formatRole(role)
                }}</text>
              </view>
            </view>
            <text class="username">@{{ auth.username }}</text>
          </view>
        </view>
      </view>

      <!-- VIP 横幅 -->
      <view class="vip-banner" @click="handleMember">
        <view class="vip-info">
          <text class="vip-title" v-if="isVip">
            {{ memberProfile.levelName }} · 成长值
            {{ memberProfile.growthValue }}
          </text>
          <text class="vip-title" v-else>开通 VIP 畅享更多次数</text>
          <text class="vip-subtitle" v-if="isVip">
            本月已用 {{ memberProfile.monthlyDehazeUsed }}/{{
              memberProfile.monthlyDehazeQuota
            }}
            次
          </text>
          <text class="vip-subtitle" v-else>解锁全部高级功能</text>
        </view>
        <view class="vip-action">
          <text>{{ isVip ? "查看权益" : "去开通" }}</text>
          <SvgIcon name="arrow-right" size="14" color="#fff" />
        </view>
      </view>

      <!-- 数据统计 -->
      <view class="stats-row">
        <view class="stat-item" @click="handleQuota">
          <text class="stat-value">{{ quota.remaining }}</text>
          <text class="stat-label">剩余额度</text>
        </view>
        <view class="stat-item" @click="handleTaskHistory">
          <text class="stat-value">{{ quota.used }}</text>
          <text class="stat-label">本月处理</text>
        </view>
        <view class="stat-item" @click="handleFavorite">
          <text class="stat-value">{{ favCount }}</text>
          <text class="stat-label">我的收藏</text>
        </view>
      </view>

      <!-- 分组入口：个人数据 -->
      <view class="menu-group">
        <text class="group-title">个人数据</text>
        <view class="menu-card">
          <MenuItem
            icon="folder"
            icon-color="#10b981"
            label="我的文件"
            @click="goPersonal('files')"
          />
          <MenuItem
            icon="database"
            icon-color="#6366f1"
            label="我的数据集"
            @click="goPage('/pages/dataset/index')"
          />
          <MenuItem
            icon="clock"
            icon-color="#6366f1"
            label="处理历史"
            @click="goPage('/pages/task-history/index')"
          />
          <MenuItem
            icon="star"
            icon-color="#f59e0b"
            label="我的收藏"
            @click="goPersonal('favorites')"
          />
        </view>
      </view>

      <!-- 分组入口：商业服务 -->
      <view class="menu-group">
        <text class="group-title">商业服务</text>
        <view class="menu-card">
          <MenuItem
            icon="man-delete"
            icon-color="#3b82f6"
            label="我的会员"
            @click="goPersonal('member')"
          />
          <MenuItem
            icon="rmb-circle"
            icon-color="#f59e0b"
            label="我的套餐"
            @click="goPersonal('package')"
          />
          <MenuItem
            icon="order"
            icon-color="#10b981"
            label="我的订单"
            @click="goPersonal('orders')"
          />
          <MenuItem
            icon="calendar"
            icon-color="#8b5cf6"
            label="我的额度"
            @click="goPersonal('quota')"
          />
          <MenuItem
            icon="edit-pen"
            icon-color="#ec4899"
            label="反馈评价"
            @click="goPersonal('feedback')"
          />
        </view>
      </view>

      <!-- 分组入口：其他 -->
      <view class="menu-group">
        <text class="group-title">其他</text>
        <view class="menu-card">
          <MenuItem
            icon="setting"
            icon-color="#6b7280"
            label="系统设置"
            @click="goPersonal('settings')"
          />
          <MenuItem
            icon="question-circle"
            icon-color="#3b82f6"
            label="帮助中心"
            @click="goPersonal('help')"
          />
          <MenuItem
            icon="info-circle"
            icon-color="#9ca3af"
            label="关于我们"
            @click="goPersonal('about')"
          />
          <MenuItem
            icon="bell"
            icon-color="#f59e0b"
            label="消息设置"
            @click="goPage('/pages/notify/index')"
          />
        </view>
      </view>

      <!-- 管理入口（权限过滤） -->
      <view v-if="hasAnyAdminPerm" class="menu-group">
        <text class="group-title">管理入口</text>
        <view class="menu-card">
          <template v-if="hasPermGroup('algorithm') || hasPermGroup('dataset')">
            <text class="sub-group-title">算法与数据</text>
            <MenuItem
              v-if="auth.hasPerm('sys:algorithm:*')"
              icon="cpu"
              icon-color="#3b82f6"
              label="算法管理"
              @click="goPage('/pages/system/algorithm/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:dataset:*')"
              icon="database"
              icon-color="#6366f1"
              label="数据集管理"
              @click="goPage('/pages/system/dataset/index')"
            />
          </template>

          <template v-if="hasPermGroup('sys')">
            <text class="sub-group-title">系统管理</text>
            <MenuItem
              v-if="auth.hasPerm('sys:user:*')"
              icon="account"
              icon-color="#10b981"
              label="用户管理"
              @click="goPage('/pages/system/user/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:role:*')"
              icon="grid"
              icon-color="#f59e0b"
              label="角色管理"
              @click="goPage('/pages/system/role/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:menu:*')"
              icon="list-dot"
              icon-color="#8b5cf6"
              label="菜单管理"
              @click="goPage('/pages/system/menu/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:dept:*')"
              icon="home"
              icon-color="#ec4899"
              label="部门管理"
              @click="goPage('/pages/system/dept/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:dict:*')"
              icon="bookmark"
              icon-color="#14b8a6"
              label="字典管理"
              @click="goPage('/pages/system/dict/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:task:*')"
              icon="play-circle"
              icon-color="#f97316"
              label="任务管理"
              @click="goPage('/pages/system/task/index')"
            />
          </template>

          <template v-if="hasPermGroup('biz')">
            <text class="sub-group-title">运营管理</text>
            <MenuItem
              v-if="auth.hasPerm('sys:member:*')"
              icon="man-delete"
              icon-color="#3b82f6"
              label="会员管理"
              @click="goPage('/pages/system/member/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:package:*')"
              icon="rmb-circle"
              icon-color="#f59e0b"
              label="套餐管理"
              @click="goPage('/pages/system/package/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:order:*')"
              icon="order"
              icon-color="#10b981"
              label="订单管理"
              @click="goPage('/pages/system/order/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:feedback:*')"
              icon="edit-pen"
              icon-color="#ec4899"
              label="反馈评价管理"
              @click="goPage('/pages/system/feedback/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:recommendation:*')"
              icon="thumb-up"
              icon-color="#8b5cf6"
              label="推荐管理"
              @click="goPage('/pages/system/recommend/index')"
            />
            <MenuItem
              v-if="auth.hasPerm('sys:notify:*')"
              icon="bell"
              icon-color="#f97316"
              label="消息管理"
              @click="goPage('/pages/system/message/index')"
            />
          </template>
        </view>
      </view>

      <!-- 退出登录 -->
      <view class="logout-area">
        <view class="logout-btn" @click="handleLogout">
          <SvgIcon name="close-circle" size="22" color="#ef4444" />
          <text class="logout-text">退出登录</text>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import MenuItem from "./components/MenuItem.vue";
import { useAuthStore } from "@/store/auth";
import { LOGIN_PATH } from "@/routers/guard";
import { MemberAPI, ModelAPI, FavoriteAPI } from "dehaze-sdk-js";

const auth = useAuthStore();

const initial = computed(() =>
  (auth.nickname || auth.username || "U").charAt(0).toUpperCase()
);

// VIP/会员信息
const memberProfile = ref({
  levelCode: "level_0",
  levelName: "",
  growthValue: 0,
  monthlyDehazeQuota: 0,
  monthlyDehazeUsed: 0,
});

const isVip = computed(
  () =>
    memberProfile.value.levelCode !== "level_0" &&
    memberProfile.value.levelCode !== "level_1"
);

// 配额
const quota = ref({ remaining: 0, used: 0, total: 0 });

// 收藏数
const favCount = ref(0);

onMounted(async () => {
  try {
    const profile = await MemberAPI.getProfile();
    memberProfile.value = profile;
  } catch {
    // 非会员或无接口
  }

  try {
    const q = await ModelAPI.getQuota();
    quota.value = q;
  } catch {
    // 无数据
  }

  try {
    const c = await FavoriteAPI.getCount();
    const total = c.reduce((sum, item) => sum + item.count, 0);
    favCount.value = total;
  } catch {
    // 无数据
  }
});

function formatRole(role: string): string {
  return role.replace("ROLE_", "");
}

function goPage(url: string) {
  uni.navigateTo({ url });
}

function goPersonal(page: string) {
  uni.navigateTo({ url: `/pages/personal/${page}/index` });
}

function handleMember() {
  uni.navigateTo({ url: "/pages/personal/member/index" });
}

function handleQuota() {
  uni.navigateTo({ url: "/pages/personal/quota/index" });
}

function handleTaskHistory() {
  uni.navigateTo({ url: "/pages/task-history/index" });
}

function handleFavorite() {
  uni.navigateTo({ url: "/pages/personal/favorites/index" });
}

// 管理权限分组判断
const hasAnyAdminPerm = computed(() => {
  const adminPerms = [
    "sys:algorithm:*",
    "sys:dataset:*",
    "sys:user:*",
    "sys:role:*",
    "sys:menu:*",
    "sys:dept:*",
    "sys:dict:*",
    "sys:task:*",
    "sys:member:*",
    "sys:package:*",
    "sys:order:*",
    "sys:feedback:*",
    "sys:recommendation:*",
    "sys:notify:*",
  ];
  return adminPerms.some((p) => auth.hasPerm(p));
});

function hasPermGroup(group: string) {
  switch (group) {
    case "algorithm":
      return auth.hasPerm("sys:algorithm:*") || auth.hasPerm("sys:dataset:*");
    case "sys":
      return [
        "sys:user:*",
        "sys:role:*",
        "sys:menu:*",
        "sys:dept:*",
        "sys:dict:*",
        "sys:task:*",
      ].some((p) => auth.hasPerm(p));
    case "biz":
      return [
        "sys:member:*",
        "sys:package:*",
        "sys:order:*",
        "sys:feedback:*",
        "sys:recommendation:*",
        "sys:notify:*",
      ].some((p) => auth.hasPerm(p));
    default:
      return false;
  }
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
  padding-bottom: calc(100rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(100rpx + env(safe-area-inset-bottom));
}

// 用户卡
.user-card {
  background: #fff;
  border-radius: $radius-xl;
  padding: $spacing-lg;
  margin-bottom: $spacing-md;
  box-shadow: $shadow-sm;
}
.user-info {
  display: flex;
  align-items: center;
  gap: 24rpx;
}
.avatar-img {
  width: 112rpx;
  height: 112rpx;
  border-radius: 50%;
  flex-shrink: 0;
}
.avatar-placeholder {
  width: 112rpx;
  height: 112rpx;
  border-radius: 50%;
  background: $color-bg-secondary;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.avatar-text {
  font-size: $font-xl;
  font-weight: 700;
  color: $color-text-secondary;
}
.user-text {
  flex: 1;
  min-width: 0;
}
.user-name-row {
  display: flex;
  align-items: center;
  gap: 12rpx;
  flex-wrap: wrap;
}
.nickname {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
}
.username {
  display: block;
  font-size: $font-sm;
  color: $color-text-placeholder;
  margin-top: 4rpx;
}
.role-tags {
  display: flex;
  gap: 8rpx;
  flex-wrap: wrap;
}
.role-tag {
  font-size: 20rpx;
  color: $color-secondary;
  background: #e0e7ff;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
  font-weight: 500;
}

// VIP 横幅
.vip-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: $gradient-primary;
  border-radius: $radius-lg;
  padding: 28rpx;
  margin-bottom: $spacing-md;
}
.vip-info {
  flex: 1;
}
.vip-title {
  font-size: $font-md;
  font-weight: 600;
  color: #fff;
  display: block;
}
.vip-subtitle {
  font-size: $font-xs;
  color: rgba(255, 255, 255, 0.8);
  margin-top: 4rpx;
  display: block;
}
.vip-action {
  display: flex;
  align-items: center;
  gap: 4rpx;
  background: rgba(255, 255, 255, 0.2);
  padding: 12rpx 24rpx;
  border-radius: 12rpx;
  font-size: $font-sm;
  color: #fff;
  flex-shrink: 0;
}

// 统计
.stats-row {
  display: flex;
  background: #fff;
  border-radius: $radius-xl;
  margin-bottom: $spacing-md;
  box-shadow: $shadow-sm;
}
.stat-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28rpx 16rpx;
  &:active {
    background: #f9fafb;
  }
  & + & {
    border-left: 1rpx solid $color-border-light;
  }
}
.stat-value {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
}
.stat-label {
  font-size: $font-xs;
  color: $color-text-placeholder;
  margin-top: 4rpx;
}

// 菜单分组
.menu-group {
  margin-bottom: $spacing-md;
}
.group-title {
  display: block;
  font-size: $font-xs;
  font-weight: 500;
  color: $color-text-placeholder;
  text-transform: uppercase;
  padding: 0 4rpx 12rpx;
}
.menu-card {
  background: #fff;
  border-radius: $radius-xl;
  overflow: hidden;
  box-shadow: $shadow-sm;
}
.sub-group-title {
  display: block;
  font-size: $font-xs;
  color: $color-text-placeholder;
  padding: 20rpx 28rpx 8rpx;
}

// 退出
.logout-area {
  margin-top: 48rpx;
  padding: $spacing-md 0;
  display: flex;
  justify-content: center;
}
.logout-btn {
  display: flex;
  align-items: center;
  gap: 12rpx;
  padding: 20rpx 48rpx;
  background: #fff;
  border-radius: 16rpx;
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
