<template>
  <view class="immersive-layout">
    <!-- 顶部深色半透明导航栏 -->
    <view class="immersive-navbar">
      <!-- 状态栏占位 -->
      <view class="status-bar" :style="{ height: statusBarHeight + 'px' }" />

      <view class="navbar-content">
        <!-- 返回按钮 -->
        <view class="navbar-back" @click="goBack">
          <SvgIcon name="arrow-left" size="20" color="#fff" />
        </view>

        <!-- 居中标题 -->
        <text class="navbar-title">{{ title }}</text>

        <!-- 右侧操作区 -->
        <view v-if="$slots.actions" class="navbar-actions">
          <slot name="actions" />
        </view>
      </view>
    </view>

    <!-- 全屏内容区 -->
    <view class="immersive-body">
      <slot />
    </view>

    <!-- 底部工具栏 -->
    <view v-if="$slots.toolbar" class="immersive-toolbar">
      <slot name="toolbar" />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  /** 导航栏居中标题 */
  title?: string;
}

withDefaults(defineProps<Props>(), {
  title: "",
});

/** 状态栏高度 */
const statusBarHeight = ref(0);

onMounted(() => {
  try {
    const sysInfo = uni.getSystemInfoSync();
    statusBarHeight.value = sysInfo.statusBarHeight || 0;
  } catch {
    // 获取失败时保持默认值 0
  }
});

/** 返回上一页 */
const goBack = () => {
  uni.navigateBack();
};
</script>

<style lang="scss" scoped>
.immersive-layout {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100vh;
  background: #000;
}

/* 顶部导航栏 */
.immersive-navbar {
  position: sticky;
  top: 0;
  z-index: 100;
}

.status-bar {
  width: 100%;
  background: rgba(0, 0, 0, 0.7);
}

.navbar-content {
  display: flex;
  align-items: center;
  height: 88rpx;
  padding: 0 32rpx;
  background: linear-gradient(to bottom, rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0));
}

.navbar-back {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 56rpx;
  height: 56rpx;
  border-radius: 50%;
  margin-left: -12rpx;

  &:active {
    background: rgba(255, 255, 255, 0.15);
  }
}

.navbar-title {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  font-size: 32rpx;
  font-weight: 600;
  color: #fff;
}

.navbar-actions {
  display: flex;
  align-items: center;
  gap: 8rpx;
  margin-left: auto;
}

/* 全屏内容区 */
.immersive-body {
  flex: 1;
  overflow: hidden;
}

/* 底部工具栏 */
.immersive-toolbar {
  flex-shrink: 0;
  background: rgba(0, 0, 0, 0.85);
}
</style>
