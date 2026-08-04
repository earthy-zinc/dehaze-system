<template>
  <view class="auth-shell">
    <!-- 品牌氛围背景：渐变 + 柔光 -->
    <view class="auth-atmosphere" aria-hidden="true">
      <view class="auth-orb auth-orb-1"></view>
      <view class="auth-orb auth-orb-2"></view>
      <view class="auth-orb auth-orb-3"></view>
    </view>

    <!-- 顶部关闭按钮（毛玻璃） -->
    <view class="auth-close-area">
      <view class="auth-glass-btn" @click="handleClose">
        <u-icon name="close" size="20" color="#ffffff" />
      </view>
    </view>

    <!-- 品牌区（浮于渐变之上） -->
    <view class="brand-area">
      <view class="auth-logo">
        <u-icon name="photo-fill" size="30" color="#ffffff" />
      </view>
      <text class="brand-title">{{ title }}</text>
      <text class="brand-subtitle">{{ subtitle }}</text>
    </view>

    <!-- 白色圆角表单面板 -->
    <view class="auth-sheet">
      <slot />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { HOME_PATH } from "@/routers/guard";

interface Props {
  /** 品牌区主标题 */
  title: string;
  /** 品牌区副标题 */
  subtitle: string;
}

defineProps<Props>();

const emit = defineEmits<{ (e: "close"): void }>();

/** 关闭：返回上一页；无历史则回首页 */
function handleClose() {
  emit("close");
  const pages = getCurrentPages();
  if (pages.length > 1) {
    uni.navigateBack();
  } else {
    uni.reLaunch({ url: HOME_PATH });
  }
}
</script>

<style lang="scss" scoped>
.auth-shell {
  position: relative;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: $color-bg-primary;
}

/* 氛围背景 */
.auth-atmosphere {
  position: fixed;
  inset: 0;
  overflow: hidden;
  background: $gradient-primary;
}

.auth-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(72rpx);
  opacity: 0.55;
  pointer-events: none;

  &-1 {
    width: 280rpx;
    height: 280rpx;
    top: -60rpx;
    right: -80rpx;
    background: rgba(147, 197, 253, 0.7);
  }

  &-2 {
    width: 240rpx;
    height: 240rpx;
    top: 120rpx;
    left: -90rpx;
    background: rgba(129, 140, 248, 0.6);
  }

  &-3 {
    width: 200rpx;
    height: 200rpx;
    top: 280rpx;
    right: 40rpx;
    background: rgba(96, 165, 250, 0.45);
  }
}

/* 顶部关闭 */
.auth-close-area {
  position: relative;
  z-index: 1;
  padding: 40rpx 48rpx 0;
}

.auth-glass-btn {
  width: 80rpx;
  height: 80rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.18);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.22);
  color: #ffffff;

  &:active {
    transform: scale(0.94);
    background: rgba(255, 255, 255, 0.28);
  }
}

/* 品牌区 */
.brand-area {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 48rpx;
  margin-bottom: 48rpx;
}

.auth-logo {
  width: 128rpx;
  height: 128rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 32rpx;
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: 0 8rpx 24rpx rgba(30, 58, 138, 0.3);
  margin-bottom: 24rpx;
}

.brand-title {
  font-size: 44rpx;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 2rpx;
  margin-bottom: 12rpx;
}

.brand-subtitle {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.82);
  letter-spacing: 1rpx;
}

/* 白色圆角面板：内容自适应，不延伸到底部，底部露出渐变背景 */
.auth-sheet {
  position: relative;
  z-index: 1;
  background: $color-white;
  border-radius: 32rpx;
  box-shadow: 0 -8rpx 32rpx rgba(0, 0, 0, 0.12);
  padding: 48rpx 48rpx 56rpx;
  margin: 32rpx 24rpx 56rpx;
}
</style>
