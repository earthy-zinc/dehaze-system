<template>
  <PageLayout level="L2" title="关于我们" class="page">
    <view class="main-content">
      <view class="about-header">
        <view class="logo-wrapper">
          <SvgIcon name="photo-fill" color="#fff" size="32" />
        </view>
        <text class="app-name">Dehaze 图像去雾系统</text>
        <text class="app-version">版本 1.0.0</text>
      </view>

      <view class="info-card">
        <text class="info-title">产品简介</text>
        <text class="info-text">
          Dehaze
          是一个基于深度学习的智能图像去雾平台，支持多种先进算法，提供图像上传、算法选择、参数调节、效果对比等完整处理流程，帮助用户快速获得高质量的去雾图像。
        </text>
      </view>

      <view class="info-card">
        <text class="info-title">法律信息</text>
        <view class="link-list">
          <view class="link-item" @click="openLink('privacy')">
            <text class="link-label">隐私政策</text>
            <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
          </view>
          <view class="link-item" @click="openLink('terms')">
            <text class="link-label">用户协议</text>
            <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
          </view>
        </view>
      </view>

      <text class="copyright">© 2024 Dehaze Team. All rights reserved.</text>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";

function openLink(type: string) {
  uni.showToast({
    title: `${type === "privacy" ? "隐私政策" : "用户协议"} 页面`,
    icon: "none",
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
  display: flex;
  flex-direction: column;
  align-items: center;
}

.about-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 64rpx 0 48rpx;
}
.logo-wrapper {
  width: 120rpx;
  height: 120rpx;
  background: $gradient-primary;
  border-radius: 28rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24rpx;
  box-shadow: 0 8rpx 32rpx rgba(59, 130, 246, 0.3);
}
.app-name {
  font-size: $font-xl;
  font-weight: 700;
  color: $color-text-primary;
  margin-bottom: 8rpx;
}
.app-version {
  font-size: $font-sm;
  color: $color-text-placeholder;
}

.info-card {
  width: 100%;
  background: #fff;
  border-radius: $radius-xl;
  padding: 28rpx;
  box-shadow: $shadow-sm;
  margin-bottom: $spacing-md;
}
.info-title {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  display: block;
  margin-bottom: 12rpx;
}
.info-text {
  font-size: $font-sm;
  color: $color-text-secondary;
  line-height: 1.8;
}

.link-list {
}
.link-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20rpx 0;
  &:active {
    opacity: 0.7;
  }
  & + & {
    border-top: 1rpx solid $color-border-light;
  }
}
.link-label {
  font-size: $font-md;
  color: $color-primary;
}

.copyright {
  font-size: $font-xs;
  color: $color-text-disabled;
  margin-top: 48rpx;
}
</style>
