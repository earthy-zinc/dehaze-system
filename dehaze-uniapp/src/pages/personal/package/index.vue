<template>
  <PageLayout level="L2" title="我的套餐" class="page">
    <view class="main-content">
      <view v-if="loading" class="loading-container">
        <view class="loading-spinner" />
        <text class="loading-text">加载中...</text>
      </view>

      <view v-else-if="packages.length > 0" class="package-list">
        <view v-for="pkg in packages" :key="pkg.id" class="package-card">
          <view class="pkg-header">
            <text class="pkg-name">{{ pkg.name }}</text>
            <text class="pkg-price"
              >¥{{ pkg.salePrice }}/{{ periodText(pkg.periodDays) }}</text
            >
          </view>
          <text class="pkg-desc" v-if="pkg.description">{{
            pkg.description
          }}</text>
          <view class="pkg-features" v-if="pkg.benefits">
            <text
              class="feature"
              v-for="(val, key) in pkg.benefits"
              :key="key"
              >{{ formatBenefit(key, val) }}</text
            >
          </view>
          <view class="pkg-action">
            <text class="buy-btn">立即购买</text>
          </view>
        </view>
      </view>

      <view v-else class="empty-state">
        <view class="empty-tip">暂无套餐</view>
        <text class="empty-hint">当前没有可购买的套餐</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { PackageAPI } from "dehaze-sdk-js";
import type { PackageDetailVO } from "dehaze-sdk-js";

const loading = ref(false);
const packages = ref<PackageDetailVO[]>([]);

function periodText(days?: number): string {
  if (!days) return "";
  if (days <= 1) return "天";
  if (days <= 31) return `${days}天`;
  if (days <= 365) return `${Math.round(days / 30)}月`;
  return `${Math.round(days / 365)}年`;
}

const BENEFIT_KEYS: Record<string, string> = {
  dehazeDaily: "每日去雾",
  dehazeMonthly: "每月去雾",
  evaluateMonthly: "每月评估",
  batchProcessing: "批量处理",
  originalImageDownload: "原图下载",
  prioritySupport: "优先客服",
};

function formatBenefit(key: string, val: number): string {
  const label = BENEFIT_KEYS[key] || key;
  return `${label} ×${val}`;
}

onMounted(async () => {
  loading.value = true;
  try {
    packages.value = await PackageAPI.listOnSale();
  } catch {
    uni.showToast({ title: "获取套餐失败", icon: "none" });
  } finally {
    loading.value = false;
  }
});
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

.package-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.package-card {
  background: $color-white;
  border-radius: $radius-xl;
  padding: 28rpx;
  box-shadow: $shadow-sm;
  position: relative;
  overflow: hidden;
}
.pkg-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12rpx;
}
.pkg-name {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
}
.pkg-price {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-warning;
}
.pkg-desc {
  font-size: $font-sm;
  color: $color-text-secondary;
  display: block;
  margin-bottom: 16rpx;
}
.pkg-features {
  display: flex;
  flex-wrap: wrap;
  gap: 8rpx;
  margin-bottom: 20rpx;
}
.feature {
  font-size: $font-xs;
  color: $color-primary;
  background: $color-primary-bg;
  padding: 6rpx 16rpx;
  border-radius: 8rpx;
}
.pkg-action {
  display: flex;
  justify-content: flex-end;
}
.buy-btn {
  font-size: $font-sm;
  font-weight: 600;
  color: $color-white;
  background: $gradient-primary;
  padding: 16rpx 40rpx;
  border-radius: 12rpx;
  &:active {
    opacity: 0.8;
  }
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.loading-text {
  margin-top: 24rpx;
  font-size: $font-md;
  color: $color-text-placeholder;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}
.empty-tip {
  font-size: $font-md;
}
.empty-hint {
  font-size: $font-sm;
  color: $color-text-placeholder;
  margin-top: 16rpx;
}
</style>
