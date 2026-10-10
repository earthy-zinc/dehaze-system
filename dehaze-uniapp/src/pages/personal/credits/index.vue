<template>
  <PageLayout level="L2" title="积分余额" class="page">
    <view class="main-content">
      <view class="credits-card">
        <view class="credits-header">
          <text class="credits-title">去雾处理积分</text>
        </view>
        <view class="credits-progress">
          <view class="progress-bar">
            <view class="progress-fill" :style="{ width: usedPercent + '%' }" />
          </view>
        </view>
        <view class="credits-numbers">
          <view class="credits-item">
            <text class="credits-num used">{{ balance.monthlyUsed }}</text>
            <text class="credits-label">本月已用</text>
          </view>
          <view class="credits-item">
            <text class="credits-num remaining">{{
              balance.creditsBalance
            }}</text>
            <text class="credits-label">剩余积分</text>
          </view>
          <view class="credits-item">
            <text class="credits-num total">{{ balance.monthlyLimit }}</text>
            <text class="credits-label">本月限额</text>
          </view>
        </view>
      </view>

      <view class="tip-card">
        <SvgIcon name="info-circle" size="20" color="#3b82f6" />
        <text class="tip-text">
          图像去雾按张扣减积分，处理失败或取消不计费；积分不足时无法提交新任务，开通
          VIP 每月获赠积分
        </text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { AiBillingAPI } from "dehaze-sdk-js";

const balance = ref<{
  creditsBalance: string;
  monthlyUsed: number;
  monthlyLimit: number;
}>({ creditsBalance: "0", monthlyUsed: 0, monthlyLimit: 0 });

/** 本月限额为 0 表示未设限，进度条不展示占用 */
const usedPercent = computed(() =>
  balance.value.monthlyLimit > 0
    ? (balance.value.monthlyUsed / balance.value.monthlyLimit) * 100
    : 0
);

onMounted(async () => {
  try {
    balance.value = await AiBillingAPI.getBalance();
  } catch {
    uni.showToast({ title: "获取积分余额失败", icon: "none" });
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

.credits-card {
  padding: 32rpx;
  margin-bottom: $spacing-md;
  background: #fff;
  border-radius: $radius-xl;
  box-shadow: $shadow-sm;

  .credits-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: $spacing-md;
  }

  .credits-title {
    font-size: 30rpx;
    font-weight: 600;
    color: $color-text-primary;
  }
}

.credits-progress {
  margin-bottom: $spacing-md;

  .progress-bar {
    width: 100%;
    height: 16rpx;
    overflow: hidden;
    background: $color-bg-secondary;
    border-radius: 999rpx;
  }

  .progress-fill {
    height: 100%;
    background: $color-primary;
    border-radius: 999rpx;
    transition: width 0.3s;
  }
}

.credits-numbers {
  display: flex;
  align-items: center;
  justify-content: space-around;

  .credits-item {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .credits-num {
    font-size: 40rpx;
    font-weight: 600;

    &.used {
      color: $color-text-secondary;
    }

    &.remaining {
      color: $color-primary;
    }

    &.total {
      color: $color-text-primary;
    }
  }

  .credits-label {
    margin-top: 8rpx;
    font-size: 24rpx;
    color: $color-text-secondary;
  }
}

.tip-card {
  display: flex;
  align-items: flex-start;
  padding: $spacing-md;
  background: #fff;
  border-radius: $radius-xl;
  box-shadow: $shadow-sm;

  .tip-text {
    flex: 1;
    margin-left: 16rpx;
    font-size: 26rpx;
    line-height: 1.6;
    color: $color-text-secondary;
  }
}
</style>
