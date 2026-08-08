<template>
  <PageLayout level="L2" title="我的额度" class="page">
    <view class="main-content">
      <view class="quota-card">
        <view class="quota-header">
          <text class="quota-title">去雾处理额度</text>
          <text class="quota-reset" v-if="quota.resetDate"
            >每月 {{ quota.resetDate }} 重置</text
          >
        </view>
        <view class="quota-progress">
          <view class="progress-bar">
            <view
              class="progress-fill"
              :style="{
                width:
                  quota.total > 0
                    ? (quota.used / quota.total) * 100 + '%'
                    : '0%',
              }"
            />
          </view>
        </view>
        <view class="quota-numbers">
          <view class="quota-item">
            <text class="quota-num used">{{ quota.used }}</text>
            <text class="quota-label">已使用</text>
          </view>
          <view class="quota-item">
            <text class="quota-num remaining">{{ quota.remaining }}</text>
            <text class="quota-label">剩余</text>
          </view>
          <view class="quota-item">
            <text class="quota-num total">{{ quota.total }}</text>
            <text class="quota-label">总量</text>
          </view>
        </view>
      </view>

      <view class="tip-card">
        <SvgIcon name="info-circle" size="20" color="#3b82f6" />
        <text class="tip-text">额度按月重置，开通 VIP 可获得更多处理次数</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { ModelAPI } from "dehaze-sdk-js";

const quota = ref({ remaining: 0, used: 0, total: 0, resetDate: "" });

onMounted(async () => {
  try {
    quota.value = await ModelAPI.getQuota();
  } catch {
    uni.showToast({ title: "获取额度失败", icon: "none" });
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

.quota-card {
  background: #fff;
  border-radius: $radius-xl;
  padding: 32rpx;
  box-shadow: $shadow-sm;
  margin-bottom: $spacing-md;
}
.quota-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24rpx;
}
.quota-title {
  font-size: $font-lg;
  font-weight: 600;
  color: $color-text-primary;
}
.quota-reset {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.quota-progress {
  margin-bottom: 24rpx;
}
.progress-bar {
  width: 100%;
  height: 16rpx;
  background: $color-bg-secondary;
  border-radius: 8rpx;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: $gradient-primary;
  border-radius: 8rpx;
  transition: width 0.3s;
}

.quota-numbers {
  display: flex;
}
.quota-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.quota-num {
  font-size: $font-xl;
  font-weight: 700;
  margin-bottom: 4rpx;
}
.quota-num.used {
  color: #ef4444;
}
.quota-num.remaining {
  color: #10b981;
}
.quota-num.total {
  color: $color-text-secondary;
}
.quota-label {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.tip-card {
  display: flex;
  align-items: center;
  gap: 16rpx;
  background: $color-primary-bg;
  border-radius: $radius-lg;
  padding: 24rpx;
}
.tip-text {
  font-size: $font-sm;
  color: $color-primary;
  flex: 1;
}
</style>
