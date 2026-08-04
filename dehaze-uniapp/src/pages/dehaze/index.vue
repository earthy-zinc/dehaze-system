<template>
  <PageLayout level="L1" title="去雾">
    <view class="dehaze-page">
      <!-- 步骤条占位（规划：上传→算法→参数→处理→对比） -->
      <view class="steps">
        <view class="step" v-for="(step, i) in steps" :key="step">
          <view class="step-dot" :class="{ done: i === 0 }">{{ i + 1 }}</view>
          <text class="step-label">{{ step }}</text>
        </view>
      </view>

      <!-- 流程内容占位 -->
      <view class="flow-placeholder">
        <u-icon name="camera" size="48" color="#d1d5db" />
        <text class="placeholder-text">去雾处理流程建设中</text>
        <text class="placeholder-sub"
          >上传 → 选择算法 → 调节参数 → 处理 → 对比</text
        >
      </view>

      <!-- 开发中提示 -->
      <view class="dev-tip">
        <u-icon name="info-circle" size="14" color="#9ca3af" />
        <text>核心流程按《菜单与页面层级规划》逐步落地</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import PageLayout from "@/layout/index.vue";

/** 处理流程步骤（05 规划：页内步骤流） */
const steps = ["上传图像", "选择算法", "调节参数", "处理", "效果对比"];
</script>

<style lang="scss" scoped>
.dehaze-page {
  padding: 24rpx;
  background: #f9fafb;
  min-height: 100vh;
}

.steps {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #ffffff;
  border-radius: 20rpx;
  padding: 24rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);

  .step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8rpx;
    flex: 1;
  }

  .step-dot {
    width: 48rpx;
    height: 48rpx;
    border-radius: 50%;
    background: #f3f4f6;
    color: #9ca3af;
    font-size: 24rpx;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;

    &.done {
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      color: #ffffff;
    }
  }

  .step-label {
    font-size: 20rpx;
    color: #6b7280;
    white-space: nowrap;
  }
}

.flow-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  background: #ffffff;
  border-radius: 20rpx;
  padding: 96rpx 24rpx;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.04);

  .placeholder-text {
    font-size: 30rpx;
    font-weight: 600;
    color: #374151;
  }

  .placeholder-sub {
    font-size: 24rpx;
    color: #9ca3af;
  }
}

.dev-tip {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  margin-top: 48rpx;
  font-size: 22rpx;
  color: #9ca3af;
}
</style>
