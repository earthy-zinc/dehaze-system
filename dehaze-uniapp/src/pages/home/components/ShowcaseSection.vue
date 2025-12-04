<template>
  <view class="showcase-section">
    <SectionHeader
      title="一键去雾，效果显著"
      subtitle="智能算法自动识别雾霾程度，精准还原图像细节"
    />
    <view class="comparison-showcase">
      <view class="comparison-item">
        <image
          src="https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/with/20b8704f-d37e-45b9-a6c8-3c5d297e8a98/image_1763727568_3_3.jpg"
          mode="widthFix"
          class="showcase-image"
        />
        <view class="comparison-label">
          <text class="label-before">去雾前</text>
          <text class="label-divider">→</text>
          <text class="label-after">去雾后</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import SectionHeader from "@/components/common/SectionHeader.vue";
</script>

<style lang="scss" scoped>
.showcase-section {
  padding: 80rpx 40rpx;
  background: #f9fafb;
}

.comparison-showcase {
  position: relative;
  border-radius: 24rpx;
  overflow: hidden;
  box-shadow: 0 20rpx 60rpx rgba(0, 0, 0, 0.15);
}

.comparison-item {
  position: relative;
}

.showcase-image {
  width: 100%;
  height: auto;
  display: block;
}

.comparison-label {
  position: absolute;
  bottom: 24rpx;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(10rpx);
  padding: 12rpx 24rpx;
  border-radius: 100rpx;
  display: flex;
  align-items: center;
  gap: 16rpx;
}

.label-before {
  color: #fbbf24;
  font-size: 24rpx;
  font-weight: 600;
}

.label-after {
  color: #34d399;
  font-size: 24rpx;
  font-weight: 600;
}

.label-divider {
  color: #9ca3af;
  font-size: 24rpx;
  font-weight: 600;
}
</style>
