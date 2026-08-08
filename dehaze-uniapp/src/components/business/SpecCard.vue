<template>
  <view class="spec-card">
    <view class="spec-icon">
      <SvgIcon :name="icon" size="36" color="#ffffff" />
    </view>
    <text class="spec-title">{{ title }}</text>
    <text class="spec-value">{{ value }}</text>
    <text class="spec-desc">{{ description }}</text>
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  icon: string;
  title: string;
  value: string;
  description: string;
}

defineProps<Props>();
</script>

<style lang="scss" scoped>
.spec-card {
  text-align: center;
  padding: 24rpx;
  background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
  border-radius: 20rpx;
  border: 2rpx solid #e5e7eb;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    transform: translateY(-8rpx);
    border-color: #3b82f6;
    box-shadow: 0 12rpx 32rpx rgba(59, 130, 246, 0.15);
  }
}

.spec-icon {
  width: 72rpx;
  height: 72rpx;
  margin: 0 auto 24rpx;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
}

.spec-card:hover .spec-icon {
  transform: scale(1.1) rotate(10deg);
}

.spec-title {
  display: block;
  font-size: 24rpx;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 12rpx;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.spec-value {
  display: block;
  font-size: 48rpx;
  font-weight: 700;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: 8rpx;
  line-height: 1;
}

.spec-desc {
  display: block;
  font-size: 22rpx;
  color: #9ca3af;
  line-height: 1.5;
}
</style>
