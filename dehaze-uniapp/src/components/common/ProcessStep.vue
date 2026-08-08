<template>
  <view class="process-step" @click="handleClick">
    <view class="step-number">{{ number }}</view>
    <view class="step-icon">
      <SvgIcon :name="icon" size="32" color="#ffffff" />
    </view>
    <text class="step-title">{{ title }}</text>
    <text class="step-desc">{{ description }}</text>
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  number: string;
  title: string;
  description: string;
  icon: string;
}

interface Emits {
  (e: "click"): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const handleClick = () => {
  emit("click");
};
</script>

<style lang="scss" scoped>
.process-step {
  position: relative;
  background: #ffffff;
  border-radius: 20rpx;
  padding: 32rpx 24rpx;
  text-align: center;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
  border: 2rpx solid transparent;

  &:active {
    transform: scale(0.98);
  }
}

.step-number {
  position: absolute;
  top: 16rpx;
  right: 16rpx;
  width: 32rpx;
  height: 32rpx;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  color: #3b82f6;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24rpx;
  font-weight: 700;
}

.step-icon {
  width: 64rpx;
  height: 64rpx;
  margin: 0 auto 20rpx;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border-radius: 16rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
}

.step-title {
  display: block;
  font-size: 28rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 12rpx;
}

.step-desc {
  display: block;
  font-size: 24rpx;
  color: #6b7280;
  line-height: 1.6;
  white-space: pre-line;
}
</style>
