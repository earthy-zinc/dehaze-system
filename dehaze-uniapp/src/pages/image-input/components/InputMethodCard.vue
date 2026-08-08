<template>
  <view
    class="input-method-card"
    :class="{ active: active }"
    @click="handleClick"
  >
    <view class="card-icon">
      <SvgIcon :name="icon" :size="28" :color="active ? '#3b82f6' : '#6b7280'" />
    </view>
    <view class="card-content">
      <text class="card-title">{{ title }}</text>
      <text class="card-subtitle">{{ subtitle }}</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import type { InputMethod } from "../data/imageInputData";
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  icon: string;
  title: string;
  subtitle: string;
  method: InputMethod;
  active?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  active: false,
});

const emit = defineEmits<{
  (e: "click", method: InputMethod): void;
}>();

const handleClick = () => {
  emit("click", props.method);
};
</script>

<style lang="scss" scoped>
.input-method-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 28rpx 16rpx;
  background: #ffffff;
  border: 2rpx solid #e5e7eb;
  border-radius: 20rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-color: #3b82f6;
    box-shadow: 0 4rpx 16rpx rgba(59, 130, 246, 0.15);
  }
}

.card-icon {
  width: 72rpx;
  height: 72rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 16rpx;
  margin-bottom: 16rpx;

  .active & {
    background: rgba(59, 130, 246, 0.1);
  }
}

.card-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4rpx;
}

.card-title {
  font-size: 28rpx;
  font-weight: 600;
  color: #1f2937;
}

.card-subtitle {
  font-size: 22rpx;
  color: #9ca3af;
}
</style>
