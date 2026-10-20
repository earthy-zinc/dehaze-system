<template>
  <view class="feature-card" @click="handleClick">
    <view class="tool-icon-wrapper">
      <SvgIcon :name="tool.icon" size="28" color="#3b82f6" />
    </view>
    <text class="tool-title">{{ tool.title }}</text>
    <text class="tool-desc">{{ tool.description }}</text>
  </view>
</template>

<script lang="ts" setup>
import type { ToolItem } from "@/pages/home/data/homeData";
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  tool: ToolItem;
}

interface Emits {
  (e: "click", tool: ToolItem): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const handleClick = () => {
  emit("click", props.tool);
};
</script>

<style lang="scss" scoped>
.feature-card {
  background: #ffffff;
  border-radius: 20rpx;
  padding: 32rpx;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
  border: 2rpx solid transparent;

  &:hover {
    transform: translateY(-8rpx);
    box-shadow: 0 12rpx 32rpx rgba(0, 0, 0, 0.1);
    border-color: #3b82f6;
  }

  &:active {
    transform: scale(0.98);
  }
}

.tool-icon-wrapper {
  width: 56rpx;
  height: 56rpx;
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border-radius: 14rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20rpx;
  transition: all 0.3s;
}

.feature-card:hover .tool-icon-wrapper {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  transform: scale(1.1);
}

.feature-card:hover .tool-icon-wrapper :deep(.svg-icon) {
  color: #ffffff !important;
}

.tool-title {
  display: block;
  font-size: 32rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 12rpx;
}

.tool-desc {
  display: block;
  font-size: 24rpx;
  color: #6b7280;
  line-height: 1.6;
}
</style>
