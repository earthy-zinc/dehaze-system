<template>
  <view class="menu-item" @click="$emit('click')">
    <view class="item-icon">
      <SvgIcon :name="icon" size="20" :color="iconColor" />
    </view>
    <text class="item-label">{{ label }}</text>
    <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
  </view>
</template>

<script lang="ts" setup>
import SvgIcon from "@/components/SvgIcon/index.vue";

interface Props {
  icon: string;
  iconColor: string;
  label: string;
}

defineProps<Props>();
defineEmits(["click"]);
</script>

<style lang="scss" scoped>
.menu-item {
  display: flex;
  align-items: center;
  gap: 20rpx;
  padding: 24rpx 28rpx;
  &:active {
    background: #f9fafb;
  }
  & + & {
    border-top: 1rpx solid $color-border-light;
  }
}
.item-icon {
  width: 40rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.item-label {
  flex: 1;
  font-size: $font-md;
  color: $color-text-primary;
}
</style>
