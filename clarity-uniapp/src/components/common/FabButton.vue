<template>
  <view class="fab-btn" @click="$emit('click')"><slot /></view>
</template>

<script setup lang="ts">
defineEmits<{ click: [] }>();
</script>

<style lang="scss" scoped>
.fab-btn {
  position: fixed;
  right: 40rpx;
  bottom: 100rpx;
  width: 96rpx;
  height: 96rpx;
  border-radius: 50%;
  background: $color-primary;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: $shadow-md;
  z-index: 99;
}
</style>
