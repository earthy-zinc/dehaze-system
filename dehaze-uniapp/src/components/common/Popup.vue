<template>
  <view v-if="show" class="popup-mask" @click="$emit('close')">
    <view
      class="popup-container"
      :class="[`popup-${mode}`, { 'popup-round': round }]"
      @click.stop
    >
      <slot />
    </view>
  </view>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    show: boolean;
    /** 居中或底部弹出 */
    mode?: "center" | "bottom";
    /** 圆角 */
    round?: boolean;
  }>(),
  { mode: "center", round: false }
);
defineEmits<{ close: [] }>();
</script>

<style lang="scss" scoped>
.popup-mask {
  position: fixed;
  inset: 0;
  background: rgb(0 0 0 / 45%);
  z-index: 1000;
  display: flex;
}
.popup-container {
  background: $color-white;
}
.popup-center {
  margin: auto;
  width: 86%;
  max-height: 80vh;
  overflow: auto;
}
.popup-bottom {
  margin-top: auto;
  width: 100%;
  max-height: 80vh;
  overflow: auto;
}
.popup-round {
  border-radius: $radius-lg;
}
.popup-bottom.popup-round {
  border-radius: $radius-lg $radius-lg 0 0;
}
</style>
