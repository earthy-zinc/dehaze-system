<template>
  <svg
    :style="{ width: pxSize, height: pxSize }"
    aria-hidden="true"
    class="svg-icon"
  >
    <use :fill="color" :xlink:href="symbolId" />
  </svg>
</template>

<script lang="ts" setup>
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    /** 图标名（对应 src/assets/icons 下的 svg 文件名，不含扩展名） */
    name: string;
    /** 图标尺寸，单位 px（u-icon 旧 size 直接传入） */
    size?: number | string;
    /** 填充色（u-icon 旧 color 直接传入） */
    color?: string;
  }>(),
  {
    size: "1em",
    color: "",
  }
);

const symbolId = computed(() => `#icon-${props.name}`);

const pxSize = computed(() => {
  const s = props.size;
  if (s === undefined || s === null || s === "") return "1em";
  if (typeof s === "number") return `${s}px`;
  // 字符串类型：纯数字补 px，否则原样（支持 1em / 16px 等）
  return /^\d+(\.\d+)?$/.test(String(s)) ? `${s}px` : String(s);
});
</script>

<style scoped>
.svg-icon {
  display: inline-block;
  width: 1em;
  height: 1em;
  overflow: hidden;
  vertical-align: -0.15em;
  outline: none;
  fill: currentcolor;
}
</style>
