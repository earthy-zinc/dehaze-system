<script lang="ts" setup>
import { CSSProperties, PropType } from "vue";

defineOptions({
  name: "ParallelImageShow",
});

const props = defineProps({
  images: {
    type: Array as PropType<string[]>,
    required: true,
  },
  showMask: {
    type: Boolean,
    default: false,
  },
  contrast: {
    type: Number,
    default: 0,
  },
  brightness: {
    type: Number,
    default: 0,
  },
});

const { images, showMask, contrast, brightness } = toRefs(props);
const maskRef = ref<HTMLDivElement>();
const maskStyle = ref<CSSProperties>({
  width: "100px",
  height: "100px",
  backgroundColor: "rgba(0, 0, 0, 0.5)",
  position: "absolute",
  display: "none",
});
</script>

<template>
  <div></div>
</template>

<style lang="scss" scoped></style>
