<script setup lang="ts">
defineOptions({
  name: "Magnifier",
});

const props = defineProps({
  src: {
    type: String,
    required: true,
  },
  highSrc: {
    type: String,
    default: "",
  },
  width: {
    type: Number,
    default: 0,
  },
  height: {
    type: Number,
    default: 0,
  },
  scale: {
    type: Number,
    default: 1.5,
  },
});
</script>

<template>
  <div>test</div>
</template>

<style lang="scss" scoped></style>
