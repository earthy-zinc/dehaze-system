<script setup lang="ts">
defineOptions({
  name: "SingleImageShow",
});

const props = defineProps({
  src: {
    type: String,
    required: true,
  },
});

const { src } = toRefs(props);
</script>

<template>
  <img :src="src" alt="" />
</template>

<style lang="scss" scoped>
img {
  width: 100%;
  max-width: calc(69vw - 2 * var(--el-card-padding));
  height: 100%;
  max-height: calc(100vh - $navbar-height - 2 * var(--el-card-padding) - 22px);
  object-fit: contain;
}
</style>
