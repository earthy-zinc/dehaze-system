<script lang="ts" setup>
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

const emit = defineEmits(["load"]);

function handleImageLoad() {
  emit("load");
}
</script>

<template>
  <img @load="handleImageLoad" :src="src" alt="加载失败" />
</template>

<style lang="scss" scoped>
img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
</style>
