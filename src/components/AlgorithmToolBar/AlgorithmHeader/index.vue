<script setup lang="ts">
defineOptions({
  name: "AlgorithmHeader",
});

defineProps({
  title: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
});
</script>

<template>
  <h2 class="text-center m2">{{ title }}</h2>
  <el-text class="m2">
    {{ description }}
  </el-text>
</template>

<style lang="scss" scoped></style>
