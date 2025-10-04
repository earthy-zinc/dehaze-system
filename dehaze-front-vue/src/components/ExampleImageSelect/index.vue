<script setup lang="ts">
defineOptions({
  name: "ExampleImageSelect",
});

defineProps({
  urls: {
    type: Array as PropType<string[]>,
    required: false,
  },
});

const emit = defineEmits(["onExampleSelect"]);
</script>

<template>
  <div class="example-container">
    <div class="text-center">选一张图片试一下吧</div>
    <div id="example-img-container">
      <img
        class="example-img"
        v-for="url in urls"
        :key="url"
        :src="url"
        alt="样张"
        @click="emit('onExampleSelect', url)"
      />
    </div>
    <div class="example-logo">
      <img src="@/assets/logo.png" alt="logo" />
    </div>
  </div>
</template>

<style lang="scss" scoped>
.example-container {
  display: grid;
  grid-template-rows: 0.1fr 0.4fr 3.2fr;
  grid-template-columns: 1fr;
  height: calc(100vh - $navbar-height - 2 * var(--el-card-padding) - 22px);
}

#example-img-container {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 20px;
  margin-bottom: 20px;

  img {
    width: auto;
    height: 100px;
    cursor: pointer;
  }

  img:hover {
    border: 2px solid var(--el-color-primary);
    transform: scale(1.1);
  }
}

.example-logo {
  text-align: center;

  img {
    width: auto;
    height: 100px;
    margin-top: 15vh;
  }
}
</style>
