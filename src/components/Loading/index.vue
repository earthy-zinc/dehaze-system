<script lang="ts" setup>
defineOptions({
  name: "Loading",
});
</script>

<template>
  <div class="wrap">
    <div class="loading"><span v-for="i in 5" :key="i"></span></div>
    <span class="ml-2">正在生成图片中，请耐心等候</span>
  </div>
</template>

<style lang="scss" scoped>
.wrap {
  width: 100%;
  text-align: center;
}

.loading {
  display: inline-block;
  width: 50px;
  height: 16px;

  span {
    display: inline-block;
    width: 8px;
    height: 100%;
    margin-left: 2px;
    background: lightgreen;
    border-radius: 4px;
    animation: loadsaven 1.04s ease infinite;
  }

  span:nth-child(1) {
    animation-delay: 0.13s;
  }

  span:nth-child(2) {
    animation-delay: 0.26s;
  }

  span:nth-child(3) {
    animation-delay: 0.39s;
  }

  span:nth-child(4) {
    animation-delay: 0.52s;
  }

  span:nth-child(5) {
    animation-delay: 0.65s;
  }

  @keyframes loadsaven {
    0%,
    100% {
      height: 20px;
      background: lightgreen;
    }

    50% {
      height: 40px;
      margin-top: -20px;
      background: lightblue;
    }
  }
}
</style>
