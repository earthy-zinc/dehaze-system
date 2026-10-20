<script lang="ts" setup>
defineOptions({
  name: "InfiniteFlowList",
});

const props = defineProps({
  urls: {
    type: Array<string>,
    required: true,
  },
  type: {
    type: String,
    default: "down",
  },
  scrollSpeed: {
    type: Number,
    default: 15,
  },
});

const { width, height } = useWindowSize();
const itemHeight = height.value * 0.4;
const doubleItems = ref([...props.urls, ...props.urls]);

const scrollIndex = ref(0);

let timer: number;
let startScroll: () => void;

if (props.type === "down") {
  let originStatus = -2 * props.urls.length * itemHeight + window.innerHeight;
  scrollIndex.value = originStatus;

  startScroll = () => {
    timer = window.setInterval(() => {
      scrollIndex.value += 1;
      if (
        scrollIndex.value >=
        -props.urls.length * itemHeight + window.innerHeight
      ) {
        scrollIndex.value = originStatus;
      }
    }, props.scrollSpeed);
  };
} else {
  startScroll = () => {
    timer = window.setInterval(() => {
      scrollIndex.value -= 1;
      if (scrollIndex.value <= -props.urls.length * itemHeight) {
        scrollIndex.value = 0;
      }
    }, props.scrollSpeed);
  };
}

const stopScroll = () => {
  window.clearInterval(timer);
};

onMounted(() => {
  startScroll();
});

onUnmounted(() => {
  stopScroll();
});
</script>

<template>
  <div class="scroll-container" @mouseout="startScroll" @mouseover="stopScroll">
    <div
      :style="{
        transform: `translateY(${scrollIndex}px)`,
        height: `${doubleItems.length * itemHeight}px`,
      }"
      class="scroll-content"
    >
      <div v-for="(url, index) in doubleItems" :key="index" class="scroll-item">
        <img :src="url.toString()" alt="数据集图片" />
      </div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
$itemHeight: 40vh; // 与script中的itemHeight保持一致

.scroll-container {
  width: 20vw;
  height: 100vh;
  overflow: hidden;

  .scroll-content {
    display: flex;
    flex-direction: column;
  }

  .scroll-item {
    width: 20vw;
    height: $itemHeight;
    padding: 2.5px 5px;
    margin: 5px 10px;
    overflow: hidden;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.2s linear;
    }

    img:hover {
      transform: scale(1.1);
    }
  }
}
</style>
