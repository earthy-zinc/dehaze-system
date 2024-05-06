<script setup lang="ts">
import DraggableLine from "@/components/DraggableLine.vue";

const image1 = ref("https://picsum.photos/id/1/800/800");
const image2 = ref("https://picsum.photos/id/2/800/800");
const imageContainerRef = ref();

const contrast = ref(0);
const brightness = ref(0);
const x = ref(0);

const sliderValue = computed(() => {
  return (1 - x.value / 800) * 100;
});

const handleContrastChange = (value: number) => {
  contrast.value = value;
  updateImageStyle();
};

const handleBrightnessChange = (value: number) => {
  brightness.value = value;
  updateImageStyle();
};

function updateImageStyle() {
  function transform(x: number) {
    return 0.5 * x + 100;
  }
  const images = document.querySelectorAll(".image-container img");
  images.forEach((img: Element) => {
    (img as HTMLImageElement).style.filter = `
    contrast(${transform(contrast.value)}%)
    brightness(${transform(brightness.value)}%)`;
  });
}
</script>

<template>
  <div class="image-comparison">
    <div class="image-container">
      <img
        :src="image1"
        :style="{ clipPath: `inset(0 ${sliderValue}% 0 0)` }"
        alt=""
      />
      <img
        :src="image2"
        :style="{ clipPath: `inset(0 0 0 ${100 - sliderValue}%)` }"
        alt=""
      />

      <DraggableLine @update:offset="(value) => (x = value)" />
    </div>

    <div class="controls">
      <div class="flex flex-items-center w-90">
        <span class="w-35">对比度调整</span>
        <el-slider
          v-model="contrast"
          @input="handleContrastChange"
          :min="-100"
          :max="100"
        />
      </div>
      <div class="ml-6 flex flex-items-center w-90">
        <span class="w-25">亮度调整</span>
        <el-slider
          v-model="brightness"
          @input="handleBrightnessChange"
          :min="-100"
          :max="100"
        />
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.image-comparison {
  position: relative;
  margin-left: 100px;
}

.image-container {
  position: relative;
  width: 800px;
  height: 800px;
  overflow: hidden;
}

.image-container img {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.controls {
  display: flex;
  margin-top: 20px;
}
</style>
