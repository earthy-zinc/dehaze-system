<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import { ImageTypeEnum } from "@/enums/ImageType";

defineOptions({
  name: "ParallelImageUpload",
});

const emit = defineEmits(["onReset"]);
const imageShowStore = useImageShowStore();
const { imageInfo } = toRefs(imageShowStore);

const urls = reactive({
  haze: "",
  pred: "",
  gt: "",
});

function handleChange(url: string, type: ImageTypeEnum) {
  imageShowStore.setImageUrl(url, type);
}

function handleReset() {
  urls.pred = "";
  urls.haze = "";
  urls.gt = "";
}

onMounted(() => {
  let haze = imageInfo.value.images.urls.filter((item) => item.id === 0)[0];
  let pred = imageInfo.value.images.urls.filter((item) => item.id === 1)[0];
  let gt = imageInfo.value.images.urls.filter((item) => item.id === 2)[0];
  if (haze) {
    urls.haze = haze.url;
  }
  if (pred) {
    urls.pred = pred.url;
  }
  if (gt) {
    urls.gt = gt.url;
  }
});
</script>

<template>
  <div class="parallel-container">
    <SingleUpload
      v-model="urls.haze"
      class="upload-component"
      tooltip="上传有雾图像"
      @on-change="(url) => handleChange(url, ImageTypeEnum.HAZE)"
    />
    <SingleUpload
      v-model="urls.pred"
      class="upload-component"
      tooltip="上传预测图像"
      @on-change="(url) => handleChange(url, ImageTypeEnum.PRED)"
    />
    <SingleUpload
      v-model="urls.gt"
      class="upload-component"
      tooltip="上传无雾图像"
      @on-change="(url) => handleChange(url, ImageTypeEnum.CLEAN)"
    />
  </div>
</template>

<style lang="scss" scoped>
.parallel-container {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.upload-component {
  width: 100%;
  height: 100%;
  object-fit: contain; /* 保证图片宽高比 */
}
</style>

<style lang="scss">
.el-upload--picture-card {
  --el-upload-picture-card-size: calc((100vw - 80px) / 3) !important;
}
</style>
