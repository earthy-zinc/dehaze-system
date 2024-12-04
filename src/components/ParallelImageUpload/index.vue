<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";

defineOptions({
  name: "ParallelImageUpload",
});

const imageShowStore = useImageShowStore();
const { imageInfo } = toRefs(imageShowStore);
</script>

<template>
  <div class="parallel-container">
    <SingleUpload
      v-model="imageInfo.images.urls[0].url"
      class="upload-component"
      tooltip="上传有雾图像"
    />
    <SingleUpload
      v-model="imageInfo.images.urls[1].url"
      class="upload-component"
      tooltip="上传预测图像"
    />
    <SingleUpload
      v-model="imageInfo.images.urls[2].url"
      class="upload-component"
      tooltip="上传无雾图像"
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
  --el-upload-picture-card-size: 60vh !important;
}
</style>
