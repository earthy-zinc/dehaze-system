<script setup lang="ts">
import { DatasetQuery } from "@/api/dataset/model";

defineOptions({
  name: "DataItem",
  inheritAttrs: false,
});

const queryParams = reactive<DatasetQuery>({});

const images = ref([
  { src: "https://picsum.photos/id/1/200/300", alt: "Image 1" },
  { src: "https://picsum.photos/id/2/200/300", alt: "Image 2" },
  { src: "https://picsum.photos/id/3/200/300", alt: "Image 3" },
  { src: "https://picsum.photos/id/4/200/300", alt: "Image 4" },
  { src: "https://picsum.photos/id/5/200/300", alt: "Image 5" },
  { src: "https://picsum.photos/id/6/200/300", alt: "Image 6" },
  { src: "https://picsum.photos/id/7/200/300", alt: "Image 7" },
  { src: "https://picsum.photos/id/8/200/300", alt: "Image 8" },
  { src: "https://picsum.photos/id/9/200/300", alt: "Image 9" },
  { src: "https://picsum.photos/id/10/200/300", alt: "Image 10" },
  { src: "https://picsum.photos/id/11/200/300", alt: "Image 11" },
  { src: "https://picsum.photos/id/12/200/300", alt: "Image 12" },
  { src: "https://picsum.photos/id/13/200/300", alt: "Image 13" },
]);

// 容器ref
const container = ref(ElForm);

function handleQuery() {}
function resetQuery() {}
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :model="queryParams" :inline="true">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            placeholder="图片名称"
            clearable
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><template #icon><i-ep-search /></template>搜索</el-button
          >
          <el-button @click="resetQuery">
            <template #icon><i-ep-refresh /></template>
            重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card shadow="never">
      <div ref="container" class="waterfall-container">
        <div
          v-for="(image, index) in images"
          :key="index"
          class="waterfall-item"
        >
          <img
            v-lazy="{ src: image.src, loading: '', error: '' }"
            :alt="image.alt"
          />
        </div>
      </div>
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.waterfall-container {
  position: relative;
  display: grid;

  /* 自动创建多列，每列最小宽度200px，最大宽度为1fr（占据可用空间的一份） */
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));

  /* 自动行高，最小为0，根据内容自适应 */
  grid-auto-rows: minmax(0, auto);
  grid-gap: 10px;
  padding-bottom: 10px;
}

.waterfall-item {
  position: relative;
  overflow: hidden;
}

.waterfall-item img {
  width: 100%;
  height: auto;
}
</style>
