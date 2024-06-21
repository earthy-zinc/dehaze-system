<script setup lang="ts">
import { DatasetQuery } from "@/api/dataset/model";
import Waterfall from "@/components/Waterfall/index.vue";
import { ViewCard } from "@/components/Waterfall/types";

defineOptions({
  name: "DataItem",
  inheritAttrs: false,
});

const queryParams = reactive<DatasetQuery>({});

const images = ref<ViewCard[]>([
  {
    src: "https://fastly.picsum.photos/id/2/200/300.jpg?hmac=HiDjvfge5yCzj935PIMj1qOf4KtvrfqWX3j4z1huDaU",
    alt: "Image 1",
  },
  {
    src: "https://fastly.picsum.photos/id/1/200/300.jpg?hmac=jH5bDkLr6Tgy3oAg5khKCHeunZMHq0ehBZr6vGifPLY",
    alt: "Image 2",
  },
  {
    src: "https://fastly.picsum.photos/id/6/200/300.jpg?hmac=a4Gfsl7hyAvOnmQtzoEkQmbiLJFl7otISIdoYQWqJCo",
    alt: "Image 3",
  },
  {
    src: "https://fastly.picsum.photos/id/7/200/300.jpg?hmac=_vgE8dZdzp3B8T1C9VrGrIMBkDOkFYbJNWqzJD47xNg",
    alt: "Image 4",
  },
  {
    src: "https://fastly.picsum.photos/id/8/200/300.jpg?hmac=t2Camsbqc4OfjWMxFDwB32A8N4eu7Ido7ZV1elq4o5M",
    alt: "Image 5",
  },
  {
    src: "https://fastly.picsum.photos/id/9/200/300.jpg?hmac=BguC5kAGl-YR4FEjhjm0b2XWbynYsk3s3QQZUie5aBo",
    alt: "Image 6",
  },
  {
    src: "https://fastly.picsum.photos/id/10/200/300.jpg?hmac=94QiqvBcKJMHpneU69KYg2pky8aZ6iBzKrAuhSUBB9s",
    alt: "Image 7",
  },
  {
    src: "https://fastly.picsum.photos/id/11/200/300.jpg?hmac=n9AzdbWCOaV1wXkmrRfw5OulrzXJc0PgSFj4st8d6ys",
    alt: "Image 8",
  },
  {
    src: "https://fastly.picsum.photos/id/12/200/300.jpg?hmac=H975kfBbjoaBk4vHQpqpz-uxYLeRtC67xb6WSe_wPkk",
    alt: "Image 9",
  },
  {
    src: "https://fastly.picsum.photos/id/13/200/300.jpg?hmac=UHtWCvsKxIfcA_gIse7Rc6MH6nI3OGl0dzaCSSsYqas",
    alt: "Image 10",
  },
]);

// 容器ref
const container = ref();

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
        <div class="waterfall-tiem" v-for="(item, index) in 8" :key="index">
          <Waterfall :list="images" :speed="item % 2 ? 1 : -1" />
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
  overflow: hidden;
}

.waterfall-item {
  position: relative;
  overflow: hidden;
}
</style>
