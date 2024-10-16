<script setup lang="ts">
import { DatasetQuery } from "@/api/dataset/model";
import LongitudinalWaterfall from "@/components/LongitudinalWaterfall/index.vue";
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
  {
    src: "https://picsum.photos/400/500",
    alt: "Image 11",
  },
  {
    src: "https://picsum.photos/400/500",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/600",
    alt: "Image 11",
  },
  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },

  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },

  {
    src: "https://picsum.photos/id/106/400/300",
    alt: "Image 12",
  },

  {
    src: "https://picsum.photos/200/800",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/800",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/200/700",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/200/400",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/200/500",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },
  {
    src: "https://picsum.photos/400/700",
    alt: "Image 12",
  },
]);

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
      <!-- 纵向瀑布流布局 -->
      <LongitudinalWaterfall :list="images" />
    </el-card>
  </div>
</template>

<style scoped lang="scss">
.waterfall-item {
  position: relative;
  overflow: hidden;
}
</style>
