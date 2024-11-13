<script lang="ts" setup>
import { ImageItem, ImageItemQuery } from "@/api/dataset/model";
import LongitudinalWaterfall from "@/components/LongitudinalWaterfall/index.vue";
import { ViewCard } from "@/components/Waterfall/types";
import DatasetAPI from "@/api/dataset";

defineOptions({
  name: "DataItem",
  inheritAttrs: false,
});

const datasetId = ref<number>(1);
const queryParams = reactive<ImageItemQuery>({ pageNum: 1, pageSize: 50 });

const images = ref<ViewCard[]>([]);
const imageData = ref<ImageItem[]>([]);

const route = useRoute();

function handleQuery() {
  DatasetAPI.getImageItem(datasetId.value, queryParams)
    .then((data) => {
      imageData.value = data.list;
      let curImages = [] as ViewCard[];
      imageData.value.map((item) =>
        curImages.push({
          src: item.imgUrl[0].originUrl,
          alt: item.id,
        })
      );
      images.value = curImages;
    })
    .catch((err) => {
      console.log(err);
    });
}

function resetQuery() {}

onMounted(() => {
  datasetId.value = Number(route.params.id);
  handleQuery();
});
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="图片名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <template #icon>
              <i-ep-search />
            </template>
            搜索
          </el-button>
          <el-button @click="resetQuery">
            <template #icon>
              <i-ep-refresh />
            </template>
            重置
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <el-card shadow="never">
      <!-- 纵向瀑布流布局 -->
      <LongitudinalWaterfall :list="images" />
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.waterfall-item {
  position: relative;
  overflow: hidden;
}
</style>
