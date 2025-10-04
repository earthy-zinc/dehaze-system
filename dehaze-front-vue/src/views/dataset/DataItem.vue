<script lang="ts" setup>
import { Dataset, ImageItem, ImageItemQuery } from "@/api/dataset/model";
import LongitudinalWaterfall from "@/components/LongitudinalWaterfall/index.vue";
import { ViewCard } from "@/components/Waterfall/types";
import DatasetAPI from "@/api/dataset";
import { api as viewerApi } from "v-viewer";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { changeUrl } from "@/utils";

defineOptions({
  name: "DataItem",
  inheritAttrs: false,
});

const datasetId = ref<number>(1);
const totalPages = ref<number>(1);
const queryParams = reactive<ImageItemQuery>({ pageNum: 1, pageSize: 10 });
const renderCount = ref<number>(0);
let datasetInfo = ref<Dataset>({
  id: 0,
  parentId: 0,
  name: "",
  type: "",
  description: "",
  createTime: new Date(),
  updateTime: new Date(),
  path: "",
  size: "",
  total: 0,
});
let images = ref<ViewCard[]>([]);
let imageData = reactive<ImageItem[]>([]);
type ImageType = { id: number; type: string; enabled: boolean };

const imageTypes = ref<ImageType[]>([
  { id: 0, type: ImageTypeEnum.CLEAN, enabled: true },
  { id: 1, type: ImageTypeEnum.HAZE, enabled: false },
]);

const curImageType = computed(() => {
  return imageTypes.value.find((item) => item.enabled);
});

let loadingBarRef = ref();
const loadingObserver = ref();

const route = useRoute();
const { width } = useWindowSize();

const itemWidth = computed(() => {
  const breakpoints = [
    { minWidth: 0, columns: 1 },
    { minWidth: 768, columns: 2 },
    { minWidth: 1024, columns: 3 },
    { minWidth: 1280, columns: 4 },
  ];
  breakpoints.forEach((breakpoint) => {
    if (width.value >= breakpoint.minWidth)
      return Math.floor((width.value - 60) / breakpoint.columns);
  });
  return 400;
});

async function handleQuery() {
  DatasetAPI.getImageItem(datasetId.value, queryParams)
    .then((data) => {
      imageData = data.list;
      totalPages.value = Math.ceil(data.total / queryParams.pageSize);
      switchImageUrl(curImageType.value?.id || 0);
    })
    .then(() => {
      if (imageData.length > 0) {
        let tempImageTypes = [] as ImageType[];
        imageData[0].imgUrl.forEach((item, index) => {
          tempImageTypes.push({
            id: index,
            type: item.type,
            enabled: index === 0,
          });
        });
        imageTypes.value = tempImageTypes;
      }
    })
    .catch((err) => {
      console.log(err);
    });
}

function switchImageUrl(id: number) {
  images.value = imageData.map((item) => {
    return {
      id: item.id,
      src: changeUrl(item.imgUrl[id].url),
      originSrc: changeUrl(item.imgUrl[id].originUrl!),
      alt: item.imgUrl[id].description,
    };
  });
}

function resetQuery() {}

function handleImageTypeChange(typeId: number) {
  imageTypes.value.forEach((item) => {
    item.enabled = item.id === typeId;
  });
  switchImageUrl(curImageType.value?.id || typeId);
}

function showBigPicture(itemId: number) {
  let curSelectedImages = () => {
    let curImageItem = imageData.find((item) => item.id === itemId);
    let result = [] as string[];
    if (curImageItem) {
      curImageItem.imgUrl.map((item) => {
        item.originUrl
          ? result.push(changeUrl(item.originUrl))
          : result.push(changeUrl(item.url));
      });
    }
    return result;
  };
  viewerApi({
    images: curSelectedImages(),
  })
    .show()
    .update()
    .view(curImageType.value?.id || 0);
}

onMounted(async () => {
  datasetId.value = Number(route.params.id);
  await DatasetAPI.getDatasetInfoById(datasetId.value).then((data) => {
    datasetInfo.value = data;
  });
  await handleQuery();
  loadingObserver.value = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        queryParams.pageNum++;
        DatasetAPI.getImageItem(datasetId.value, queryParams).then((data) => {
          imageData.push(...data.list);
          switchImageUrl(curImageType.value?.id || 0);
        });
      }
    });
  });

  if (loadingBarRef.value) {
    let loadingBarEl = loadingBarRef.value.$el as HTMLElement;
    loadingBarEl.style.transform = "translate3d(0, 3000px, 0)";
    loadingObserver.value.observe(loadingBarEl);
    setTimeout(() => (loadingBarEl.style.transform = "none"), 1000);
  }
});

onUnmounted(() => loadingObserver.value?.disconnect());
</script>

<template>
  <div class="app-container">
    <el-card shadow="never">
      <!-- 纵向瀑布流布局 -->
      <h1 class="mt-2 mb-3" style="text-align: center">
        {{ datasetInfo.name }} {{ datasetInfo.type }}数据集
      </h1>
      <p class="mr-3 ml-3 mb-6" style="text-indent: 2em">
        {{ datasetInfo.description }}
      </p>
      <div class="mb-1" style="display: flex; justify-content: space-between">
        <el-button-group>
          <el-button
            v-for="imageType in imageTypes"
            :key="imageType.id"
            :type="imageType.enabled ? 'primary' : ''"
            plain
            @click="handleImageTypeChange(imageType.id)"
          >
            {{ imageType.type }}
          </el-button>
        </el-button-group>

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
      <el-skeleton
        v-if="renderCount === 0 && datasetInfo.total !== 0"
        :rows="12"
        animated
      />
      <LongitudinalWaterfall
        :list="images"
        :width="itemWidth"
        @click-item="showBigPicture"
        @after-render="() => renderCount++"
      />
      <el-divider
        v-show="
          totalPages > 1 &&
          renderCount >= queryParams.pageNum - 1 &&
          queryParams.pageNum < totalPages
        "
        ref="loadingBarRef"
        >正在加载，请稍后
      </el-divider>
    </el-card>
  </div>
</template>

<style lang="scss" scoped></style>
