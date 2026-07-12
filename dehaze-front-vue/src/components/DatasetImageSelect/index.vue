<script lang="ts" setup>
import LongitudinalWaterfall from "@/components/LongitudinalWaterfall/index.vue";
import { ViewCard } from "@/components/Waterfall/types";
import {
  IMAGE_TYPE_LABELS,
} from "@/enums/ImageTypeEnum";
import { changeUrl } from "@/utils";
import {
  Dataset,
  DatasetAPI,
  DatasetItemAPI,
  DatasetItemQuery,
  DatasetItemVO,
  ImageUrlVO,
} from "dehaze-sdk-js";

defineOptions({
  name: "DatasetImageSelect",
  inheritAttrs: false,
});

const emit = defineEmits(["onSelected"]);
const selectedDatasetId = ref<number>(1);
const totalPages = ref<number>(1);
const queryParams = reactive<DatasetItemQuery>({ pageNum: 1, pageSize: 10 });
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
  total: 0,
});
let images = ref<ViewCard[]>([]);
let imageData = reactive<DatasetItemVO[]>([]);

/** 当前选中的图片类型（按 type 字符串过滤，如 clear/hazy/trans/depth/segment） */
const selectedType = ref<string>("hazy");

let loadingBarRef = ref();
const loadingObserver = ref();

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

/** 展平数据项下所有图片 */
function extractImagesFromItem(item: DatasetItemVO): ImageUrlVO[] {
  const allImages: ImageUrlVO[] = [];
  if (item.clearImage) allImages.push(item.clearImage);
  if (item.hazyImages && item.hazyImages.length > 0)
    allImages.push(...item.hazyImages);
  return allImages;
}

/** 当前数据集中可用的图片类型集合（用于切换 Tab） */
const availableTypes = computed<string[]>(() => {
  const set = new Set<string>();
  imageData.forEach((item) => {
    extractImagesFromItem(item).forEach((img) => {
      if (img.type) set.add(img.type);
    });
  });
  return Array.from(set);
});

async function handleQuery() {
  queryParams.datasetId = selectedDatasetId.value;
  DatasetItemAPI.getList(queryParams)
    .then((data) => {
      imageData = data.list;
      totalPages.value = Math.ceil(data.total / queryParams.pageSize!);
      // 若当前选中类型不在可用类型中，回退到第一个可用类型
      if (availableTypes.value.length > 0 && !availableTypes.value.includes(selectedType.value)) {
        selectedType.value = availableTypes.value[0];
      }
      switchImageUrl();
    })
    .catch((err) => {
      console.log(err);
    });
}

function switchImageUrl() {
  images.value = imageData.map((item) => {
    const imgs = extractImagesFromItem(item);
    const img = imgs.find((i) => i.type === selectedType.value) || imgs[0];
    if (!img) {
      return { id: item.id, src: "", originSrc: "", alt: "" } as ViewCard;
    }
    return {
      id: item.id,
      src: changeUrl(img.url),
      originSrc: changeUrl(img.originUrl || img.url),
      alt: img.description || "",
    };
  });
}

function resetQuery() {}

function handleImageTypeChange(type: string) {
  selectedType.value = type;
  switchImageUrl();
}

/**
 * 选择图片：按 type 查找清晰图和有雾图，而不是按索引硬编码。
 * 适配新规范：清晰图和有雾图均为可选。
 */
function selectImage(itemId: number) {
  const curImageItem = imageData.find((item) => item.id === itemId);
  if (curImageItem) {
    const imgs = extractImagesFromItem(curImageItem);
    const haze = imgs.find((i) => i.type === "hazy");
    const clear = imgs.find((i) => i.type === "clear");
    if (haze && clear) {
      emit("onSelected", changeUrl(haze.originUrl || haze.url), changeUrl(clear.originUrl || clear.url));
    }
  }
}

async function handleSelectDataset() {
  await DatasetAPI.getDatasetInfoById(selectedDatasetId.value).then((data) => {
    datasetInfo.value = data;
  });
  await handleQuery();
  loadingObserver.value = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        queryParams.pageNum!++;
        DatasetItemAPI.getList(queryParams).then((data) => {
          imageData.push(...data.list);
          switchImageUrl();
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
}

watch(
  () => selectedDatasetId.value,
  () => handleSelectDataset()
);

const datasetOptions = ref<OptionType[]>([]);

onMounted(async () => {
  DatasetAPI.getOptions().then((res) => {
    datasetOptions.value = res;
  });
  await handleSelectDataset();
});

onUnmounted(() => loadingObserver.value?.disconnect());
</script>

<template>
  <el-card :body-style="{ overflowY: 'scroll', height: '73vh' }" shadow="never">
    <el-form>
      <el-form-item label="选择数据集">
        <el-tree-select
          v-model="selectedDatasetId"
          :data="datasetOptions"
          check-strictly
        />
      </el-form-item>
    </el-form>

    <div class="mb-1" style="display: flex; justify-content: space-between">
      <el-button-group>
        <el-button
          v-for="type in availableTypes"
          :key="type"
          :type="selectedType === type ? 'primary' : ''"
          plain
          @click="handleImageTypeChange(type)"
        >
          {{ IMAGE_TYPE_LABELS[type] || type }}
        </el-button>
      </el-button-group>

      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keyword">
          <el-input
            v-model="queryParams.keyword"
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
      @click-item="selectImage"
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
</template>

<style lang="scss" scoped></style>
