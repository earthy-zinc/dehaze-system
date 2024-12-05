<script lang="ts" setup>
import AlgorithmToolBar from "@/components/AlgorithmToolBar/index.vue";
import { useAlgorithmStore, useSettingsStore } from "@/store";
import Camera from "@/components/Camera/index.vue";
import SingleImageShow from "@/components/SingleImageShow/index.vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import Loading from "@/components/Loading/index.vue";
import FileAPI from "@/api/file";
import ModelAPI from "@/api/model";
import ExampleImageSelect from "@/components/ExampleImageSelect/index.vue";
import { useImageShowStore } from "@/store/modules/imageShow";
import DatasetImageSelect from "@/components/DatasetImageSelect/index.vue";
import { ImageTypeEnum } from "@/enums/ImageType";
import examples from "@/views/presentation/dehaze/exampleImages";

const algorithmStore = useAlgorithmStore();
const imageShowStore = useImageShowStore();
const { themeColor } = useSettingsStore();

const { imageInfo } = toRefs(imageShowStore);
const { images } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);

const exampleImages = ref(examples);
const exampleHazeUrls = computed(() =>
  exampleImages.value.map((item) => item.haze)
);
const cleanUrl = ref("");
const modelOptions = ref<OptionType[]>([]);
const selectedModel = ref<number>();

const show = reactive({
  camera: false,
  singleImage: false,
  example: false,
  loading: false,
  overlap: false,
  effect: true,
});

const disableMore = computed(() => !show.overlap);

function activePage(
  page: "camera" | "singleImage" | "example" | "overlap" | "loading" | "effect"
) {
  show.camera = page === "camera";
  show.singleImage = page === "singleImage";
  show.example = page === "example";
  show.overlap = page === "overlap";
  show.loading = page === "loading";
  show.effect = page === "effect";
}

function handleCameraSave(file: File) {
  // 上传文件
  handleImageUpload(file);
}

function handleImageUpload(file: File) {
  // 上传文件
  FileAPI.upload(file, imageShowStore.modelId)
    .then((res) => {
      // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
      activePage("loading");
      imageShowStore.setImageUrl(res.url, ImageTypeEnum.HAZE);
    })
    .then(() => {
      // 将文件显示到 SingleImageShow 组件中
      activePage("singleImage");
    })
    .catch((err) => {
      ElMessage.error(err);
    });
}

function handleReset() {
  imageShowStore.setImageUrls([]);
  imageShowStore.toggleMagnifierShow();
  activePage("example");
}

// 选择模型后生成对比图（原图 | 去雾图）
function handleGenerateImage() {
  activePage("loading");
  const imgUrl = imgUrls.value[0].url;
  ModelAPI.prediction({
    modelId: Number(selectedModel.value) || 1,
    url: imgUrl,
  })
    .then((res) => {
      // 获取生成后的图片url
      imageShowStore.setImageUrl(res.hazeUrl, ImageTypeEnum.HAZE);
      imageShowStore.setImageUrl(res.predUrl, ImageTypeEnum.PRED);
    })
    .then(() => activePage("overlap"))
    .catch((err) => {
      ElMessage.error(err);
      activePage("singleImage");
    });
}

function handleExampleImageClick(url: string) {
  imageShowStore.setImageUrl(url, ImageTypeEnum.HAZE);
  const clear = exampleImages.value.filter((item) => item.haze === url)[0]
    .clean;
  handleCleanUrl(clear).then((res) => (cleanUrl.value = res));
  activePage("singleImage");
}

// 获取模型选项列表
const getAlgorithmList = async () => {
  await algorithmStore.getAlgorithmOptions();
  modelOptions.value = algorithmStore.algorithmOptions;
};

function handleDatasetImageSelect(haze: string, clear: string) {
  imageShowStore.setImageUrl(haze, ImageTypeEnum.HAZE);
  handleCleanUrl(clear).then((res) => (cleanUrl.value = res));
  dialogVisible.value = false;
  activePage("singleImage");
}

async function handleCleanUrl(url: string) {
  const res = await fetch(url);
  const blob = await res.blob();
  const cleanFile = new File([blob], "clean.jpg", { type: "image/jpeg" });
  const cleanRes = await FileAPI.upload(cleanFile, imageShowStore.modelId);
  return cleanRes.url;
}

const router = useRouter();

function handleEval() {
  router.push("/evaluation/index").then(async () => {
    imageShowStore.setModelId(Number(selectedModel.value) || 1);
    imageShowStore.setImageUrls(imgUrls.value);
    console.log(cleanUrl.value);
    if (cleanUrl.value !== "") {
      imageShowStore.setImageUrl(cleanUrl.value, ImageTypeEnum.CLEAN);
    }
  });
}

const dialogVisible = ref(false);

onMounted(() => {
  getAlgorithmList();
  imageShowStore.setImageUrls([]);
  activePage("example");
});
</script>

<template>
  <div class="app-container">
    <!-- 左侧工具栏 -->
    <AlgorithmToolBar
      :disable-more="disableMore"
      @on-upload="handleImageUpload"
      @on-eval="handleEval"
      @on-take-photo="activePage('camera')"
      @on-reset="handleReset"
      @on-generate="handleGenerateImage"
      @on-select-from-dataset="() => (dialogVisible = true)"
    >
      <!-- 选择模型区域 -->
      <template #default>
        <div class="select-wrap">
          <span>选择去雾模型</span>
          <el-tree-select
            v-model="selectedModel"
            :data="modelOptions"
            placeholder="请选择去雾模型算法"
            style="width: 240px"
          />
        </div>
      </template>
    </AlgorithmToolBar>
    <!-- 右侧功能栏 -->
    <el-card class="flex-center">
      <!-- 样例图片显示 -->
      <ExampleImageSelect
        v-if="show.example"
        :urls="exampleHazeUrls"
        class="example"
        @on-example-select="handleExampleImageClick"
      />
      <!-- 拍照上传 -->
      <Camera
        v-if="show.camera"
        class="camera"
        @on-cancel="activePage('example')"
        @on-save="handleCameraSave"
      />
      <!-- 单图展示 -->
      <SingleImageShow
        v-if="show.singleImage"
        :src="imgUrls[0].url || ''"
        class="single-image"
      />
      <Loading v-if="show.loading" />
      <!-- 重叠展示 -->
      <OverlapImageShow v-if="show.overlap" class="overlap" />
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      title="选择数据集图片"
      top="8vh"
      width="70%"
    >
      <DatasetImageSelect @on-selected="handleDatasetImageSelect" />
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.app-container {
  display: flex;
  height: calc(100vh - $navbar-height);
}

.select-wrap {
  span {
    margin-right: 20px;
    font-size: 18px;
    font-weight: 700;
  }
}

.flex-center {
  width: 64vw;
  overflow-y: hidden;

  .example {
    padding-top: 100px;
  }

  .single-image {
    max-width: calc(64vw - 6vw);
    max-height: calc(100vh - $navbar-height - 40px - 20px - 10px);
  }

  .camera {
    width: calc(64vw - 6vw);
    height: calc(100vh - $navbar-height - 40px - 20px - 10px);
  }

  .overlap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: calc(64vw - 6vw);
    height: calc(100vh - $navbar-height - 40px - 20px - 10px);
    margin: 0 auto;
  }

  .effect-wrap {
    width: 60vw;
  }

  .ev-all-wrap {
    display: flex;
    justify-content: space-between;
    width: 60vw;
    margin-bottom: 20px;

    .ev-wrap {
      width: 30%;
      min-width: 250px;
    }
  }
}

@media screen and (width <=992px) {
  .app-container {
    display: flex;
    flex-wrap: wrap;
    height: auto;
  }

  .flex-center {
    width: 100vw;
    padding-top: 0;
    margin-top: 10px;

    .overlap {
      width: calc(100vw - 6vw);
      height: calc(100vh - $navbar-height - 40px);
    }

    .ev-all-wrap {
      display: flex;
      flex-direction: column;
      width: 82vw;
      margin: 0 auto;

      .ev-wrap {
        width: 100%;
        margin: 10px 0;
      }
    }
  }
}
</style>
