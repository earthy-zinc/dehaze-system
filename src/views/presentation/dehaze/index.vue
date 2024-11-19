<script lang="ts" setup>
import { MagnifierInfo, Point } from "@/components/AlgorithmToolBar/types";
import AlgorithmToolBar from "@/components/AlgorithmToolBar/index.vue";
import { useAlgorithmStore } from "@/store";
import EffectDisplay from "@/components/EffectDisplay/index.vue";
import Camera from "@/components/Camera/index.vue";
import SingleImageShow from "@/components/SingleImageShow/index.vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import Loading from "@/components/Loading/index.vue";
import Evaluation from "@/components/Evaluation/index.vue";
import FileAPI from "@/api/file";
import ModelAPI from "@/api/model";

const algorithmStore = useAlgorithmStore();

const image1 = ref(
  "https://ai-resource.ailabtools.com/resource/166-before.webp"
);
const image2 = ref(
  "https://ai-resource.ailabtools.com/resource/166-after.webp"
);
const showMask = ref(false);
const contrast = ref(0);
const brightness = ref(0);
const originScale = ref(1);
const point = ref<Point>({
  x: 0,
  y: 0,
});
const exampleImageUrls = ref<String[]>([
  "http://172.16.3.113:8989/api/v1/files/dataset/thumbnail/Dense-Haze/hazy/01_hazy.png",
]);
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

const { width } = useWindowSize();
const disableMore = computed(() => !show.overlap);
const magnifier = computed(() => {
  return {
    imgUrls: [image1.value, image2.value],
    radius: Math.floor((width.value * 0.3 - 90) / 4),
    originScale: originScale.value,
    point: point.value,
  } as MagnifierInfo;
});

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
  activePage("camera");
}

function handleImageUpload(file: File) {
  // 上传文件
  FileAPI.upload(file)
    .then((res) => {
      // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
      activePage("loading");
      image1.value = res.url;
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
  image1.value = "";
  image2.value = "";
  showMask.value = false;
  activePage("example");
  // activePage("effect");
}

// 选择模型后生成对比图（原图 | 去雾图）
function handleGenerateImage() {
  activePage("loading");
  console.log(selectedModel.value);
  ModelAPI.prediction({
    modelId: Number(selectedModel.value) || 1,
    input: image1.value,
  })
    .then((res) => {
      // 获取生成后的图片url
      image2.value = res[0].output.url;
    })
    .then(() => activePage("overlap"))
    .catch((err) => {
      ElMessage.error(err);
      activePage("singleImage");
    });
}

function handleExampleImageClick(url: string) {
  image1.value = url;
  activePage("singleImage");
}

function handleMouseover(p: Point) {
  point.value.x = p.x;
  point.value.y = p.y;
}

// 获取模型选项列表
const getAlgorithmList = async () => {
  await algorithmStore.getAlgorithmOptions();
  modelOptions.value = algorithmStore.algorithmOptions;
};

onMounted(() => {
  getAlgorithmList();
  activePage("example");
});
</script>

<template>
  <div class="app-container">
    <!-- 左侧工具栏 -->
    <AlgorithmToolBar
      :disable-more="disableMore"
      :magnifier="magnifier"
      @on-upload="handleImageUpload"
      @on-take-photo="activePage('camera')"
      @on-reset="handleReset"
      @on-generate="handleGenerateImage"
      @on-magnifier-change="(flag) => (showMask = flag)"
      @on-brightness-change="(value) => (brightness = value)"
      @on-contrast-change="(value) => (contrast = value)"
    >
      <!-- 选择模型区域 -->
      <template #default>
        <div class="select-wrap">
          <span>选择去雾模型</span>
          <el-select
            v-model="selectedModel"
            filterable
            placeholder="请选择去雾模型算法"
            style="width: 240px"
          >
            <el-option-group
              v-for="group in modelOptions"
              :key="group.label"
              :label="group.label"
            >
              <el-option
                v-for="item in group.children"
                :key="item.value"
                :label="item.label"
                :value="item.value"
              />
            </el-option-group>
          </el-select>
        </div>
      </template>
    </AlgorithmToolBar>
    <!-- 右侧功能栏 -->
    <el-card class="flex-center">
      <EffectDisplay
        v-show="show.effect"
        :urls="[image1, image2]"
        class="effect-wrap"
      />
      <!-- 样例图片显示 -->
      <ExampleImageSelect
        class="example"
        v-if="show.example"
        :urls="exampleImageUrls"
        @on-example-select="handleExampleImageClick"
      />
      <!-- 拍照上传 -->
      <Camera
        v-if="show.camera"
        @on-cancel="activePage('example')"
        @on-save="handleCameraSave"
      />
      <!-- 单图展示 -->
      <SingleImageShow
        v-if="show.singleImage"
        :src="image1"
        class="single-image"
      />
      <Loading v-if="show.loading" />
      <!-- 评价指标 -->
      <!-- <div v-if="show.overlap" ref="evRef" class="ev-all-wrap">
        <div class="ev-wrap">
          <Evaluation />
        </div>
        <div class="ev-wrap">
          <Evaluation />
        </div>
        <div class="ev-wrap">
          <Evaluation />
        </div>
      </div> -->
      <!-- 重叠展示 -->
      <OverlapImageShow
        v-if="show.overlap"
        :brightness="brightness"
        :contrast="contrast"
        :image1="image1"
        :image2="image2"
        :show-mask="showMask"
        class="overlap"
        @on-origin-scale-change="(value) => (originScale = value)"
        @on-mouseover="handleMouseover"
      />
    </el-card>
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
  overflow-y: scroll;

  .example {
    padding-top: 100px;
  }

  .single-image {
    height: 500px;
  }

  .overlap {
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
      width: 100%;
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
