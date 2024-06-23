<script lang="ts" setup>
import { MagnifierInfo, Point } from "@/components/AlgorithmToolBar/types";
import AlgorithmToolBar from "@/components/AlgorithmToolBar/index.vue";
import ExampleImageSelect from "@/components/ExampleImageSelect/index.vue";
import Camera from "@/components/Camera/index.vue";
import SingleImageShow from "@/components/SingleImageShow/index.vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import Loading from "@/components/Loading/index.vue";

const image1 = ref("");
const image2 = ref("");
const showMask = ref(false);
const contrast = ref(0);
const brightness = ref(0);
const originScale = ref(1);
const point = ref<Point>({
  x: 0,
  y: 0,
});
const exampleImageUrls = ref<String[]>([]);

const show = reactive({
  camera: false,
  singleImage: false,
  example: true,
  loading: false,
  overlap: false,
});

const { width, height } = useWindowSize();
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
  page: "camera" | "singleImage" | "example" | "overlap" | "loading"
) {
  show.camera = page === "camera";
  show.singleImage = page === "singleImage";
  show.example = page === "example";
  show.overlap = page === "overlap";
  show.loading = page === "loading";
}

function handleCameraSave(file: File) {
  // 上传文件
}

function handleImageUpload(file: File) {
  // 上传文件
}

function handleReset() {
  image1.value = "";
  image2.value = "";
  showMask.value = false;
  activePage("example");
}

function handleGenerateImage() {
  activePage("loading");
  setTimeout(() => {
    activePage("overlap");
  }, 1000);
}

function handleExampleImageClick(url: string) {
  image1.value = url;
  activePage("singleImage");
}

function handleMouseover(p: Point) {
  point.value.x = p.x;
  point.value.y = p.y;
}
</script>

<template>
  <div class="app-container">
    <AlgorithmToolBar
      :disable-more="disableMore"
      :magnifier="magnifier"
      @on-upload="handleImageUpload"
      @on-take-photo="activePage('camera')"
      @on-reset="handleReset"
      @on-generate="handleGenerateImage"
      @on-magnifier-change="(flag: boolean) => (showMask = flag)"
      @on-brightness-change="(value: number) => (brightness = value)"
      @on-contrast-change="(value: number) => (contrast = value)"
    />
    <el-card class="flex-center" style="width: 69vw">
      <ExampleImageSelect
        v-if="show.example"
        :urls="exampleImageUrls"
        @on-example-select="handleExampleImageClick"
      />
      <Camera
        v-if="show.camera"
        @on-cancel="activePage('example')"
        @on-save="handleImageUpload"
      />
      <SingleImageShow v-if="show.singleImage" :src="image1" />
      <Loading v-if="show.loading" />
      <OverlapImageShow
        v-if="show.overlap"
        :brightness="brightness"
        :contrast="contrast"
        :image1="image1"
        :image2="image2"
        :show-mask="showMask"
        @on-origin-scale-change="(value: number) => (originScale = value)"
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
</style>
