<script lang="ts" setup>
import { ImageItem, ImageItemQuery } from "@/api/dataset/model";
import { ViewCard } from "../Waterfall/types";

defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
});

const datasetId = ref<number>(1);
const totalPages = ref<number>(1);
const queryParams = reactive<ImageItemQuery>({ pageNum: 1, pageSize: 50 });

let images = ref<ViewCard[]>([]);
const imageData = ref<ImageItem[]>([]);
type ImageType = { id: number; type: string; enabled: boolean };
const imageTypes = ref<ImageType[]>([
  { id: 0, type: "清晰图像", enabled: true },
  { id: 1, type: "有雾图像", enabled: false },
]);
const route = useRoute();
const { width } = useWindowSize();
const dialogVisible = ref(false);
</script>

<template>
  <el-dialog v-model="dialogVisible" title="数据集图片选择" width="800">
    <el-form ref="queryFormRef" :inline="true" :model="queryParams">
      <el-form-item label="关键字" prop="keywords">
        <el-input
          v-model="queryParams.keywords"
          clearable
          placeholder="图片名称"
        />
      </el-form-item>
    </el-form>

    <LongitudinalWaterfall :list="images" />
    <el-divider v-if="queryParams.pageNum < totalPages"
      >正在加载，请稍后</el-divider
    >
  </el-dialog>
</template>

<style lang="scss" scoped></style>
