<!-- 多图上传组件 -->
<template>
  <el-upload
    v-model:file-list="fileList"
    :before-upload="handleBeforeUpload"
    :before-remove="handleBeforeRemove"
    :http-request="handleUpload"
    :limit="props.limit"
    :on-preview="previewImg"
    :on-remove="handleRemove"
    list-type="picture-card"
  >
    <i-ep-plus />
  </el-upload>

  <el-dialog v-model="dialogVisible">
    <img :src="previewImgUrl" alt="Preview Image" w-full />
  </el-dialog>
</template>

<script lang="ts" setup>
import { computeFileMd5 } from "@/utils";
import { FileAPI } from "dehaze-sdk-js";
import {
  UploadProps,
  UploadRawFile,
  UploadRequestOptions,
  UploadUserFile,
} from "element-plus";

type UploadFileItem = UploadUserFile & { fileId?: number };

const emit = defineEmits(["update:modelValue"]);

const props = defineProps({
  /**
   * 文件路径集合
   */
  modelValue: {
    type: Array<string>,
    default: () => [],
  },
  /**
   * 文件上传数量限制
   */
  limit: {
    type: Number,
    default: 10,
  },
});

const previewImgUrl = ref("");
const dialogVisible = ref(false);

const fileList = ref([] as UploadFileItem[]);
watch(
  () => props.modelValue,
  (newVal: string[]) => {
    const filePaths = fileList.value.map((file) => file.url);
    // 监听modelValue文件集合值未变化时，跳过赋值
    if (
      filePaths.length > 0 &&
      filePaths.length === newVal.length &&
      filePaths.every((x) => newVal.some((y) => y === x)) &&
      newVal.every((y) => filePaths.some((x) => x === y))
    ) {
      return;
    }

    fileList.value = newVal.map((filePath) => {
      return { url: filePath } as UploadFileItem;
    });
  },
  { immediate: true }
);

/**
 * 自定义图片上传（支持 MD5 秒传）
 *
 * @param params
 */
async function handleUpload(options: UploadRequestOptions): Promise<any> {
  // 计算文件 MD5，检查是否已存在（秒传）
  const md5 = await computeFileMd5(options.file);
  const existing = await FileAPI.uploadCheck(md5);
  const data = existing ?? (await FileAPI.upload(options.file));

  // 上传成功需手动替换文件路径为远程URL，否则图片地址为预览地址 blob:http://
  const fileIndex = fileList.value.findIndex(
    (file) => file.uid == (options.file as any).uid
  );

  fileList.value.splice(fileIndex, 1, {
    name: data.name,
    url: data.url,
    fileId: data.id,
  } as UploadFileItem);

  emit(
    "update:modelValue",
    fileList.value.map((file) => file.url)
  );
}

/**
 * 删除前确认弹窗，确认后调用 deleteById 删除文件
 */
const handleBeforeRemove: UploadProps["beforeRemove"] = (uploadFile) => {
  return ElMessageBox.confirm("确认删除该文件吗？删除后不可恢复。", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(async () => {
      const item = fileList.value.find((f) => f.uid === uploadFile.uid);
      if (item?.fileId) {
        await FileAPI.deleteById(item.fileId);
      }
      return true;
    })
    .catch(() => false);
};

/**
 * 删除图片后同步 modelValue
 */
function handleRemove() {
  emit(
    "update:modelValue",
    fileList.value.map((file) => file.url)
  );
}

/**
 * 限制用户上传文件的格式和大小
 */
function handleBeforeUpload(file: UploadRawFile) {
  if (file.size > 100 * 1048 * 1048) {
    ElMessage.warning("上传图片不能大于100M");
    return false;
  }
  return true;
}

/**
 * 预览图片
 */
const previewImg: UploadProps["onPreview"] = (uploadFile) => {
  previewImgUrl.value = uploadFile.url!;
  dialogVisible.value = true;
};
</script>
