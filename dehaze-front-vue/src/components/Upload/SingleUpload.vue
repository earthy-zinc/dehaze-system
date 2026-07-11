<template>
  <!-- 上传组件 -->
  <el-upload
    v-model="imgUrl"
    :before-upload="handleBeforeUpload"
    :http-request="uploadFile"
    :show-file-list="false"
    class="single-uploader"
    list-type="picture-card"
  >
    <img
      v-if="imgUrl"
      :src="imgUrl"
      alt="图片解析失败"
      class="single-uploader__image"
    />
    <div v-else class="flex flex-col justify-center items-center">
      <el-icon class="single-uploader__icon">
        <i-ep-plus />
      </el-icon>
      <div class="mt-5 font-size-4">{{ tooltip }}</div>
    </div>
    <div
      v-if="imgUrl"
      class="single-uploader__delete"
      @click.stop="handleDelete"
    >
      <el-icon><i-ep-close /></el-icon>
    </div>
  </el-upload>
</template>

<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import { computeFileMd5 } from "@/utils";
import { FileAPI } from "dehaze-sdk-js";
import { UploadRawFile, UploadRequestOptions } from "element-plus";

const props = defineProps({
  modelValue: {
    type: String,
    default: "",
  },
  tooltip: {
    type: String,
    default: "上传图片",
  },
});

const emit = defineEmits(["update:modelValue", "onChange"]);
const imgUrl = useVModel(props, "modelValue", emit);
const imageShowStore = useImageShowStore();
const fileId = ref<number | null>(null);

/**
 * 自定义图片上传（支持 MD5 秒传）
 *
 * @param options
 */
async function uploadFile(options: UploadRequestOptions): Promise<any> {
  // 计算文件 MD5，检查是否已存在（秒传）
  const md5 = await computeFileMd5(options.file);
  const existing = await FileAPI.uploadCheck(md5);
  if (existing) {
    imgUrl.value = existing.url;
    fileId.value = existing.id;
    emit("onChange", existing.url);
    return;
  }
  const data = await FileAPI.upload(options.file, imageShowStore.modelId);
  imgUrl.value = data.url;
  fileId.value = data.id;
  emit("onChange", data.url);
}

/**
 * 删除图片（带确认弹窗）
 */
function handleDelete() {
  ElMessageBox.confirm("确认删除该文件吗？删除后不可恢复。", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(async () => {
      if (fileId.value) {
        await FileAPI.deleteById(fileId.value);
      }
      imgUrl.value = "";
      fileId.value = null;
      emit("onChange", "");
    })
    .catch(() => {});
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

defineExpose({ handleBeforeUpload, uploadFile, handleDelete });
</script>

<style lang="scss" scoped>
.single-uploader {
  position: relative;
  overflow: hidden;
  cursor: pointer;
  border: 1px var(--el-border-color) solid;
  border-radius: 6px;

  &:hover {
    border-color: var(--el-color-primary);
  }

  &__image {
    display: block;
  }

  &__delete {
    position: absolute;
    top: 4px;
    right: 4px;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    color: #fff;
    cursor: pointer;
    background-color: rgba(0, 0, 0, 0.5);
    border-radius: 50%;

    &:hover {
      background-color: rgba(0, 0, 0, 0.7);
    }
  }
}

.single-uploader__icon {
  font-size: 30px;
  color: #8c939d;
  text-align: center;
}
</style>
