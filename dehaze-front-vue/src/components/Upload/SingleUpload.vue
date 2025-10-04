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
  </el-upload>
</template>

<script lang="ts" setup>
import { UploadRawFile, UploadRequestOptions } from "element-plus";
import FileAPI from "@/api/file";
import { useImageShowStore } from "@/store/modules/imageShow";

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

/**
 * 自定义图片上传
 *
 * @param options
 */
async function uploadFile(options: UploadRequestOptions): Promise<any> {
  const data = await FileAPI.upload(options.file, imageShowStore.modelId);
  imgUrl.value = data.url;
  emit("onChange", data.url);
}

/**
 * 限制用户上传文件的格式和大小
 */
function handleBeforeUpload(file: UploadRawFile) {
  if (file.size > 10 * 1048 * 1048) {
    ElMessage.warning("上传图片不能大于10M");
    return false;
  }
  return true;
}
</script>

<style lang="scss" scoped>
.single-uploader {
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
}

.single-uploader__icon {
  font-size: 30px;
  color: #8c939d;
  text-align: center;
}
</style>
