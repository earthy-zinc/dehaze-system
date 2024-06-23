<script setup lang="ts">
import { UploadFile, UploadUserFile } from "element-plus";

defineOptions({
  name: "ImageUploadButton",
});

const emit = defineEmits(["onUpload", "onTakePhoto", "onReset"]);

function handleUploadChange(uploadFile: UploadFile) {
  emit("onUpload", uploadFile.raw);
}

function handleUploadExceed(files: File[], uploadFiles: UploadUserFile[]) {
  uploadFiles.shift();
  uploadFiles.push(files[0]);
  handleUploadChange(uploadFiles[0] as UploadFile);
}
</script>

<template>
  <div class="flex justify-evenly m-4">
    <el-dropdown split-button type="primary">
      <el-upload
        :show-file-list="false"
        :limit="1"
        action="#"
        :auto-upload="false"
        :on-change="handleUploadChange"
        :on-exceed="handleUploadExceed"
        accept="image/gif, image/jpeg, image/jpg, image/png, image/svg"
      >
        本地上传
      </el-upload>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item @click="emit('onTakePhoto')">
            拍照上传
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    <el-button id="algorithm-reset-button" @click="emit('onReset')"
      >清除结果</el-button
    >
  </div>
</template>

<style lang="scss" scoped></style>
