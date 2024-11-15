<script lang="ts" setup></script>

<template>
  <el-dialog v-model="dialogTableVisible" title="数据集图片选择" width="800">
    <el-form ref="queryFormRef" :inline="true" :model="queryParams">
      <el-form-item label="关键字" prop="keywords">
        <el-input
          v-model="queryParams.keywords"
          clearable
          placeholder="图片名称"
          @keyup.enter="handleQuery"
        />
      </el-form-item>
    </el-form>
  </el-dialog>
</template>

<style lang="scss" scoped></style>
