<script setup lang="ts">
import DatasetAPI from "@/api/dataset";
import { DatasetVO } from "@/api/dataset/model";

defineOptions({
  name: "EditDialog",
  inheritAttrs: false,
});

const props = defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
  datasetId: {
    type: Number,
    require: false,
  },
  /**
   * 是新增还是编辑
   */
  type: {
    type: String,
    require: true,
  },
});

const title = ref<string>();

const formData = reactive<DatasetVO>({
  id: 0,
  parentId: 0,
  type: "",
  name: "",
  path: "",
  size: "",
  total: 0,
  children: [],
});

function closeDialog() {}

function submitForm() {}

function resetForm() {}
onMounted(() => {
  if (props.type === "编辑") {
    title.value = "编辑数据集基本信息";
    DatasetAPI.getList().then((data) => {});
  } else {
    title.value = "新增数据集基本信息";
  }
});
</script>

<template>
  <el-dialog
    :title="title"
    destroy-on-close
    append-to-body
    @close="closeDialog"
  >
    <el-form>
      <el-form-item label="数据集名称">
        <el-input />
      </el-form-item>
      <el-form-item label="所属类别">
        <el-input />
      </el-form-item>
    </el-form>
    <template #footer>
      <div class="dialog-footer">
        <el-button>确定</el-button>
        <el-button>取消</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped lang="scss"></style>
