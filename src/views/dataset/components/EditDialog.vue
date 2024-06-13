<script setup lang="ts">
import DatasetAPI from "@/api/dataset";
import { Dataset } from "@/api/dataset/model";

defineOptions({
  name: "EditDialog",
  inheritAttrs: false,
});

// 弹框开关
const dialogVisible = ref<boolean>(false);

const title = ref<string>();
const formData = reactive<Dataset>({
  id: 0,
  parentId: 0,
  type: "",
  name: "",
  description: "",
  path: "",
  size: "",
  total: 0,
  children: [],
});

const formRef = ref(ElForm);

// 数据集大小单位
const select = ref("");
const sizeNumber = ref(0);

// 数据集大小校验规则
function validatorSize(rule: any, value: any, callback: any) {
  if (select.value !== "" && sizeNumber.value !== 0) {
    callback();
  } else {
    callback(new Error("数据集大小不能为空"));
  }
}

// 表单校验规则
const rules = reactive({
  name: [{ required: true, message: "请输入数据集名称", trigger: "blur" }],
  type: [{ required: true, message: "请输入数据集类别", trigger: "blur" }],
  description: [
    { required: true, message: "请输入数据集描述", trigger: "blur" },
  ],
  size: [{ validator: validatorSize, trigger: "blur" }],
  path: [{ required: true, message: "请输入数据集存储位置", trigger: "blur" }],
  total: [{ required: true, message: "请输入数据集图片数量", trigger: "blur" }],
});
// 编辑/新增
const type = ref<string>();

// 打开编辑/新增对话框
function open(operation: String, row: Dataset) {
  if (operation === "编辑") {
    Object.assign(formData, row);
    // 获取数据集单位 MB | DB
    const size = formData.size;
    select.value = size.slice(-2);
    sizeNumber.value = Number(size.substring(0, size.length - 2));
    type.value = "编辑";
  } else {
    type.value = "新增";
    // 设置新增的数据集的父 id
    formData.parentId === row.id;
  }
  title.value =
    type.value === "编辑" ? "编辑数据集基本信息" : "新增数据集基本信息";
  dialogVisible.value = true;
}

defineExpose({
  open,
});

function closeDialog() {
  Object.assign(formData, {
    id: 0,
    parentId: 0,
    type: "",
    name: "",
    description: "",
    path: "",
    size: "",
    total: 0,
    children: [],
  });
  dialogVisible.value = false;
  select.value = "";
  sizeNumber.value = 0;
}

// 编辑
const emit = defineEmits(["on-update", "on-add"]);
function onUpdated() {
  formData.size = sizeNumber.value + select.value;
  DatasetAPI.update(formData.id, formData);
  closeDialog();
  emit("on-update");
  ElMessage.success("修改成功");
}

// 新增
function onAdd() {
  formData.size = sizeNumber.value + select.value;
  DatasetAPI.add(formData);
  closeDialog();
  emit("on-add");
  ElMessage.success("添加成功");
}

async function submitForm() {
  // 保证全部表单校验通过再发请求
  await formRef.value.validate();

  if (type.value === "编辑") {
    onUpdated();
  } else {
    onAdd();
  }
}
</script>

<template>
  <el-dialog
    v-model="dialogVisible"
    :title="title"
    destroy-on-close
    append-to-body
    @close="closeDialog"
  >
    <el-form ref="formRef" :model="formData" :rules="rules">
      <el-form-item label="数据集名称" prop="name">
        <el-input v-model="formData.name" clearable />
      </el-form-item>
      <el-form-item label="数据集类别" prop="type">
        <el-input v-model="formData.type" clearable />
      </el-form-item>
      <el-form-item label="数据集描述" prop="description">
        <el-input v-model="formData.description" clearable />
      </el-form-item>
      <el-form-item label="&nbsp;&nbsp;数据集大小" prop="size">
        <el-input-number
          v-model="sizeNumber"
          :precision="2"
          :step="0.01"
          :max="1024"
        />
        <el-select v-model="select" placeholder="选择单位" style="width: 115px">
          <el-option label="MB" value="MB" />
          <el-option label="GB" value="GB" />
        </el-select>
      </el-form-item>
      <el-form-item label="总图片数量" prop="total">
        <el-input-number v-model="formData.total" :min="1" />
      </el-form-item>
      <el-form-item label="存 储 位 置" prop="path">
        <el-input v-model="formData.path" clearable />
      </el-form-item>
    </el-form>
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="submitForm">确定</el-button>
        <el-button @click="closeDialog">取消</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped lang="scss"></style>
