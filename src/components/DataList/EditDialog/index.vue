<script setup lang="ts">
import AlgorithmAPI from "@/api/algorithm";
import DatasetAPI from "@/api/dataset";
import { Algorithm } from "@/api/algorithm/model";
import { Dataset } from "@/api/dataset/model";

defineOptions({
  name: "EditDialog",
  inheritAttrs: false,
});

const props = defineProps<{
  isDatasetList: boolean;
}>();

const { isDatasetList } = props;
const NAME_CONSTANT = isDatasetList ? "数据集" : "模型";
const API = isDatasetList ? DatasetAPI : AlgorithmAPI;

// 弹框开关
const dialogVisible = ref<boolean>(false);

const title = ref<string>();

type FormData = {
  id: number;
  parentId: number;
  type: string;
  name: string;
  description: string;
  path: string;
  size: string;
  status: number;
  importPath?: string;
  total?: number;
  children: [];
};

const formData = reactive<FormData>({
  id: 0,
  parentId: 0,
  type: "",
  name: "",
  description: "",
  importPath: "",
  total: 0,
  path: "",
  size: "",
  children: [],
  status: 0,
});

const formRef = ref(ElForm);

// 数据集/模型大小单位
const select = ref("");
const sizeNumber = ref(0);

// 数据集/模型大小校验规则
function validatorSize(rule: any, value: any, callback: any) {
  if (select.value !== "" && sizeNumber.value !== 0) {
    callback();
  } else {
    callback(new Error(`${NAME_CONSTANT}大小不能为空`));
  }
}

// 表单校验规则
const rules = reactive({
  name: [
    { required: true, message: `请输入${NAME_CONSTANT}名称`, trigger: "blur" },
  ],
  type: [
    { required: true, message: `请输入${NAME_CONSTANT}类别`, trigger: "blur" },
  ],
  description: [
    { required: true, message: `请输入${NAME_CONSTANT}描述`, trigger: "blur" },
  ],
  size: [{ validator: validatorSize, trigger: "blur" }],
  path: [
    {
      required: true,
      message: `请输入${NAME_CONSTANT}存储位置`,
      trigger: "blur",
    },
  ],
  total: [{}],
  importPath: [{}],
});
if (isDatasetList) {
  Object.assign(rules.total, [
    {
      required: true,
      message: `请输入${NAME_CONSTANT}图片数量`,
      trigger: "blur",
    },
  ]);
} else {
  Object.assign(rules.importPath, [
    {
      required: true,
      message: `请输入${NAME_CONSTANT}代码导入路径`,
      trigger: "blur",
    },
  ]);
}
Object.assign(
  rules.total,
  isDatasetList
    ? [
        {
          required: true,
          message: `请输入${NAME_CONSTANT}图片数量`,
          trigger: "blur",
        },
      ]
    : [
        {
          required: true,
          message: `请输入${NAME_CONSTANT}代码导入位置`,
          trigger: "blur",
        },
      ]
);
// 编辑/新增
const type = ref<string>("");

// 打开编辑/新增对话框
function open<T extends Dataset | Algorithm>(operation: String, row: T) {
  if (operation === "编辑") {
    Object.assign(formData, row);
    // 获取数据集/模型大小单位 MB | DB
    const size = formData.size + "";
    select.value = size.slice(-2);
    sizeNumber.value = Number(size.substring(0, size.length - 2));
    type.value = "编辑";
  } else {
    type.value = "新增";
    // 设置新增的数据集的父 id
    formData.parentId === row.id;
  }
  title.value =
    type.value === "编辑"
      ? `编辑${NAME_CONSTANT}基本信息`
      : `新增${NAME_CONSTANT}基本信息`;
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
    status: 0,
  });
  dialogVisible.value = false;
  select.value = "";
  sizeNumber.value = 0;
}

// 编辑
const emit = defineEmits(["on-update", "on-add"]);
function onUpdated() {
  formData.size = sizeNumber.value + select.value;
  if (isDatasetList) {
    // 移除 importPath 属性
    delete formData.importPath;
  } else {
    // 移除 total 属性
    delete formData.total;
  }
  // @ts-ignore
  API.update(formData.id, formData);
  closeDialog();
  emit("on-update");
  ElMessage.success("修改成功");
}

// 新增
function onAdd() {
  formData.size = sizeNumber.value + select.value;
  if (isDatasetList) {
    // 移除 importPath 属性
    delete formData.importPath;
  } else {
    // 移除 total 属性
    delete formData.total;
  }
  // @ts-ignore
  API.add(formData);
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
      <el-form-item :label="`${NAME_CONSTANT}名称`" prop="name">
        <el-input v-model="formData.name" clearable />
      </el-form-item>
      <el-form-item :label="`${NAME_CONSTANT}类别`" prop="type">
        <el-input v-model="formData.type" clearable />
      </el-form-item>
      <el-form-item :label="`${NAME_CONSTANT}描述`" prop="description">
        <el-input v-model="formData.description" clearable />
      </el-form-item>
      <el-form-item :label="`&nbsp;&nbsp;${NAME_CONSTANT}大小`" prop="size">
        <el-input-number v-model="sizeNumber" :precision="2" :step="0.01" />
        <el-select v-model="select" placeholder="选择单位" style="width: 115px">
          <el-option label="MB" value="MB" />
          <el-option label="GB" value="GB" />
        </el-select>
      </el-form-item>
      <el-form-item label="总图片数量" prop="total" v-if="isDatasetList">
        <el-input-number v-model="formData.total" :min="1" />
      </el-form-item>
      <el-form-item label="导 入 路 径" prop="importPath" v-else>
        <el-input v-model="formData.importPath" clearable />
      </el-form-item>
      <el-form-item label="存 储 位 置" prop="path">
        <el-input v-model="formData.path" clearable />
      </el-form-item>
      <el-form-item :label="`&nbsp;&nbsp;${NAME_CONSTANT}状态`">
        <el-switch
          size="large"
          v-model="formData.status"
          inline-prompt
          :active-text="`${isDatasetList ? '显示' : '启用'}`"
          :inactive-text="`${isDatasetList ? '隐藏' : '禁用'}`"
          active-value="1"
          inactive-value="0"
        />
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
