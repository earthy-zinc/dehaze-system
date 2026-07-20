<!-- 字典数据 -->
<template>
  <div class="app-container">
    <div class="search-container">
      <!-- 搜索表单 -->
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="字典名称"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><el-icon><Search /></el-icon>搜索</el-button
          >
          <el-button @click="resetQuery"> <el-icon><Refresh /></el-icon>重置</el-button>
        </el-form-item>
      </el-form>
    </div>
    <el-card shadow="never">
      <template #header>
        <el-button
          v-hasPerm="['sys:dict:data:add']"
          type="success"
          @click="openDialog()"
          ><el-icon><Plus /></el-icon>新增</el-button
        >
        <el-button
          v-hasPerm="['sys:dict:data:delete']"
          :disabled="ids.length === 0"
          type="danger"
          @click="handleDelete()"
          ><el-icon><Delete /></el-icon>删除</el-button
        >
      </template>

      <!-- 数据表格 -->
      <el-table
        v-loading="loading"
        :data="dictList"
        border
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="50" />
        <el-table-column label="字典名称" prop="name" />
        <el-table-column label="字典值" prop="value" />
        <el-table-column align="center" label="状态">
          <template #default="scope">
            <el-tag v-if="scope.row.status === 1" type="success">启用</el-tag>
            <el-tag v-else type="info">禁用</el-tag>
          </template>
        </el-table-column>
        <el-table-column align="center" label="排序" prop="sort" width="80" />
        <el-table-column label="备注" prop="remark" width="150" />
        <el-table-column label="创建时间" prop="createTime" width="180" />
        <el-table-column align="center" fixed="right" label="操作">
          <template #default="scope">
            <el-button
              v-hasPerm="['sys:dict:data:edit']"
              link
              type="primary"
              @click="openDialog(scope.row.id)"
              ><el-icon><Edit /></el-icon>编辑</el-button
            >
            <el-button
              v-hasPerm="['sys:dict:data:delete']"
              link
              type="primary"
              @click.stop="handleDelete(scope.row)"
              ><el-icon><Delete /></el-icon>删除</el-button
            >
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>

    <!-- 表单弹窗 -->
    <el-dialog
      v-model="dialog.visible"
      :title="dialog.title"
      width="500px"
      @close="closeDialog"
    >
      <el-form
        ref="dataFormRef"
        :model="formData"
        :rules="rules"
        label-width="100px"
      >
        <el-form-item label="字典类型">{{ typeName }}</el-form-item>
        <el-form-item label="字典名称" prop="name">
          <el-input v-model="formData.name" placeholder="请输入字典名称" />
        </el-form-item>
        <el-form-item label="字典值" prop="value">
          <el-input v-model="formData.value" placeholder="字典值" />
        </el-form-item>
        <el-form-item label="排序" prop="sort">
          <el-input-number
            v-model="formData.sort"
            :min="0"
            controls-position="right"
          />
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-radio-group v-model="formData.status">
            <el-radio :label="1">正常</el-radio>
            <el-radio :label="0">停用</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="是否默认" prop="defaulted">
          <el-switch
            v-model="formData.defaulted"
            :active-value="1"
            :inactive-value="0"
          />
        </el-form-item>
        <el-form-item label="备注" prop="remark">
          <el-input v-model="formData.remark" type="textarea" />
        </el-form-item>
      </el-form>
      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleSubmit">确 定</el-button>
          <el-button @click="closeDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
defineOptions({
  name: "DictData",
  inheritAttrs: false,
});

import { DictAPI, DictForm, DictPageVO, DictQuery } from "dehaze-sdk-js";
import { Delete, Edit, Plus, Refresh, Search } from "@element-plus/icons-vue";

const props = defineProps({
  typeCode: {
    type: String,
    default: () => {
      return "";
    },
  },
  typeName: {
    type: String,
    default: () => {
      return "";
    },
  },
});

watch(
  () => props.typeCode,
  (newVal: string) => {
    queryParams.typeCode = newVal;
    formData.typeCode = newVal;
    resetQuery();
  }
);

const queryFormRef = ref(ElForm);
const dataFormRef = ref(ElForm);

const loading = ref(false);
const ids = ref<number[]>([]);
const total = ref(0);

const queryParams = reactive<DictQuery>({
  pageNum: 1,
  pageSize: 10,
  typeCode: props.typeCode,
});

const dictList = ref<DictPageVO[]>();

const dialog = reactive({
  title: "",
  visible: false,
});

const formData = reactive<DictForm>({
  status: 1,
  defaulted: 0,
  typeCode: props.typeCode,
  sort: 1,
});

const rules = reactive({
  name: [{ required: true, message: "请输入字典名称", trigger: "blur" }],
  value: [{ required: true, message: "请输入字典值", trigger: "blur" }],
  sort: [
    { required: true, message: "请输入排序", trigger: "blur", type: "number" },
  ],
});

/** 查询 */
function handleQuery() {
  if (queryParams.typeCode) {
    loading.value = true;
    DictAPI.getDictPage(queryParams)
      .then((data) => {
        dictList.value = data.list;
        total.value = data.total;
      })
      .finally(() => (loading.value = false));
  }
}

/** 重置查询 */
function resetQuery() {
  queryFormRef.value.resetFields();
  queryParams.pageNum = 1;
  handleQuery();
}

/**
 * 行checkbox change事件
 *
 * @param selection
 */
function handleSelectionChange(selection: any) {
  ids.value = selection.map((item: any) => item.id);
}

/**
 * 打开字典表单弹窗
 *
 * @param dictId 字典ID
 */
function openDialog(dictId?: number) {
  dialog.visible = true;
  if (dictId) {
    dialog.title = "修改字典";
    DictAPI.getDictFormData(dictId).then((data) => {
      Object.assign(formData, data);
    });
  } else {
    dialog.title = "新增字典";
  }
}

/** 字典表单提交 */
function handleSubmit() {
  dataFormRef.value.validate((isValid: boolean) => {
    if (isValid) {
      loading.value = false;
      const dictId = formData.id;
      if (dictId) {
        DictAPI.updateDict(dictId, formData)
          .then(() => {
            ElMessage.success("修改成功");
            closeDialog();
            resetQuery();
          })
          .finally(() => (loading.value = false));
      } else {
        DictAPI.addDict(formData)
          .then(() => {
            ElMessage.success("新增成功");
            closeDialog();
            resetQuery();
          })
          .finally(() => (loading.value = false));
      }
    }
  });
}

/** 关闭弹窗 */
function closeDialog() {
  dialog.visible = false;
  resetForm();
}

/** 重置表单 */
function resetForm() {
  dataFormRef.value.resetFields();
  dataFormRef.value.clearValidate();

  formData.id = undefined;
  formData.status = 1;
  formData.defaulted = 0;
  formData.sort = 1;
  formData.typeCode = props.typeCode;
}

/** 删除字典 */
function handleDelete(row?: any) {
  const dictIds = (row ? [row.id] : ids.value).join(",");
  if (!dictIds) {
    ElMessage.warning("请勾选删除项");
    return;
  }

  const confirmMsg = row
    ? `确认删除字典数据「${row.name}」吗？删除后不可恢复。`
    : "确认删除选中的字典数据吗？删除后不可恢复。";
  ElMessageBox.confirm(confirmMsg, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  }).then(() => {
    DictAPI.deleteDictByIds(dictIds).then(() => {
      ElMessage.success("删除成功");
      resetQuery();
    });
  });
}

onMounted(() => {
  handleQuery();
});
</script>
