<script setup lang="ts">
import DatasetAPI from "@/api/dataset";
import { DatasetQuery, DatasetVO } from "@/api/dataset/model";

defineOptions({
  name: "DatasetList",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const loading = ref(false);
const queryParams = reactive<DatasetQuery>({});
const datasetList = ref<DatasetVO[]>([]);
const selectedDatasetId = ref<number>();
function handleQuery() {
  loading.value = true;
  DatasetAPI.getList(queryParams)
    .then((data) => {
      datasetList.value = data;
    })
    .then(() => {
      loading.value = false;
    });
}
function resetQuery() {
  queryFormRef.value.resetFields();
  handleQuery();
}

function onRowClick(row: DatasetVO) {
  selectedDatasetId.value = row.id;
}

const router = useRouter();
function handleShow(row: DatasetVO) {
  selectedDatasetId.value = row.id;
  router.push(`/dataset/${selectedDatasetId.value}`);
}

function handleDelete(datasetId: number) {
  if (!datasetId) {
    ElMessage.warning("请勾选删除项");
    return false;
  }
  ElMessageBox.confirm(
    "确认删除已选中的数据集？其中的图片也将一并删除！",
    "警告",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
      ElMessage.success("删除成功");
    })
    .catch(() => ElMessage.info("已取消删除"));
}
onMounted(() => {
  handleQuery();
});
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :model="queryParams" :inline="true">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            placeholder="数据集名称"
            clearable
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><template #icon><i-ep-search /></template>搜索</el-button
          >
          <el-button @click="resetQuery">
            <template #icon><i-ep-refresh /></template>
            重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card shadow="never" class="table-container">
      <el-table
        v-loading="loading"
        :data="datasetList"
        highlight-current-row
        row-key="id"
        default-expand-all
        @row-click="onRowClick"
        :tree-props="{
          children: 'children',
          hasChildren: 'hasChildren',
        }"
      >
        <el-table-column label="名称" prop="name" />
        <el-table-column label="类型" prop="type" />
        <el-table-column label="描述" prop="description" />

        <el-table-column label="大小" prop="size" />
        <el-table-column label="图片数量" prop="total" />
        <el-table-column label="存储位置" prop="path" />
        <el-table-column label="状态" align="center" width="80">
          <template #default="scope">
            <el-tag v-if="scope.row.visible === 1" type="success">显示</el-tag>
            <el-tag v-else type="info">隐藏</el-tag>
          </template>
        </el-table-column>
        <el-table-column fixed="right" align="center" label="操作" width="260">
          <template #default="scope">
            <el-button
              type="primary"
              link
              size="small"
              @click="handleShow(scope.row)"
            >
              <svg-icon icon-class="eye-open" />查看
            </el-button>

            <el-button type="primary" link size="small">
              <i-ep-plus />新增
            </el-button>

            <el-button type="primary" link size="small">
              <i-ep-edit />编辑
            </el-button>
            <el-button
              type="primary"
              link
              size="small"
              @click="handleDelete(scope.row.id)"
              ><i-ep-delete />
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped lang="scss"></style>
