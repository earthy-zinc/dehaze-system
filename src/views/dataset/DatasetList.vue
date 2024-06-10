<script lang="ts" setup>
import DatasetAPI from "@/api/dataset";
import { Dataset, DatasetQuery } from "@/api/dataset/model";

defineOptions({
  name: "DatasetList",
  inheritAttrs: false,
});

const queryFormRef = ref(ElForm);
const loading = ref(false);
const queryParams = reactive<DatasetQuery>({});
const datasetList = ref<Dataset[]>([]);
const selectedDatasetId = ref<number>();
const selectedColumns = ref([
  "name",
  "type",
  "description",
  "size",
  "total",
  "path",
  "visible",
]);
const columns = [
  { label: "名称", value: "name" },
  { label: "类型", value: "type" },
  { label: "描述", value: "description" },
  { label: "大小", value: "size" },
  { label: "图片数量", value: "total" },
  { label: "存储位置", value: "path" },
  { label: "状态", value: "visible" },
];

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

function onRowClick(row: Dataset) {
  selectedDatasetId.value = row.id;
}

const router = useRouter();

function handleShow(row: Dataset) {
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

function handleSettings() {}

onMounted(() => {
  handleQuery();
});
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="数据集名称"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <template #icon>
              <i-ep-search />
            </template>
            搜索
          </el-button>
          <el-button @click="resetQuery">
            <template #icon>
              <i-ep-refresh />
            </template>
            重置
          </el-button>
          <el-dropdown>
            <el-button>
              <template #icon>
                <i-ep-setting />
              </template>
              设置
            </el-button>
            <template #dropdown>
              <el-checkbox-group v-model="selectedColumns">
                <el-checkbox
                  v-for="column in columns"
                  :key="column.value"
                  :label="column.label"
                  :value="column.value"
                />
              </el-checkbox-group>
            </template>
          </el-dropdown>
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <el-table
        v-loading="loading"
        :data="datasetList"
        :tree-props="{
          children: 'children',
          hasChildren: 'hasChildren',
        }"
        default-expand-all
        highlight-current-row
        row-key="id"
        @row-click="onRowClick"
      >
        <el-table-column
          v-if="selectedColumns.includes('name')"
          label="名称"
          prop="name"
        />
        <el-table-column
          v-if="selectedColumns.includes('type')"
          label="类型"
          prop="type"
        />
        <el-table-column label="描述" prop="description" />

        <el-table-column
          v-if="selectedColumns.includes('size')"
          label="大小"
          prop="size"
        />
        <el-table-column
          v-if="selectedColumns.includes('total')"
          label="图片数量"
          prop="total"
        />
        <el-table-column
          v-if="selectedColumns.includes('path')"
          label="存储位置"
          prop="path"
        />
        <el-table-column
          v-if="selectedColumns.includes('visible')"
          align="center"
          label="状态"
          width="80"
        >
          <template #default="scope">
            <el-tag v-if="scope.row.visible === 1" type="success">显示</el-tag>
            <el-tag v-else type="info">隐藏</el-tag>
          </template>
        </el-table-column>
        <el-table-column align="center" fixed="right" label="操作" width="260">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="handleShow(scope.row)"
            >
              <svg-icon icon-class="eye-open" />
              查看
            </el-button>

            <el-button link size="small" type="primary">
              <i-ep-plus />
              新增
            </el-button>

            <el-button link size="small" type="primary">
              <i-ep-edit />
              编辑
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click="handleDelete(scope.row.id)"
            >
              <i-ep-delete />
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style lang="scss" scoped></style>
