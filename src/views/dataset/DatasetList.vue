<script lang="ts" setup>
import DatasetAPI from "@/api/dataset";
import { Dataset, DatasetQuery } from "@/api/dataset/model";
import EditDialog from "./components/EditDialog.vue";

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
  "status",
]);
const columns = [
  { label: "名称", value: "name" },
  { label: "类型", value: "type" },
  { label: "描述", value: "description" },
  { label: "大小", value: "size" },
  { label: "图片数量", value: "total" },
  { label: "存储位置", value: "path" },
  { label: "状态", value: "status" },
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
      console.log([...datasetId.toString()]);
      DatasetAPI.deleteByIds([...datasetId.toString()]);
      ElMessage.success("删除成功");
    })
    .catch(() => ElMessage.info("已取消删除"));
}

const dialogRef = ref();

function onEdit(type: string, row: Dataset) {
  dialogRef.value.open(type, row);
}

function openDialog(type: string, dataset: Dataset) {
  onEdit(type, dataset);
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
          <el-dropdown class="setting-button">
            <el-button>
              <template #icon>
                <i-ep-setting />
              </template>
              设置
            </el-button>
            <template #dropdown>
              <div class="setting-title">列选项</div>
              <el-divider class="p-0" />
              <el-checkbox-group
                v-model="selectedColumns"
                class="setting-checkbox"
              >
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
          v-if="selectedColumns.includes('status')"
          align="center"
          label="状态"
          width="80"
        >
          <template #default="scope">
            <el-tag v-if="scope.row.status === 1" type="success">显示</el-tag>
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

            <el-button
              link
              size="small"
              type="primary"
              @click="openDialog('新增', scope.row)"
            >
              <i-ep-plus />
              新增
            </el-button>

            <el-button
              link
              size="small"
              type="primary"
              @click="openDialog('编辑', scope.row)"
            >
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

    <!-- 新增/编辑对话框 -->
    <EditDialog
      ref="dialogRef"
      @on-update="handleQuery"
      @on-add="handleQuery"
    />
  </div>
</template>

<style lang="scss" scoped>
.setting-button {
  margin-left: 12px;
}

.setting-title {
  margin-top: 8px;
  margin-bottom: 8px;
  font-size: 16px;
  font-weight: bold;
  text-align: center;
}

.setting-checkbox {
  display: flex;
  flex-direction: column;
  margin: 0 15px;
}
</style>

<style lang="scss">
.el-divider--horizontal {
  margin: 5px 0;
}
</style>
