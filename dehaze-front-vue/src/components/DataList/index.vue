<script lang="ts" setup>
import EditDialog from "@/components/DataList/EditDialog/index.vue";
import { DatasetAPI, DatasetQuery, DatasetVO } from "dehaze-sdk-js";
import {
  Delete,
  Edit,
  Plus,
  Refresh,
  Search,
  Setting,
} from "@element-plus/icons-vue";
import ImportExportToolbar from "@/components/ImportExportToolbar/index.vue";

const queryFormRef = ref(ElForm);
const loading = ref(false);
const queryParams = reactive<DatasetQuery>({
  pageNum: 1,
  pageSize: 20,
});
const list = ref<DatasetVO[]>([]);
const total = ref(0);
const selectedId = ref<number>();
const ids = ref<number[]>([]);
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
      list.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value.resetFields();
  queryParams.pageNum = 1;
  handleQuery();
}

function handleSizeChange(size: number) {
  queryParams.pageSize = size;
  queryParams.pageNum = 1;
  handleQuery();
}

function handleCurrentChange(page: number) {
  queryParams.pageNum = page;
  handleQuery();
}

function loadChildren(
  tree: any,
  treeNode: unknown,
  resolve: (children: any[]) => void
) {
  if (tree.hasChildren === false) {
    resolve([]);
    return;
  }
  DatasetAPI.getChildren(tree.id)
    .then((children) => {
      resolve(children);
    })
    .catch(() => {
      resolve([]);
    });
}

const router = useRouter();

function onRowClick(row: DatasetVO) {
  selectedId.value = row.id;
}

function handleShow(row: DatasetVO) {
  selectedId.value = row.id;
  router.push(`/dataset/${selectedId.value}`);
}

function handleDelete(row?: DatasetVO) {
  const tip = row
    ? `确认删除数据集「${row.name}」？其中的图片也将一并删除！`
    : `确认删除选中的 ${ids.value.length} 个数据集？其中的图片也将一并删除！`;
  if (!row && ids.value.length === 0) {
    ElMessage.warning("请勾选删除项");
    return;
  }
  ElMessageBox.confirm(tip, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      if (row) {
        DatasetAPI.deleteById(row.id);
      } else {
        DatasetAPI.batchDelete({ ids: ids.value });
      }
      ElMessage.success("删除成功");
      handleQuery();
    })
    .catch(() => ElMessage.info("已取消删除"));
}

/** 行复选框选中记录选中ID集合 */
function handleSelectionChange(selection: DatasetVO[]) {
  ids.value = selection.map((item) => item.id);
}

const dialogRef = ref();

function openDialog(type: string, dataset: DatasetVO | object) {
  dialogRef.value.open(type, dataset);
}

onMounted(() => {
  handleQuery();
});
</script>

<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keyword">
          <el-input
            v-model="queryParams.keyword"
            placeholder="数据集名称"
            clearable
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <template #icon>
              <el-icon><Search /></el-icon>
            </template>
            搜索
          </el-button>
          <el-button @click="resetQuery">
            <template #icon>
              <el-icon><Refresh /></el-icon>
            </template>
            重置
          </el-button>
          <el-dropdown class="setting-button">
            <el-button>
              <template #icon>
                <el-icon><Setting /></el-icon>
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
      <div class="toolbar">
        <el-button type="success" @click="openDialog('新增', {})">
          <el-icon><Plus /></el-icon>
          新增
        </el-button>
        <el-button
          type="danger"
          :disabled="ids.length === 0"
          @click="handleDelete()"
        >
          <el-icon><Delete /></el-icon>
          删除
        </el-button>
        <ImportExportToolbar
          :importable="false"
          module="dataset"
          :query-params="queryParams"
          @import-complete="handleQuery"
        />
      </div>

      <el-table
        v-loading="loading"
        :data="list"
        :tree-props="{
          children: 'children',
          hasChildren: 'hasChildren',
        }"
        lazy
        :load="loadChildren"
        highlight-current-row
        row-key="id"
        @row-click="onRowClick"
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
        <el-table-column
          v-if="selectedColumns.includes('name')"
          label="名称"
          prop="name"
          width="200"
        />
        <el-table-column
          v-if="selectedColumns.includes('type')"
          label="类型"
          prop="type"
          width="100"
        />
        <el-table-column
          v-if="selectedColumns.includes('description')"
          label="描述"
          min-width="500"
          prop="description"
        />
        <el-table-column
          v-if="selectedColumns.includes('size')"
          label="大小"
          prop="size"
          width="120"
        />
        <el-table-column
          v-if="selectedColumns.includes('total')"
          label="图片数量"
          prop="total"
          width="100"
        />
        <el-table-column
          v-if="selectedColumns.includes('path')"
          label="存储位置"
          prop="path"
          width="300"
        />
        <el-table-column
          v-if="selectedColumns.includes('status')"
          align="center"
          label="状态"
          min-width="65"
        >
          <template #default="scope">
            <el-tag v-if="scope.row.status === 1" type="success">启用 </el-tag>
            <el-tag v-else-if="scope.row.status === 0" type="info"
              >禁用
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column align="center" fixed="right" label="操作" width="260">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="handleShow(scope.row as DatasetVO)"
            >
              <svg-icon icon-class="eye-open" />
              查看
            </el-button>

            <el-button
              link
              size="small"
              type="primary"
              @click="openDialog('新增', scope.row as DatasetVO)"
            >
              <el-icon><Plus /></el-icon>
              新增
            </el-button>

            <el-button
              link
              size="small"
              type="primary"
              @click="openDialog('编辑', scope.row as DatasetVO)"
            >
              <el-icon><Edit /></el-icon>
              编辑
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click="handleDelete(scope.row as DatasetVO)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="queryParams.pageNum"
        v-model:page-size="queryParams.pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="total"
        layout="total, sizes, prev, pager, next, jumper"
        background
        class="pagination-container"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </el-card>

    <EditDialog
      ref="dialogRef"
      @on-update="handleQuery"
      @on-add="handleQuery"
    />
  </div>
</template>

<style lang="scss" scoped>
.toolbar {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}

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

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
</style>

<style lang="scss">
.el-divider--horizontal {
  margin: 5px 0;
}
</style>
