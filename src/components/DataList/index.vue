<script lang="ts" setup>
import AlgorithmAPI from "@/api/algorithm";
import DatasetAPI from "@/api/dataset";
import { Algorithm, AlgorithmQuery } from "@/api/algorithm/model";
import { Dataset, DatasetQuery } from "@/api/dataset/model";
import EditDialog from "@/components/DataList/EditDialog/index.vue";

const props = defineProps<{
  listType: string;
}>();

const isDatasetList: boolean = props.listType === "dataset";

const API = isDatasetList ? DatasetAPI : AlgorithmAPI;

const queryFormRef = ref(ElForm);
const loading = ref(false);
const queryParams = reactive<AlgorithmQuery | DatasetQuery>({});
const list = ref<Algorithm[] | Dataset[]>([]);
const selectedId = ref<number>();
const selectedColumns = ref([
  "name",
  "type",
  "description",
  "size",
  isDatasetList ? "total" : "importPath",
  "path",
  "status",
]);
const columns = [
  { label: "名称", value: "name" },
  { label: "类型", value: "type" },
  { label: "描述", value: "description" },
  { label: "大小", value: "size" },
  isDatasetList
    ? { label: "图片数量", value: "total" }
    : { label: "代码导入路径", value: "importPath" },
  { label: "存储位置", value: "path" },
  { label: "状态", value: "status" },
];

function handleQuery() {
  loading.value = true;
  API.getList(queryParams)
    .then((data) => {
      list.value = data;
    })
    .then(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value.resetFields();
  handleQuery();
}

function onRowClick<T extends Algorithm | Dataset>(row: T) {
  selectedId.value = row.id;
}

const router = useRouter();

function handleShow<T extends Algorithm | Dataset>(row: T) {
  selectedId.value = row.id;
  router.push(`/dataset/${selectedId.value}`);
}

function handleDelete(algorithmId: number) {
  let tip = isDatasetList
    ? "确认删除已选中的数据集？其中的图片也将一并删除！"
    : "确认删除已选中的模型？";
  if (!algorithmId) {
    ElMessage.warning("请勾选删除项");
    return false;
  }
  ElMessageBox.confirm(tip, "警告", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      console.log([...algorithmId.toString()]);
      API.deleteByIds([...algorithmId.toString()]);
      ElMessage.success("删除成功");
    })
    .catch(() => ElMessage.info("已取消删除"));
}

const dialogRef = ref();

function onEdit<T extends Algorithm | Dataset>(type: string, row: T) {
  dialogRef.value.open(type, row);
}

function openDialog<T extends Algorithm | Dataset>(type: string, dataset: T) {
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
            :placeholder="isDatasetList ? '数据集名称' : '模型名称'"
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
        :data="list"
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
        <el-table-column
          v-if="selectedColumns.includes('description')"
          label="描述"
          prop="description"
        />
        <el-table-column
          v-if="selectedColumns.includes('size')"
          label="大小"
          prop="size"
        />
        <el-table-column
          v-if="isDatasetList && selectedColumns.includes('total')"
          label="图片数量"
          prop="total"
        />
        <el-table-column
          v-if="!isDatasetList && selectedColumns.includes('importPath')"
          label="代码导入路径"
          prop="importPath"
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
            <el-tag
              v-if="
                !(
                  scope.row.parentId === -1 && scope.row.children.length !== 0
                ) && scope.row.status === 1
              "
              type="success"
              >启用</el-tag
            >
            <el-tag
              v-else-if="
                !(
                  scope.row.parentId === -1 && scope.row.children.length !== 0
                ) && scope.row.status === 0
              "
              type="info"
              >禁用</el-tag
            >
          </template>
        </el-table-column>
        <el-table-column align="center" fixed="right" label="操作" width="260">
          <template #default="scope">
            <el-button
              link
              v-if="isDatasetList"
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
      :isDatasetList="isDatasetList"
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
