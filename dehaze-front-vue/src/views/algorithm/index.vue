<script lang="ts" setup>
import EditDialog from "@/components/DataList/EditDialog/index.vue";
import type { FormInstance } from "element-plus";
import { Algorithm, AlgorithmAPI, AlgorithmQuery } from "dehaze-sdk-js";

defineOptions({
  name: "AlgorithmList",
  inheritAttrs: false,
});

const queryFormRef = ref<FormInstance>();
const loading = ref(false);
const queryParams = reactive<AlgorithmQuery>({
  keywords: "",
});
const list = ref<Algorithm[]>([]);

// 列选项
const selectedColumns = ref([
  "name",
  "type",
  "description",
  "size",
  "importPath",
  "path",
  "status",
]);
const columns = [
  { label: "名称", value: "name" },
  { label: "类型", value: "type" },
  { label: "描述", value: "description" },
  { label: "大小", value: "size" },
  { label: "代码导入路径", value: "importPath" },
  { label: "存储位置", value: "path" },
  { label: "状态", value: "status" },
];

// 查询算法列表
function handleQuery() {
  loading.value = true;
  AlgorithmAPI.getList(queryParams)
    .then((data) => {
      list.value = data;
    })
    .catch((e) => {
      ElMessage.error("查询失败：" + e.message);
    })
    .finally(() => {
      loading.value = false;
    });
}

// 搜索防抖（300ms）
const debouncedQuery = useDebounceFn(handleQuery, 300);

// 关键字变化时防抖搜索
watch(
  () => queryParams.keywords,
  () => {
    debouncedQuery();
  }
);

function resetQuery() {
  queryFormRef.value?.resetFields();
  handleQuery();
}

const dialogRef = ref();

function openDialog<T extends Algorithm>(type: string, row: T) {
  dialogRef.value.open(type, row);
}

// 删除算法（带算法名确认文案）
function handleDelete(row: Algorithm) {
  ElMessageBox.confirm(
    `确认删除算法「${row.name}」吗？删除后不可恢复。`,
    "警告",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(async () => {
      await AlgorithmAPI.deleteByIds([row.id.toString()]);
      ElMessage.success("删除成功");
      handleQuery();
    })
    .catch((err) => {
      // 仅用户取消时静默，接口错误需提示
      if (err !== "cancel" && err !== "close") {
        ElMessage.error("删除失败：" + (err.message || "未知错误"));
      }
    });
}

// 切换算法状态
function handleStatusChange(row: Algorithm, val: string | number | boolean) {
  const status = Number(val);
  AlgorithmAPI.updateStatus(row.id, status)
    .then(() => {
      row.status = status;
      ElMessage.success(status === 1 ? "已启用" : "已禁用");
    })
    .catch((e) => {
      // 接口失败时回滚状态并提示
      row.status = status === 1 ? 0 : 1;
      ElMessage.error("状态切换失败：" + e.message);
    });
}

// 算法详情弹窗
const detailVisible = ref(false);
const detailData = ref<Algorithm>();

function handleShowDetail(row: Algorithm) {
  detailData.value = row;
  detailVisible.value = true;
}

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
            placeholder="算法名称"
            clearable
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
        :default-expand-all="true"
        highlight-current-row
        row-key="id"
      >
        <el-table-column
          v-if="selectedColumns.includes('name')"
          label="名称"
          prop="name"
          width="200"
        >
          <template #default="scope">
            <el-button
              link
              type="primary"
              @click.stop="handleShowDetail(scope.row)"
            >
              {{ scope.row.name }}
            </el-button>
          </template>
        </el-table-column>
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
          v-if="selectedColumns.includes('importPath')"
          label="代码导入路径"
          min-width="180"
          prop="importPath"
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
          min-width="90"
        >
          <template #default="scope">
            <el-switch
              :model-value="scope.row.status"
              :active-value="1"
              :inactive-value="0"
              active-text="启用"
              inactive-text="禁用"
              inline-prompt
              @change="(val) => handleStatusChange(scope.row, val)"
              @click.stop
            />
          </template>
        </el-table-column>
        <el-table-column align="center" fixed="right" label="操作" width="220">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click.stop="openDialog('新增', scope.row)"
            >
              <i-ep-plus />
              新增
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click.stop="openDialog('编辑', scope.row)"
            >
              <i-ep-edit />
              编辑
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click.stop="handleDelete(scope.row)"
            >
              <i-ep-delete />
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <EditDialog
      ref="dialogRef"
      :isDatasetList="false"
      @on-update="handleQuery"
      @on-add="handleQuery"
    />

    <!-- 算法详情弹窗 -->
    <el-dialog
      v-model="detailVisible"
      title="算法详情"
      width="640px"
      append-to-body
      destroy-on-close
    >
      <el-descriptions
        v-if="detailData"
        :column="2"
        border
        label-class-name="detail-label"
      >
        <el-descriptions-item label="算法名称">
          {{ detailData.name }}
        </el-descriptions-item>
        <el-descriptions-item label="算法类型">
          {{ detailData.type }}
        </el-descriptions-item>
        <el-descriptions-item label="算法大小">
          {{ detailData.size }}
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag v-if="detailData.status === 1" type="success">启用</el-tag>
          <el-tag v-else type="info">禁用</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="导入路径" :span="2">
          {{ detailData.importPath }}
        </el-descriptions-item>
        <el-descriptions-item label="存储位置" :span="2">
          {{ detailData.path }}
        </el-descriptions-item>
        <el-descriptions-item label="创建时间" :span="2">
          {{ detailData.createTime }}
        </el-descriptions-item>
        <el-descriptions-item label="算法描述" :span="2">
          {{ detailData.description }}
        </el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click="detailVisible = false">关闭</el-button>
      </template>
    </el-dialog>
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

:deep(.detail-label) {
  width: 110px;
  font-weight: bold;
}
</style>

<style lang="scss">
.el-divider--horizontal {
  margin: 5px 0;
}
</style>
