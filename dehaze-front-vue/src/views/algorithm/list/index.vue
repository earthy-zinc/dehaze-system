<script lang="ts" setup>
import EditDialog from "@/components/DataList/EditDialog/index.vue";
import type { FormInstance } from "element-plus";
import { Algorithm, AlgorithmAPI, AlgorithmQuery } from "dehaze-sdk-js";
import {
  Search,
  Refresh,
  Setting,
  Plus,
  Edit,
  Delete,
} from "@element-plus/icons-vue";
import ImportExportToolbar from "@/components/ImportExportToolbar/index.vue";

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
const ids = ref<number[]>([]);
/** 全量算法（用于构建类型树） */
const allAlgorithms = ref<Algorithm[]>([]);
/** 当前选中的类型（空数组表示全部） */
const selectedTypes = ref<string[]>([]);
/** 类型树数据 */
const typeTree = computed(() => {
  const counts: Record<string, number> = {};
  allAlgorithms.value.forEach((a) => {
    const t = a.type || "未分类";
    counts[t] = (counts[t] || 0) + 1;
  });
  return Object.entries(counts).map(([type, count]) => ({
    label: type,
    value: type,
    count,
  }));
});

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
      allAlgorithms.value = data;
      applyTypeFilter();
    })
    .catch((e) => {
      ElMessage.error("查询失败：" + e.message);
    })
    .finally(() => {
      loading.value = false;
    });
}

/** 按选中的类型筛选列表 */
function applyTypeFilter() {
  if (selectedTypes.value.length === 0) {
    list.value = allAlgorithms.value;
  } else {
    list.value = allAlgorithms.value.filter((a) =>
      selectedTypes.value.includes(a.type || "未分类")
    );
  }
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  selectedTypes.value = [];
  handleQuery();
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

const dialogRef = ref();

function openDialog(type: string, row: Partial<Algorithm>) {
  dialogRef.value.open(type, row);
}

// 删除算法（支持单行和批量）
function handleDelete(row?: Algorithm) {
  if (row) {
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
        if (err !== "cancel" && err !== "close") {
          ElMessage.error("删除失败：" + (err.message || "未知错误"));
        }
      });
  } else if (ids.value.length > 0) {
    ElMessageBox.confirm(
      `确认删除选中的 ${ids.value.length} 个算法吗？删除后不可恢复。`,
      "警告",
      {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      }
    )
      .then(async () => {
        await AlgorithmAPI.deleteByIds(ids.value.map(String));
        ElMessage.success("删除成功");
        handleQuery();
      })
      .catch((err) => {
        if (err !== "cancel" && err !== "close") {
          ElMessage.error("删除失败：" + (err.message || "未知错误"));
        }
      });
  } else {
    ElMessage.warning("请勾选删除项");
  }
}

/** 行复选框选中记录选中ID集合 */
function handleSelectionChange(selection: any) {
  ids.value = selection.map((item: any) => item.id);
}

const statusMap: Record<
  number,
  { label: string; type: "primary" | "success" | "warning" | "info" | "danger" }
> = {
  1: { label: "草稿", type: "info" },
  2: { label: "测试中", type: "warning" },
  3: { label: "待审核", type: "warning" },
  4: { label: "已发布", type: "success" },
  5: { label: "已停用", type: "danger" },
  6: { label: "已归档", type: "info" },
};

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
  <div class="algorithm-page">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="算法类型">
          <el-checkbox-group v-model="selectedTypes" @change="applyTypeFilter">
            <el-checkbox-button
              v-for="node in typeTree"
              :key="node.value"
              :value="node.value"
              size="small"
            >
              {{ node.label }} ({{ node.count }})
            </el-checkbox-button>
          </el-checkbox-group>
        </el-form-item>
        <el-form-item>
          <el-input
            v-model="queryParams.keywords"
            placeholder="搜索算法名称"
            clearable
            size="default"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery">
            <el-icon><Search /></el-icon>
            搜索
          </el-button>
          <el-button @click="resetQuery">
            <el-icon><Refresh /></el-icon>
            重置
          </el-button>
          <el-dropdown class="setting-button">
            <el-button>
              <el-icon><Setting /></el-icon>
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
          新增算法
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
          module="algorithm"
          :query-params="queryParams"
          @import-complete="handleQuery"
        />
        <span class="result-tip">
          共 <strong>{{ list.length }}</strong> 个算法
          <span v-if="selectedTypes.length > 0">
            （已筛选：{{ selectedTypes.join("、") }}）
          </span>
        </span>
      </div>

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
        @selection-change="handleSelectionChange"
      >
        <el-table-column align="center" type="selection" width="55" />
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
              @click.stop="handleShowDetail(scope.row as Algorithm)"
            >
              {{ scope.row.name }}
            </el-button>
          </template>
        </el-table-column>
        <el-table-column
          v-if="selectedColumns.includes('type')"
          label="类型"
          prop="type"
          width="120"
        >
          <template #default="scope">
            <el-tag v-if="scope.row.type" size="small" type="info">
              {{ scope.row.type }}
            </el-tag>
            <span v-else class="text-muted">-</span>
          </template>
        </el-table-column>
        <el-table-column
          v-if="selectedColumns.includes('description')"
          label="描述"
          min-width="300"
          prop="description"
          show-overflow-tooltip
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
          show-overflow-tooltip
        />
        <el-table-column
          v-if="selectedColumns.includes('path')"
          label="存储位置"
          prop="path"
          width="300"
          show-overflow-tooltip
        />
        <el-table-column
          v-if="selectedColumns.includes('status')"
          align="center"
          label="状态"
          min-width="90"
        >
          <template #default="scope">
            <el-tag
              v-if="scope.row.status != null && statusMap[scope.row.status]"
              :type="statusMap[scope.row.status].type"
              size="small"
            >
              {{ statusMap[scope.row.status].label }}
            </el-tag>
            <span v-else class="text-muted">-</span>
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
              <el-icon><Plus /></el-icon>
              新增
            </el-button>
            <el-button
              link
              size="small"
              type="primary"
              @click.stop="openDialog('编辑', scope.row)"
            >
              <el-icon><Edit /></el-icon>
              编辑
            </el-button>
            <el-button
              link
              size="small"
              type="danger"
              @click.stop="handleDelete(scope.row as Algorithm)"
            >
              <el-icon><Delete /></el-icon>
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
        label-width="100px"
      >
        <el-descriptions-item label="算法名称">
          {{ detailData.name }}
        </el-descriptions-item>
        <el-descriptions-item label="算法类型">
          <el-tag v-if="detailData.type" size="small" type="info">
            {{ detailData.type }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="算法大小">
          {{ detailData.size }}
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag
            v-if="detailData.status != null && statusMap[detailData.status]"
            :type="statusMap[detailData.status].type"
            size="small"
          >
            {{ statusMap[detailData.status].label }}
          </el-tag>
          <span v-else class="text-muted">-</span>
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
.algorithm-page {
  display: flex;
  flex-direction: column;
  padding: 16px;
}

.search-container {
  padding: 16px;
  background: #fff;
  border-radius: 4px;

  .el-form-item {
    margin-bottom: 0;
  }
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

.toolbar {
  display: flex;
  gap: 16px;
  align-items: center;
  margin-bottom: 12px;

  .result-tip {
    font-size: 13px;
    color: var(--el-text-color-secondary);

    strong {
      margin: 0 2px;
      font-size: 14px;
      color: var(--el-color-primary);
    }
  }
}

.text-muted {
  color: var(--el-text-color-placeholder);
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
