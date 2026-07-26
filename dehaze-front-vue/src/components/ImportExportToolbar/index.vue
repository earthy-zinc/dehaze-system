<script lang="ts" setup>
import { ExportModule, ImportModule } from "dehaze-sdk-js";
import { ArrowDown, Download, Top, Upload } from "@element-plus/icons-vue";
import { useImportExport } from "@/composables/useImportExport";
import ImportDialog from "./ImportDialog.vue";
import ExportDialog from "./ExportDialog.vue";
import TaskListDrawer from "./TaskListDrawer.vue";

defineOptions({
  name: "ImportExportToolbar",
});

const props = withDefaults(
  defineProps<{
    module: ExportModule;
    importable?: boolean;
    queryParams: Record<string, any>;
    extraImportParams?: Record<string, unknown>;
    fields?: { label: string; value: string }[];
  }>(),
  {
    importable: true,
  }
);

const emit = defineEmits<{
  (e: "import-complete"): void;
}>();

const moduleRef = computed(() => props.module);
const queryParamsRef = computed(() => props.queryParams);
const extraImportParamsRef = computed(() => props.extraImportParams);

const { downloadTemplate, templateLoading } = useImportExport({
  module: moduleRef,
  queryParams: queryParamsRef,
  extraImportParams: extraImportParamsRef,
});

const importDialogVisible = ref(false);
const exportDialogVisible = ref(false);
const exportInitialFormat = ref<"excel" | "csv">("excel");
const taskDrawerVisible = ref(false);

function handleDownloadTemplate(format: "excel" | "csv") {
  downloadTemplate(format);
}

function openImportDialog() {
  importDialogVisible.value = true;
}

function handleExportCommand(command: string) {
  exportInitialFormat.value = command as "excel" | "csv";
  exportDialogVisible.value = true;
}

function openTaskDrawer() {
  taskDrawerVisible.value = true;
}

function handleImportComplete() {
  emit("import-complete");
}

const importModule = computed(() => props.module as ImportModule);

const exportFields = computed(() => props.fields);
</script>

<template>
  <div class="import-export-toolbar">
    <el-dropdown v-if="importable" split-button @click="openImportDialog">
      <el-icon><Top /></el-icon>导入
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item @click="handleDownloadTemplate('excel')">
            <el-icon><Download /></el-icon>
            下载 Excel 模板
          </el-dropdown-item>
          <el-dropdown-item @click="handleDownloadTemplate('csv')">
            <el-icon><Download /></el-icon>
            下载 CSV 模板
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>

    <el-dropdown class="ml-3" @command="handleExportCommand">
      <el-button :loading="templateLoading">
        <el-icon><Download /></el-icon>
        导出
        <el-icon class="el-icon--right"><ArrowDown /></el-icon>
      </el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="excel">导出为 Excel</el-dropdown-item>
          <el-dropdown-item command="csv">导出为 CSV</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>

    <el-button class="ml-3" @click="openTaskDrawer">
      <el-icon><Upload /></el-icon>
      任务列表
    </el-button>

    <ImportDialog
      v-if="importable"
      v-model="importDialogVisible"
      :module="importModule"
      :extra-import-params="extraImportParams"
      @import-complete="handleImportComplete"
    />

    <ExportDialog
      v-model="exportDialogVisible"
      :module="module"
      :query-params="queryParams"
      :fields="exportFields"
      :initial-format="exportInitialFormat"
    />

    <TaskListDrawer v-model="taskDrawerVisible" :module="module" />
  </div>
</template>

<style lang="scss" scoped>
.import-export-toolbar {
  display: inline-flex;
  align-items: center;
}
</style>
