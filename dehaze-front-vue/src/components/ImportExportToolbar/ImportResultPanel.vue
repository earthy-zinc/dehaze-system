<script lang="ts" setup>
import { ImportResult } from "dehaze-sdk-js";
import { Download } from "@element-plus/icons-vue";
import { downloadBlob } from "@/composables/useImportExport";

defineOptions({
  name: "ImportResultPanel",
});

const props = defineProps<{
  result: ImportResult;
}>();

const errorReportLoading = ref(false);

const summary = computed(() => [
  { label: "总行数", value: props.result.totalRows, type: "info" as const },
  { label: "成功", value: props.result.successCount, type: "success" as const },
  { label: "失败", value: props.result.failureCount, type: "danger" as const },
  { label: "跳过", value: props.result.skippedCount, type: "warning" as const },
]);

const hasErrors = computed(() => props.result.failureCount > 0);

async function downloadErrorReport() {
  if (!props.result.errorReportUrl) {
    ElMessage.warning("无错误报告可下载");
    return;
  }
  errorReportLoading.value = true;
  try {
    const res = await fetch(props.result.errorReportUrl!);
    if (!res.ok) throw new Error("下载失败");
    const blob = await res.blob();
    downloadBlob(blob, "import_error_report.xlsx");
  } catch (e: any) {
    ElMessage.error(e.message || "错误报告下载失败");
  } finally {
    errorReportLoading.value = false;
  }
}
</script>

<template>
  <div class="import-result-panel">
    <el-alert
      :title="
        hasErrors ? `导入完成，${result.failureCount} 条数据失败` : '导入完成'
      "
      :type="hasErrors ? 'warning' : 'success'"
      :closable="false"
      show-icon
    />

    <div class="summary-row">
      <el-statistic
        v-for="item in summary"
        :key="item.label"
        :title="item.label"
        :value="item.value"
      />
    </div>

    <div v-if="result.errorReportUrl" class="report-download">
      <el-button
        type="primary"
        plain
        :loading="errorReportLoading"
        @click="downloadErrorReport"
      >
        <el-icon><Download /></el-icon>
        下载错误报告
      </el-button>
    </div>

    <el-table
      v-if="result.errors && result.errors.length > 0"
      :data="result.errors"
      border
      max-height="300"
      class="error-table"
    >
      <el-table-column label="行号" prop="row" width="80" align="center" />
      <el-table-column label="字段" prop="field" width="140" />
      <el-table-column label="错误信息" prop="message" show-overflow-tooltip />
    </el-table>
  </div>
</template>

<style lang="scss" scoped>
.import-result-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.summary-row {
  display: flex;
  justify-content: space-around;
  padding: 8px 0;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.report-download {
  display: flex;
  justify-content: center;
}

.error-table {
  width: 100%;
}
</style>
