import {
  ExportModule,
  ExportRequest,
  ExportResult,
  ImportExportAPI,
  ImportModule,
  ImportRequest,
  ImportResult,
  ImportTaskResult,
} from "dehaze-sdk-js";
import type { Ref } from "vue";

interface UseImportExportOptions {
  module: Ref<ExportModule>;
  queryParams: Ref<Record<string, any>>;
  extraImportParams?: Ref<Record<string, any> | undefined>;
}

const buildFileName = (prefix: string, format: "excel" | "csv") => {
  const ext = format === "excel" ? "xlsx" : "csv";
  const ts = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `${prefix}_${ts}.${ext}`;
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

const isExportResult = (data: unknown): data is ExportResult =>
  typeof data === "object" && data !== null && "taskId" in data;

const isImportTaskResult = (data: unknown): data is ImportTaskResult =>
  typeof data === "object" &&
  data !== null &&
  "taskId" in data &&
  "status" in data;

const filterQueryParams = (
  params: Record<string, any>
): Record<string, any> => {
  const { pageNum, pageSize, ...rest } = params;
  return rest;
};

export function useImportExport(options: UseImportExportOptions) {
  const exportLoading = ref(false);
  const importLoading = ref(false);
  const templateLoading = ref(false);

  const downloadTemplate = async (format: "excel" | "csv" = "excel") => {
    const module = options.module.value as ImportModule;
    templateLoading.value = true;
    try {
      const blob = await ImportExportAPI.downloadTemplate(module, format);
      const filename = buildFileName(`${module}_template`, format);
      downloadBlob(blob, filename);
      ElMessage.success("模板下载成功");
    } catch (e: any) {
      ElMessage.error(e.message || "模板下载失败");
    } finally {
      templateLoading.value = false;
    }
  };

  const exportData = async (
    format: "excel" | "csv",
    fields?: string[],
    forceAsync?: boolean
  ): Promise<{ isAsync: boolean; taskId?: string; blob?: Blob }> => {
    const module = options.module.value;
    exportLoading.value = true;
    try {
      const params: ExportRequest = {
        ...filterQueryParams(options.queryParams.value),
        format,
        fields: fields && fields.length > 0 ? fields : undefined,
        async: forceAsync,
      };
      const result = await ImportExportAPI.export(module, params);
      if (result instanceof Blob) {
        return { isAsync: false, blob: result };
      }
      if (isExportResult(result)) {
        return { isAsync: true, taskId: result.taskId };
      }
      throw new Error("导出响应格式异常");
    } finally {
      exportLoading.value = false;
    }
  };

  const downloadExportBlob = (blob: Blob, format: "excel" | "csv") => {
    const filename = buildFileName(options.module.value, format);
    downloadBlob(blob, filename);
  };

  const importData = async (
    file: File,
    mode: "all" | "partial" = "all",
    forceAsync?: boolean
  ): Promise<
    { isAsync: false; result: ImportResult } | { isAsync: true; taskId: string }
  > => {
    const module = options.module.value as ImportModule;
    importLoading.value = true;
    try {
      const params: ImportRequest = {
        mode,
        async: forceAsync,
        ...(options.extraImportParams?.value || {}),
      };
      const result = await ImportExportAPI.import(module, params, file);
      if (isImportTaskResult(result)) {
        return { isAsync: true, taskId: result.taskId };
      }
      return { isAsync: false, result };
    } finally {
      importLoading.value = false;
    }
  };

  return {
    exportLoading,
    importLoading,
    templateLoading,
    downloadTemplate,
    exportData,
    downloadExportBlob,
    importData,
  };
}

export { downloadBlob, buildFileName };
