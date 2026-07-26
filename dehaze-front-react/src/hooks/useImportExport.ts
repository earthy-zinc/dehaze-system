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
import { message } from "antd";
import { useCallback, useState } from "react";

interface UseImportExportOptions {
  module: ExportModule;
  queryParams: object;
  extraImportParams?: Record<string, unknown>;
}

export type ExportOutcome =
  { isAsync: false; blob: Blob } | { isAsync: true; taskId: string };

export type ImportOutcome =
  { isAsync: false; result: ImportResult } | { isAsync: true; taskId: string };

const buildFileName = (prefix: string, format: "excel" | "csv") => {
  const ext = format === "excel" ? "xlsx" : "csv";
  const ts = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `${prefix}_${ts}.${ext}`;
};

export const downloadBlob = (blob: Blob, filename: string) => {
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

const filterQueryParams = (params: object): Record<string, unknown> => {
  const { pageNum, pageSize, ...rest } = params as Record<string, unknown>;
  return rest;
};

export function useImportExport(options: UseImportExportOptions) {
  const { module, queryParams, extraImportParams } = options;
  const [exportLoading, setExportLoading] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [templateLoading, setTemplateLoading] = useState(false);

  const downloadTemplate = useCallback(
    async (format: "excel" | "csv" = "excel") => {
      const importModule = module as ImportModule;
      setTemplateLoading(true);
      try {
        const blob = await ImportExportAPI.downloadTemplate(
          importModule,
          format
        );
        const filename = buildFileName(`${module}_template`, format);
        downloadBlob(blob, filename);
        message.success("模板下载成功");
      } catch (error: unknown) {
        message.error((error as Error)?.message || "模板下载失败");
      } finally {
        setTemplateLoading(false);
      }
    },
    [module]
  );

  const exportData = useCallback(
    async (
      format: "excel" | "csv",
      fields?: string[],
      forceAsync?: boolean
    ): Promise<ExportOutcome> => {
      setExportLoading(true);
      try {
        const params: ExportRequest = {
          ...filterQueryParams(queryParams),
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
        setExportLoading(false);
      }
    },
    [module, queryParams]
  );

  const downloadExportBlob = useCallback(
    (blob: Blob, format: "excel" | "csv") => {
      const filename = buildFileName(module, format);
      downloadBlob(blob, filename);
    },
    [module]
  );

  const importData = useCallback(
    async (
      file: File,
      mode: "all" | "partial" = "all",
      forceAsync?: boolean
    ): Promise<ImportOutcome> => {
      const importModule = module as ImportModule;
      setImportLoading(true);
      try {
        const params: ImportRequest = {
          mode,
          async: forceAsync,
          ...(extraImportParams || {}),
        };
        const result = await ImportExportAPI.import(importModule, params, file);
        if (isImportTaskResult(result)) {
          return { isAsync: true, taskId: result.taskId };
        }
        return { isAsync: false, result };
      } finally {
        setImportLoading(false);
      }
    },
    [module, extraImportParams]
  );

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
