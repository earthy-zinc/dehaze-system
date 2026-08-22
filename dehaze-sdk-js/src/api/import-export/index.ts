import request from "@/utils/request";
import {
  ExportModule,
  ExportRequest,
  ExportResult,
  ImportModule,
  ImportRequest,
  ImportResult,
  ImportTaskResult,
} from "./model";

/**
 * 模块标识到 URL 路径段的映射。
 * 算法模块的 CRUD 路径为 /api/v1/algorithms（复数），
 * 导入导出路径需与 CRUD 保持一致，故映射为 "algorithms"。
 */
// 通用导入导出端点的模块段即原始模块名（/api/v1/{module}/_export），
// 无需映射；单个算法配置 JSON 导出走 /api/v1/algorithms/{id}/_export 专用路由
function modulePath(module: string): string {
  return module;
}

class ImportExportAPI {
  /**
   * 简单查询条件导出（GET）
   *
   * - 同步模式：返回 Blob（文件流）
   * - 异步模式：返回 ExportResult（含 taskId）
   *
   * @param module 模块标识
   * @param params 查询参数（含 format/async/fields 及模块特定筛选条件）
   */
  static export(module: ExportModule, params: ExportRequest) {
    const { fields, ...rest } = params;
    const query: Record<string, unknown> = { ...rest };
    if (fields && fields.length > 0) {
      query.fields = fields.join(",");
    }
    return request<ExportResult | Blob>({
      url: `/api/v1/${module}/_export`,
      method: "get",
      params: query,
      responseType: "blob",
    });
  }

  /**
   * 复杂查询条件导出（POST 请求体）
   *
   * - 同步模式：返回 Blob（文件流）
   * - 异步模式：返回 ExportResult（含 taskId）
   *
   * @param module 模块标识
   * @param data 查询参数（请求体传递）
   */
  static exportByPost(module: ExportModule, data: ExportRequest) {
    return request<ExportResult | Blob>({
      url: `/api/v1/${module}/_export`,
      method: "post",
      data,
      responseType: "blob",
    });
  }

  /**
   * 导入数据
   *
   * - 同步模式：返回 ImportResult（含成功/失败统计）
   * - 异步模式：返回 ImportTaskResult（含 taskId）
   *
   * @param module 模块标识
   * @param params 导入参数（mode/async 及模块特定参数如 deptId）
   * @param file 上传的 Excel/CSV 文件
   */
  static import(module: ImportModule, params: ImportRequest, file: File) {
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, String(value));
      }
    });
    return request<ImportResult | ImportTaskResult>({
      url: `/api/v1/${module}/_import`,
      method: "post",
      data: formData,
      headers: { "Content-Type": "multipart/form-data" },
    });
  }

  /**
   * 下载导入模板
   *
   * @param module 模块标识
   * @param format 文件格式：excel(默认) / csv
   */
  static downloadTemplate(module: ImportModule, format: "excel" | "csv" = "excel") {
    return request<Blob>({
      url: `/api/v1/${module}/template`,
      method: "get",
      params: { format },
      responseType: "blob",
    });
  }
}

export default ImportExportAPI;
