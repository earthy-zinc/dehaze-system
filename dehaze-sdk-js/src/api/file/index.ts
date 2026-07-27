import { PageResult } from "@/types";
import request from "@/utils/request";
import { FileInfo, FileQuery } from "./model";

class FileAPI {
  /**
   * 文件上传检查（MD5秒传）
   *
   * @param md5 文件md5
   * @returns Promise<FileInfo | null> 文件已存在则返回文件信息，否则返回 null
   */
  static uploadCheck(md5: string) {
    return request<FileInfo | null>({
      url: "/api/v1/files/check",
      method: "get",
      params: { md5 },
    });
  }

  /**
   * 上传文件
   *
   * @param file
   * @param modelId
   * @param onUploadProgress 上传进度回调
   */
  static upload(
    file: File,
    modelId?: number,
    onUploadProgress?: (progressEvent: { loaded: number; total?: number }) => void
  ) {
    const formData = new FormData();
    if (modelId) {
      formData.append("modelId", modelId.toString());
    }
    formData.append("file", file);
    return request<FileInfo>({
      url: "/api/v1/files",
      method: "post",
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
      ...(onUploadProgress ? { onUploadProgress } : {}),
    });
  }

  /**
   * 删除文件（通过文件ID）
   *
   * @param fileId 文件ID
   */
  static deleteById(fileId: number) {
    return request<void>({
      url: "/api/v1/files",
      method: "delete",
      params: { fileId },
    });
  }

  /**
   * 文件分页查询
   *
   * @param query 查询参数
   */
  static getPage(query?: FileQuery) {
    return request<PageResult<FileInfo[]>>({
      url: "/api/v1/files/page",
      method: "get",
      params: query,
    });
  }

  /**
   * 获取文件详情
   *
   * @param fileId 文件ID
   */
  static getById(fileId: number) {
    return request<FileInfo>({
      url: `/api/v1/files/${fileId}`,
      method: "get",
    });
  }

  /**
   * 下载文件（返回 Blob，适用于浏览器端下载）
   *
   * @param objectName 文件存储对象名
   */
  static download(objectName: string) {
    return request<Blob>({
      url: `/api/v1/files/download/${objectName}`,
      method: "get",
      responseType: "blob",
    });
  }
}

export default FileAPI;
