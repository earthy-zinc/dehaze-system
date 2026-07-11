import request from "@/utils/request";
import { FileInfo } from "./model";

class FileAPI {
  /**
   * 文件上传检查（MD5秒传）
   *
   * @param md5 文件md5
   * @returns Promise<FileInfo | null> 文件已存在则返回文件信息，否则返回 null
   */
  static uploadCheck(md5: string) {
    return request<any, FileInfo | null>({
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
    return request<any, FileInfo>({
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
    return request<any, void>({
      url: "/api/v1/files",
      method: "delete",
      params: { fileId },
    });
  }
}

export default FileAPI;
