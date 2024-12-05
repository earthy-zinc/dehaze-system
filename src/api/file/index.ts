import request from "@/utils/request";
import { FileInfo, ImageFileInfo } from "./model";

class FileAPI {
  /**
   * 文件上传检查
   *
   * @param md5 文件md5
   */
  static uploadCheck(md5: string) {
    return request({
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
   */
  static upload(file: File, modelId: number | undefined = undefined) {
    const formData = new FormData();
    if (modelId) {
      formData.append("modelId", modelId.toString());
    }
    formData.append("file", file);
    return request<any, FileInfo | ImageFileInfo>({
      url: "/api/v1/files",
      method: "post",
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * 删除文件
   *
   * @param filePath 文件完整路径
   */
  static deleteByPath(filePath?: string) {
    return request({
      url: "/api/v1/files",
      method: "delete",
      params: { filePath: filePath },
    });
  }
}

export default FileAPI;
