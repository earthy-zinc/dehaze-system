/**
 * 文件上传 API
 *
 * API 路径：
 * - POST /files       上传文件
 * - GET  /files/{id}  获取文件信息
 */

import { get } from "./request";
import type { ImageData } from "@/pages/image-input/data/imageInputData";

// ==================== 类型定义 ====================

/** 后端返回的文件信息 */
export interface FileInfo {
  id: number;
  name: string;
  path: string;
  url: string;
}

/** 文件记录（从 /files/page 返回） */
export interface SysFile {
  id: number;
  type?: string;
  url?: string;
  name: string;
  objectName: string;
  size: string;
  path: string;
  md5: string;
  createTime: string;
  updateTime: string;
}

/** 文件分页结果 */
export interface FilePageResult {
  list: SysFile[];
  total: number;
  pageNum: number;
  pageSize: number;
}

// ==================== API 方法 ====================

/** 获取文件信息 */
export async function getFileInfo(fileId: number): Promise<FileInfo> {
  return get<FileInfo>(`/files/${fileId}`);
}

/** 获取文件分页列表 */
export async function getFileList(
  pageNum = 1,
  pageSize = 20,
  keywords = ""
): Promise<FilePageResult> {
  const params: Record<string, unknown> = { pageNum, pageSize };
  if (keywords) params.keywords = keywords;
  return get<FilePageResult>("/files/page", { data: params });
}

/**
 * 上传图片并返回文件信息
 *
 * 使用 uni.uploadFile（multipart/form-data），
 * 不支持 request.ts 中的拦截器，需要手动传 token。
 */
export async function uploadImage(
  imageData: Omit<ImageData, "fileId">,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  const accessToken = uni.getStorageSync("access_token") || "";
  const authorization = accessToken.startsWith("Bearer ") ? accessToken : `Bearer ${accessToken}`;

  return new Promise((resolve, reject) => {
    const uploadTask = uni.uploadFile({
      url: getBaseUrl() + "/files",
      filePath: imageData.url,
      name: "file",
      formData: {},
      header: {
        Authorization: authorization,
      },
      success: (res) => {
        if (res.statusCode === 200) {
          try {
            const response = JSON.parse(res.data) as { code: string; data: FileInfo; msg: string };
            if (response.code === "00000") {
              resolve(response.data);
            } else {
              reject(new Error(response.msg || "上传失败"));
            }
          } catch {
            reject(new Error("解析响应失败"));
          }
        } else {
          reject(new Error(`上传失败: ${res.statusCode}`));
        }
      },
      fail: (err) => {
        reject(new Error(err.errMsg || "上传失败"));
      },
    });

    if (onProgress) {
      uploadTask.onProgressUpdate((res) => {
        onProgress(res.progress);
      });
    }
  });
}

/** 获取平台相关的 baseURL */
function getBaseUrl(): string {
  // #ifdef H5
  return "/api/v1";
  // #endif
  // #ifndef H5
  return "http://127.0.0.1:8989/api/v1";
  // #endif
}
