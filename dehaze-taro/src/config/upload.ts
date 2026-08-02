/**
 * 文件上传工具
 *
 * H5 端：blob URL → File，走 SDK 的 FileAPI.upload（复用 axios 拦截器，统一 baseURL/Token/错误处理）
 * 小程序端：Taro.uploadFile（小程序无 FormData/File 全局对象，SDK 不支持，需手动注入 X-Session-Id 与 baseURL）
 */
import Taro from "@tarojs/taro";
import { FileAPI } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import { storage } from "@/utils/storage";
import { UPLOAD_URL } from "./constants";

interface UploadResponse {
  code: string;
  msg: string;
  data: FileInfo;
}


/**
 * H5 端上传：blob URL → File → SDK FileAPI.upload
 */
async function uploadInH5(
  filePath: string,
  fileName: string,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  const blob = await fetch(filePath).then((res) => res.blob());
  const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });
  const onUploadProgress = onProgress
    ? (e: { loaded: number; total?: number }) => {
        if (e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      }
    : undefined;
  return FileAPI.upload(file, undefined, onUploadProgress);
}

function uploadInMini(
  filePath: string,
  fileName: string,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  return new Promise((resolve, reject) => {
    const header: Record<string, string> = {};
    const sessionId = storage.getSessionId();
    if (sessionId) {
      header["X-Session-Id"] = sessionId;
    }

    const uploadTask = Taro.uploadFile({
      url: UPLOAD_URL,
      filePath,
      name: "file",
      formData: { name: fileName },
      header,
      success: (res) => {
        if (res.statusCode !== 200) {
          reject(new Error(`上传失败: HTTP ${res.statusCode}`));
          return;
        }
        try {
          const response = JSON.parse(res.data as string) as UploadResponse;
          if (response.code === "00000") {
            resolve(response.data);
          } else {
            reject(new Error(response.msg || "上传失败"));
          }
        } catch {
          reject(new Error("上传响应解析失败"));
        }
      },
      fail: (err) => {
        reject(new Error(err.errMsg || "上传失败，请检查网络"));
      },
    });

    if (
      onProgress &&
      uploadTask &&
      typeof uploadTask.onProgressUpdate === "function"
    ) {
      uploadTask.onProgressUpdate((res) => {
        onProgress(res.progress);
      });
    }
  });
}

export function uploadImage(
  filePath: string,
  fileName: string,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  if (process.env.TARO_ENV === "h5") {
    return uploadInH5(filePath, fileName, onProgress);
  }
  return uploadInMini(filePath, fileName, onProgress);
}
