/**
 * 文件上传工具
 *
 * H5 端通过 Taro.chooseMedia 选择/拍摄的图片是 blob: URL，
 * 需将 blob 转为带正确文件名与 MIME 的 File，再以 FormData 上传。
 * 小程序端临时路径自带扩展名，沿用 Taro.uploadFile。
 *
 * 上传接口不走 SDK axios 拦截器，需手动注入 X-Session-Id header 与 baseURL。
 */
import Taro from "@tarojs/taro";
import type { FileInfo } from "dehaze-sdk-js";
import { storage } from "@/utils/storage";
import { apiConfig } from "@/config/api";

interface UploadResponse {
  code: string;
  msg: string;
  data: FileInfo;
}

function getUploadUrl(): string {
  return `${apiConfig.java}/api/v1/files`;
}

/**
 * H5 端上传：blob URL → File → FormData → XHR
 */
async function uploadInH5(
  filePath: string,
  fileName: string,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  const blob = await fetch(filePath).then((res) => res.blob());
  const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", getUploadUrl());
    const sessionId = storage.getSessionId();
    if (sessionId) {
      xhr.setRequestHeader("X-Session-Id", sessionId);
    }
    xhr.onload = () => {
      if (xhr.status !== 200) {
        reject(new Error(`上传失败: HTTP ${xhr.status}`));
        return;
      }
      try {
        const response = JSON.parse(xhr.responseText) as UploadResponse;
        if (response.code === "00000") {
          resolve(response.data);
        } else {
          reject(new Error(response.msg || "上传失败"));
        }
      } catch {
        reject(new Error("上传响应解析失败"));
      }
    };
    xhr.onerror = () => reject(new Error("上传失败，请检查网络"));
    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
    }
    xhr.send(formData);
  });
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
      url: getUploadUrl(),
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
