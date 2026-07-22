/**
 * 文件上传工具
 *
 * 关键点：H5 端通过 Taro.chooseMedia 选择/拍摄的图片是 blob: URL，
 * 若直接用 Taro.uploadFile 上传，multipart 文件名会退化为 blob 的 UUID（无扩展名），
 * 后端依据文件名推断 MIME 类型失败而返回 400。
 * 因此 H5 端需先将 blob 转为带正确文件名与 MIME 的 File，再以 FormData 上传；
 * 小程序端临时路径自带扩展名，沿用 Taro.uploadFile。
 *
 * 上传接口不走 SDK 的 axios 拦截器，需手动注入 Authorization 与 baseURL，
 * 与 dehaze-uniapp 的上传实现保持一致。
 */
import Taro from "@tarojs/taro";
import type { FileInfo } from "dehaze-sdk-js";
import { storage } from "@/utils/storage";
import { apiConfig } from "@/config/api";

/** 统一响应结构（上传返回的 data 为 JSON 字符串，需自行解析） */
interface UploadResponse {
  code: string;
  msg: string;
  data: FileInfo;
}

/** 组装 Authorization 请求头 */
function getAuthorization(): string {
  const token = storage.getToken() || "";
  return token.startsWith("Bearer ") ? token : `Bearer ${token}`;
}

/** 上传接口地址（H5 端 baseURL 为空，拼接后为相对路径 /api/v1/files，走代理） */
function getUploadUrl(): string {
  return `${apiConfig.java}/api/v1/files`;
}

/**
 * H5 端上传：blob URL → File（携带正确文件名与 MIME）→ FormData → XHR
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
    xhr.setRequestHeader("Authorization", getAuthorization());
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

/**
 * 小程序端上传：Taro.uploadFile（multipart/form-data）
 */
function uploadInMini(
  filePath: string,
  fileName: string,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  return new Promise((resolve, reject) => {
    const uploadTask = Taro.uploadFile({
      url: getUploadUrl(),
      filePath,
      name: "file",
      formData: { name: fileName },
      header: {
        Authorization: getAuthorization(),
      },
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

/**
 * 上传图片到文件服务，返回文件信息（含服务端可访问的 url 与 fileId）
 *
 * @param filePath 本地临时文件路径（H5 为 blob: URL，小程序为 wxfile:// 路径）
 * @param fileName 文件名（需带扩展名，后端据此推断 MIME 类型）
 * @param onProgress 上传进度回调（0-100）
 */
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
