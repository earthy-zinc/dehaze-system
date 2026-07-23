/**
 * 文件上传 API（uni-app 平台专用）
 *
 * 查询类接口直接使用 dehaze-sdk-js 的 FileAPI。
 * 文件上传使用 uni.uploadFile（小程序/App 全平台兼容），
 * 因 SDK 的 FileAPI.upload 依赖浏览器 FormData + File API。
 */

import { ResultEnum, TOKEN_KEY } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import { uploadFileByUni } from "./uni-adapter";
import { BASE_URL } from "./config";

/** 上传图片所需的最小参数 */
export interface UploadImageParams {
  /** 本地图片路径（tempFilePath / 本地文件路径） */
  url: string;
}

/**
 * 上传图片并返回文件信息
 *
 * 使用 uni.uploadFile（multipart/form-data），
 * 不支持 axios 拦截器，需要手动传 token。
 */
export async function uploadImage(
  imageData: UploadImageParams,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  const accessToken = uni.getStorageSync(TOKEN_KEY) || "";
  const authorization = accessToken.startsWith("Bearer ")
    ? accessToken
    : `Bearer ${accessToken}`;

  const { data, statusCode } = await uploadFileByUni(
    `${BASE_URL}/files`,
    imageData.url,
    {
      name: "file",
      header: { Authorization: authorization },
      onProgress,
    }
  );

  if (statusCode !== 200) {
    throw new Error(`上传失败: ${statusCode}`);
  }

  const response = data as {
    code: string;
    data: FileInfo;
    msg: string;
  };

  if (response.code !== ResultEnum.SUCCESS) {
    throw new Error(response.msg || "上传失败");
  }

  return response.data;
}
