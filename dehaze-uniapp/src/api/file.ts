import { ResultEnum, SESSION_KEY } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import { uploadFileByUni } from "./uni-adapter";
import { BASE_URL } from "./constants";

export interface UploadImageParams {
  url: string;
}

export async function uploadImage(
  imageData: UploadImageParams,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  const sessionId = uni.getStorageSync(SESSION_KEY) || "";

  const header: Record<string, string> = {};
  if (sessionId) {
    header["X-Session-Id"] = sessionId;
  }

  const { data, statusCode } = await uploadFileByUni(
    `${BASE_URL}/files`,
    imageData.url,
    {
      name: "file",
      header,
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
