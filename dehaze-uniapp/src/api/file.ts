import { ResultEnum, SESSION_KEY } from "dehaze-sdk-js";
import type { FileInfo } from "dehaze-sdk-js";
import { uploadFileByUni } from "./uni-adapter";
import { BASE_URL } from "./constants";
import { redirectToLogin } from "./session";
import { useAuthStore } from "@/store/auth";

export interface UploadImageParams {
  url: string;
}

export async function uploadImage(
  imageData: UploadImageParams,
  onProgress?: (progress: number) => void
): Promise<FileInfo> {
  // 在函数内部调用 useAuthStore，避免模块加载期循环依赖
  const authStore = useAuthStore();
  // 优先从 auth store 读取（单一数据源），store 尚未初始化时回退 storage
  const sessionId =
    authStore.sessionId || uni.getStorageSync(SESSION_KEY) || "";

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

  if (
    response.code === "A0230" ||
    response.code === "A0231" ||
    response.code === "A0301"
  ) {
    redirectToLogin();
    throw new Error(response.msg || "登录已失效");
  }

  if (response.code !== ResultEnum.SUCCESS) {
    throw new Error(response.msg || "上传失败");
  }

  return response.data;
}
