/**
 * axios → uni.request 适配器
 *
 * 将 dehaze-sdk-js 内部的 axios 实例底层传输层替换为 uni.request，
 * 使 SDK 能在 uni-app 全平台（H5 / 小程序 / App）运行。
 *
 * 适配器只负责 HTTP 传输，业务逻辑（token 注入、响应码判断、data 提取）
 * 仍由 SDK 的 axios 拦截器处理。
 */

import type {
  AxiosAdapter,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "dehaze-sdk-js";

/** 将 params 对象序列化为 query string */
function buildQueryString(
  params: Record<string, unknown> | undefined
): string {
  if (!params || Object.keys(params).length === 0) return "";
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      value.forEach((v) => searchParams.append(key, String(v)));
    } else {
      searchParams.append(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

/** 组装完整 URL（baseURL + url + params） */
function buildFullUrl(config: InternalAxiosRequestConfig): string {
  const baseURL = config.baseURL || "";
  const url = config.url || "";
  const queryString = buildQueryString(config.params);
  return `${baseURL}${url}${queryString}`;
}

/** 将 AxiosHeaders 转换为 uni.request 的 header 对象 */
function extractHeaders(
  config: InternalAxiosRequestConfig
): Record<string, string> {
  const headers: Record<string, string> = {};
  if (config.headers) {
    for (const [key, value] of Object.entries(config.headers)) {
      if (typeof value === "string") {
        headers[key] = value;
      } else if (value !== undefined && value !== null) {
        headers[key] = String(value);
      }
    }
  }
  return headers;
}

/** uni.request 支持的 method 类型 */
type UniMethod = "GET" | "POST" | "PUT" | "DELETE" | "HEAD" | "OPTIONS" | "TRACE" | "CONNECT";

/**
 * 创建基于 uni.request 的 axios 适配器
 *
 * 处理两种场景：
 * 1. 普通请求（JSON / 表单）→ uni.request
 * 2. 文件上传（FormData 含文件）→ uni.uploadFile
 *
 * 注意：uni.request 不支持 PATCH，若收到 PATCH 请求将转为 POST。
 */
export function createUniRequestAdapter(): AxiosAdapter {
  return (config: InternalAxiosRequestConfig): Promise<AxiosResponse> => {
    return new Promise((resolve, reject) => {
      const fullUrl = buildFullUrl(config);
      const rawMethod = (config.method || "GET").toUpperCase();
      // uni.request 不支持 PATCH，回退为 POST
      const method = (rawMethod === "PATCH" ? "POST" : rawMethod) as UniMethod;
      const header = extractHeaders(config);

      // 响应类型处理
      const responseType = config.responseType;
      if (responseType === "arraybuffer") {
        header["responseType"] = "arraybuffer";
      }

      uni.request({
        url: fullUrl,
        method,
        data: config.data,
        header,
        timeout: config.timeout || 30000,
        // responseType: "arraybuffer" 时让 uni.request 返回 ArrayBuffer
        responseType: responseType === "arraybuffer" ? "arraybuffer" : "text",
        success: (res) => {
          const response: AxiosResponse = {
            data: res.data,
            status: res.statusCode,
            statusText: "",
            headers: (res.header || {}) as Record<string, string>,
            config,
            request: res,
          };
          resolve(response);
        },
        fail: (err) => {
          const error = new Error(err.errMsg || "网络请求失败") as Error & {
            config?: InternalAxiosRequestConfig;
            isAxiosError?: boolean;
          };
          error.config = config;
          error.isAxiosError = true;
          reject(error);
        },
      });
    });
  };
}

/**
 * 基于 uni.uploadFile 的文件上传适配器
 *
 * 适用于需要上传文件路径（filePath）的场景。
 * SDK 的 FileAPI.upload 使用 FormData + Blob，在小程序端不可用，
 * 因此 uniapp 端通过此函数直接调用 uni.uploadFile。
 */
export function uploadFileByUni(
  url: string,
  filePath: string,
  options?: {
    name?: string;
    formData?: Record<string, string>;
    header?: Record<string, string>;
    onProgress?: (progress: number) => void;
  }
): Promise<{ data: unknown; statusCode: number }> {
  const { name = "file", formData = {}, header = {}, onProgress } = options || {};

  return new Promise((resolve, reject) => {
    const uploadTask = uni.uploadFile({
      url,
      filePath,
      name,
      formData,
      header,
      success: (res) => {
        let parsedData: unknown;
        try {
          parsedData = JSON.parse(res.data);
        } catch {
          parsedData = res.data;
        }
        resolve({ data: parsedData, statusCode: res.statusCode });
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
