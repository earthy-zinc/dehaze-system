/**
 * HTTP 请求封装
 *
 * 基于 uni.request 的统一请求层，提供：
 * - 请求拦截器（自动注入 Authorization 头）
 * - 响应拦截器（统一响应码处理）
 * - Token 自动刷新
 * - 401/403 自动跳转登录页
 * - 防重复提交
 * - 统一错误提示
 */

import { BASE_URL, REQUEST_TIMEOUT, ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY } from "./config";
import { ResultCode, TOKEN_INVALID_CODES } from "./enums";

/** 统一响应结构 */
export interface ApiResponse<T = unknown> {
  code: string;
  msg: string;
  data: T;
  traceId?: string;
  timestamp?: number;
}

/** 请求选项 */
export interface RequestOptions {
  /** 请求 URL（相对于 baseURL） */
  url: string;
  /** 请求方法 */
  method?: "GET" | "POST" | "PUT" | "DELETE";
  /** 请求数据 */
  data?: Record<string, unknown> | string | ArrayBuffer;
  /** 请求头 */
  header?: Record<string, string>;
  /** 超时时间（毫秒） */
  timeout?: number;
  /** 是否显示 loading */
  showLoading?: boolean;
  /** loading 文字 */
  loadingText?: string;
  /** 是否显示错误提示 */
  showErrorToast?: boolean;
  /** 是否可以重试（token过期时） */
  retryOnTokenExpired?: boolean;
}

// ==================== Token 刷新相关 ====================

/** 是否正在刷新 Token */
let isRefreshing = false;

/** 等待刷新的请求队列 */
let refreshQueue: Array<{
  resolve: (token: string) => void;
  reject: (error: Error) => void;
}> = [];

/** 刷新 Token */
async function refreshToken(): Promise<string> {
  try {
    const refreshTokenStr = uni.getStorageSync(REFRESH_TOKEN_KEY);
    if (!refreshTokenStr) {
      throw new Error("No refresh token");
    }

    const res = await uni.request({
      url: `${BASE_URL}/auth/refresh`,
      method: "POST",
      data: { refreshToken: refreshTokenStr },
      timeout: REQUEST_TIMEOUT,
    });

    const response = res.data as ApiResponse<{
      accessToken: string;
      refreshToken: string;
    }>;

    if (res.statusCode === 200 && response.code === ResultCode.SUCCESS) {
      const { accessToken, refreshToken: newRefreshToken } = response.data;
      uni.setStorageSync(ACCESS_TOKEN_KEY, accessToken);
      uni.setStorageSync(REFRESH_TOKEN_KEY, newRefreshToken);
      return accessToken;
    }
    throw new Error("Refresh token failed");
  } catch (error) {
    // 刷新失败，清除 token 并跳转登录页
    clearAuth();
    uni.reLaunch({ url: "/pages/login/index" });
    throw error;
  }
}

/** 处理 Token 刷新队列 */
function processRefreshQueue(token: string | null, error: Error | null) {
  refreshQueue.forEach((item) => {
    if (token) {
      item.resolve(token);
    } else {
      item.reject(error!);
    }
  });
  refreshQueue = [];
}

/** 清除认证信息 */
export function clearAuth() {
  uni.removeStorageSync(ACCESS_TOKEN_KEY);
  uni.removeStorageSync(REFRESH_TOKEN_KEY);
}

// ==================== 重复提交防护 ====================

/** 正在进行的请求映射（url + method + data 为 key） */
const pendingRequests = new Map<string, ReturnType<typeof uni.request>>();

function getRequestKey(url: string, method: string, data?: unknown): string {
  return `${method}:${url}:${JSON.stringify(data || "")}`;
}

// ==================== 核心请求方法 ====================

/** 发起 HTTP 请求 */
export async function request<T = unknown>(
  options: RequestOptions
): Promise<T> {
  const {
    url,
    method = "GET",
    data,
    header = {},
    timeout = REQUEST_TIMEOUT,
    showLoading = false,
    loadingText = "加载中...",
    showErrorToast = true,
    retryOnTokenExpired = true,
  } = options;

  // 显示 loading
  if (showLoading) {
    uni.showLoading({ title: loadingText, mask: true });
  }

  // 获取 token
  const accessToken = uni.getStorageSync(ACCESS_TOKEN_KEY);

  // 组装请求头
  const headers: Record<string, string> = {
    "Content-Type": "application/json;charset=utf-8",
    ...header,
  };
  if (accessToken) {
    headers.Authorization = accessToken.startsWith("Bearer ")
      ? accessToken
      : `Bearer ${accessToken}`;
  }

  // 防重复提交：相同 GET 以外请求不重复发送
  const requestKey = method !== "GET" ? getRequestKey(url, method, data) : "";
  if (requestKey && pendingRequests.has(requestKey)) {
    console.warn("[Request] 检测到重复请求:", requestKey);
    return pendingRequests.get(requestKey)!.then(
      (res) => (res.data as ApiResponse<T>).data
    );
  }

  return new Promise<T>((resolve, reject) => {
    const doRequest = (retryToken?: string) => {
      if (retryToken) {
        headers.Authorization = retryToken.startsWith("Bearer ")
          ? retryToken
          : `Bearer ${retryToken}`;
      }

      const task = uni.request({
        url: `${BASE_URL}${url}`,
        method,
        data,
        header: headers,
        timeout,
        success: (res) => {
          if (showLoading) {
            uni.hideLoading();
          }

          const statusCode = res.statusCode;
          const response = res.data as ApiResponse<T>;

          // HTTP 200 且业务码成功
          if (statusCode === 200 && response.code === ResultCode.SUCCESS) {
            resolve(response.data);
            return;
          }

          // Token 过期处理
          if (
            retryOnTokenExpired &&
            TOKEN_INVALID_CODES.includes(response.code as typeof ResultCode.TOKEN_INVALID)
          ) {
            handleTokenExpired(url, method, data, headers, timeout, doRequest, resolve, reject);
            return;
          }

          // 业务错误
          if (showErrorToast) {
            uni.showToast({
              title: response.msg || "请求失败",
              icon: "none",
              duration: 2000,
            });
          }
          reject(new ApiError(response.code, response.msg));
        },
        fail: (err) => {
          if (showLoading) {
            uni.hideLoading();
          }

          const message = getNetworkErrorMessage(err);
          if (showErrorToast) {
            uni.showToast({
              title: message,
              icon: "none",
              duration: 2000,
            });
          }
          reject(new Error(message));
        },
        complete: () => {
          // 清理重复请求记录
          if (requestKey) {
            pendingRequests.delete(requestKey);
          }
        },
      });

      // 记录请求（用于防重复提交）
      if (requestKey) {
        pendingRequests.set(requestKey, task);
      }
    };

    doRequest();
  });
}

/** 处理 Token 过期 */
function handleTokenExpired<T>(
  url: string,
  method: string,
  data: unknown,
  headers: Record<string, string>,
  timeout: number,
  doRequest: (retryToken?: string) => void,
  resolve: (value: T) => void,
  reject: (reason: Error) => void
) {
  // 加入刷新队列
  if (!isRefreshing) {
    isRefreshing = true;
    refreshToken()
      .then((newToken) => {
        isRefreshing = false;
        processRefreshQueue(newToken, null);
      })
      .catch((err) => {
        isRefreshing = false;
        processRefreshQueue(null, err);
        reject(err);
      });
  }

  refreshQueue.push({
    resolve: (newToken: string) => {
      doRequest(newToken);
    },
    reject: (error: Error) => {
      reject(error);
    },
  });
}

/** API 错误类 */
export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = "ApiError";
  }
}

/** 获取网络错误提示消息 */
function getNetworkErrorMessage(err: UniApp.GeneralCallbackResult): string {
  const errMsg = err.errMsg || "";
  if (errMsg.includes("timeout")) return "请求超时，请重试";
  if (errMsg.includes("fail")) return "网络异常，请检查网络连接";
  return "请求失败，请重试";
}

// ==================== 便捷方法 ====================

/** GET 请求 */
export function get<T = unknown>(url: string, options?: Omit<RequestOptions, "url" | "method">) {
  return request<T>({ ...options, url, method: "GET" });
}

/** POST 请求 */
export function post<T = unknown>(url: string, data?: Record<string, unknown>, options?: Omit<RequestOptions, "url" | "method" | "data">) {
  return request<T>({ ...options, url, method: "POST", data });
}

/** PUT 请求 */
export function put<T = unknown>(url: string, data?: Record<string, unknown>, options?: Omit<RequestOptions, "url" | "method" | "data">) {
  return request<T>({ ...options, url, method: "PUT", data });
}

/** DELETE 请求 */
export function del<T = unknown>(url: string, options?: Omit<RequestOptions, "url" | "method">) {
  return request<T>({ ...options, url, method: "DELETE" });
}

/**
 * 文件上传
 *
 * 注意：uni.uploadFile 使用 multipart/form-data 格式，
 * 无法通过 uni.request 的拦截器自动添加 Authorization 头，
 * 因此需要手动在 header 中传入 token。
 */
export function uploadFile(
  filePath: string,
  options?: {
    url?: string;
    name?: string;
    formData?: Record<string, string>;
    onProgress?: (progress: number) => void;
  }
): Promise<string> {
  const { url = "/files", name = "file", formData = {}, onProgress } = options || {};

  return new Promise((resolve, reject) => {
    const accessToken = uni.getStorageSync(ACCESS_TOKEN_KEY) || "";

    const uploadTask = uni.uploadFile({
      url: `${BASE_URL}${url}`,
      filePath,
      name,
      formData,
      header: {
        Authorization: accessToken.startsWith("Bearer ") ? accessToken : `Bearer ${accessToken}`,
      },
      success: (res) => {
        if (res.statusCode === 200) {
          try {
            const response = JSON.parse(res.data) as ApiResponse<{ url: string }>;
            if (response.code === ResultCode.SUCCESS) {
              resolve(response.data.url);
            } else {
              reject(new ApiError(response.code, response.msg));
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

    // 上传进度回调
    if (onProgress) {
      uploadTask.onProgressUpdate((res) => {
        onProgress(res.progress);
      });
    }
  });
}
