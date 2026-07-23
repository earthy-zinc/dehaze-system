/**
 * SDK 初始化
 *
 * 通过 dehaze-sdk-js 的 configJavaAxios / configPythonAxios 注入宿主逻辑：
 * - axios adapter 替换为 uni.request（全平台兼容）
 * - baseURL（Java 主后端 / Python 辅助后端）
 * - token 同步读取
 * - 响应错误处理（token 失效自动刷新 + 跳转登录）
 *
 * SDK 内部响应拦截器已校验 code===SUCCESS 并解包返回 response.data.data，
 * 因此业务调用方直接拿到载荷数据。
 *
 * 路径前缀说明：SDK 的 URL 已含 /api/v1 前缀，故 baseURL 仅设置 host（非 H5）
 * 或空字符串（H5，走 vite proxy）。
 */

import {
  configJavaAxios,
  configPythonAxios,
  javaService,
  pythonService,
  REFRESH_TOKEN_KEY,
  ResultEnum,
  TOKEN_KEY,
} from "dehaze-sdk-js";
import type { AxiosError, InternalAxiosRequestConfig } from "dehaze-sdk-js";
import { createUniRequestAdapter } from "./uni-adapter";
import { USER_INFO_KEY } from "./config";

/** Java 主后端 host（非 H5 平台使用完整 URL） */
const JAVA_HOST = "http://127.0.0.1:8989";

/** Python 辅助后端 host（算法推荐/收藏/对比） */
const PYTHON_HOST = "http://127.0.0.1:8991";

/** 请求超时时间（毫秒） */
const REQUEST_TIMEOUT = 30000;

/**
 * 获取当前平台的 Java baseURL
 * - H5：空字符串，SDK URL /api/v1/* 走 vite proxy（/api → 8989）
 * - 非 H5：完整 host
 */
function getJavaBaseURL(): string {
  // #ifdef H5
  return "";
  // #endif
  // #ifndef H5
  return JAVA_HOST;
  // #endif
}

/**
 * 获取当前平台的 Python baseURL
 * - H5：空字符串，SDK URL /api/v1/algorithm-select/* 走 vite proxy
 * - 非 H5：完整 host
 */
function getPythonBaseURL(): string {
  // #ifdef H5
  return "";
  // #endif
  // #ifndef H5
  return PYTHON_HOST;
  // #endif
}

// ==================== Token 刷新队列 ====================

/** Token 失效错误码 */
const TOKEN_INVALID_CODES: string[] = [
  ResultEnum.TOKEN_INVALID,
  ResultEnum.TOKEN_ACCESS_FORBIDDEN,
  ResultEnum.ACCESS_UNAUTHORIZED,
];

/** 是否正在刷新 Token */
let isRefreshing = false;

/** 等待 Token 刷新的请求队列 */
let refreshQueue: Array<{
  resolve: (token: string) => void;
  reject: (error: Error) => void;
}> = [];

/** 处理刷新队列 */
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

/** 清除本地认证信息 */
export function clearAuth() {
  uni.removeStorageSync(TOKEN_KEY);
  uni.removeStorageSync(REFRESH_TOKEN_KEY);
  uni.removeStorageSync(USER_INFO_KEY);
}

/** 跳转登录页（防重复跳转） */
let isRedirecting = false;
function redirectToLogin() {
  if (isRedirecting) return;
  isRedirecting = true;
  clearAuth();
  uni.reLaunch({
    url: "/pages/login/index",
    complete: () => {
      isRedirecting = false;
    },
  });
}

/** 统一响应结构（用于直接调用 uni.request 刷新 token） */
interface ApiResponse<T = unknown> {
  code: string;
  msg: string;
  data: T;
}

/**
 * 刷新 Token（直接使用 uni.request，避免经 SDK 触发递归）
 *
 * 刷新失败会清除认证信息并跳转登录页。
 */
async function refreshToken(): Promise<string> {
  const refreshTokenStr = uni.getStorageSync(REFRESH_TOKEN_KEY);
  if (!refreshTokenStr) {
    redirectToLogin();
    throw new Error("No refresh token, please login again");
  }

  try {
    const res = await uni.request({
      url: `${JAVA_HOST}/api/v1/auth/refresh`,
      method: "POST",
      data: { refreshToken: refreshTokenStr },
      timeout: REQUEST_TIMEOUT,
    });

    const response = res.data as ApiResponse<{
      accessToken: string;
      refreshToken?: string;
    }>;

    if (
      res.statusCode === 200 &&
      response.code === ResultEnum.SUCCESS
    ) {
      const { accessToken, refreshToken: newRefreshToken } = response.data;
      uni.setStorageSync(TOKEN_KEY, accessToken);
      if (newRefreshToken) {
        uni.setStorageSync(REFRESH_TOKEN_KEY, newRefreshToken);
      }
      return accessToken;
    }
    throw new Error(response.msg || "Refresh token failed");
  } catch (error) {
    redirectToLogin();
    throw error instanceof Error
      ? error
      : new Error("Refresh token failed");
  }
}

/** 判断错误是否为 token 失效 */
function isTokenInvalidError(error: unknown): boolean {
  const err = error as AxiosError<{ code?: string }>;
  const code = err.response?.data?.code;
  return typeof code === "string" && TOKEN_INVALID_CODES.includes(code);
}

/**
 * 响应错误处理：token 失效时自动刷新并重试
 *
 * - 首个失效请求触发刷新，后续请求入队等待
 * - 刷新成功后重试所有排队请求
 * - 刷新失败或非 token 错误则拒绝
 */
function handleResponseError(error: unknown): unknown {
  if (!isTokenInvalidError(error)) {
    return Promise.reject(error);
  }

  const axiosError = error as AxiosError & { config?: InternalAxiosRequestConfig };
  const originalConfig = axiosError.config;

  // 无法获取原始配置，直接拒绝
  if (!originalConfig) {
    return Promise.reject(error);
  }

  // 已在重试中，入队等待
  if (isRefreshing) {
    return new Promise((resolve, reject) => {
      refreshQueue.push({
        resolve: (newToken: string) => {
          // 用新 token 重发原请求
          originalConfig.headers.Authorization = newToken.startsWith("Bearer ")
            ? newToken
            : `Bearer ${newToken}`;
          javaService.request(originalConfig).then(resolve).catch(reject);
        },
        reject: (err: Error) => reject(err),
      });
    });
  }

  // 触发刷新
  isRefreshing = true;
  return refreshToken()
    .then((newToken) => {
      isRefreshing = false;
      processRefreshQueue(newToken, null);
      // 重发原请求
      originalConfig.headers.Authorization = newToken.startsWith("Bearer ")
        ? newToken
        : `Bearer ${newToken}`;
      return javaService.request(originalConfig);
    })
    .catch((err: Error) => {
      isRefreshing = false;
      processRefreshQueue(null, err);
      return Promise.reject(err);
    });
}

// ==================== SDK 配置 ====================

/** 注入 uni.request 适配器（Java + Python 共用） */
const uniAdapter = createUniRequestAdapter();
javaService.defaults.adapter = uniAdapter;
pythonService.defaults.adapter = uniAdapter;

/** 配置 Java 主后端 */
configJavaAxios({
  getToken: () => uni.getStorageSync(TOKEN_KEY) || null,
  onRequest: (config: InternalAxiosRequestConfig) => ({
    ...config,
    baseURL: getJavaBaseURL(),
    timeout: config.timeout || REQUEST_TIMEOUT,
  }),
  onResponseError: handleResponseError,
});

/** 配置 Python 辅助后端 */
configPythonAxios({
  getToken: () => uni.getStorageSync(TOKEN_KEY) || null,
  onRequest: (config: InternalAxiosRequestConfig) => ({
    ...config,
    baseURL: getPythonBaseURL(),
    timeout: config.timeout || REQUEST_TIMEOUT,
  }),
  onResponseError: handleResponseError,
});

// 导出标记，确保此模块被 import 后才生效
export const sdkReady = true;
