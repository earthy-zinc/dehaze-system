import { configAxios, service, SESSION_KEY } from "dehaze-sdk-js";
import type { AxiosError, InternalAxiosRequestConfig } from "dehaze-sdk-js";
import { createUniRequestAdapter } from "./uni-adapter";
import { API_HOST, SESSION_INVALID_EVENT, USER_INFO_KEY } from "./constants";
import { LOGIN_PATH } from "@/routers/guard";

let isRedirecting = false;
function redirectToLogin() {
  if (isRedirecting) return;
  isRedirecting = true;
  // 清理本地认证态（storage + 内存态由 auth store 监听事件同步清空）
  uni.removeStorageSync(SESSION_KEY);
  uni.removeStorageSync(USER_INFO_KEY);
  uni.$emit(SESSION_INVALID_EVENT);
  uni.reLaunch({
    url: LOGIN_PATH,
    complete: () => {
      isRedirecting = false;
    },
  });
}

function handleResponseError(error: unknown): unknown {
  const axiosError = error as AxiosError<{ code?: string }>;
  const code = axiosError.response?.data?.code;

  if (code === "A0230" || code === "A0231" || code === "A0301") {
    redirectToLogin();
    return Promise.reject(error);
  }

  return Promise.reject(error);
}

const uniAdapter = createUniRequestAdapter();
service.defaults.adapter = uniAdapter;

configAxios({
  onRequest: (config: InternalAxiosRequestConfig) => {
    const sessionId = uni.getStorageSync(SESSION_KEY) || null;
    if (sessionId) {
      if (!config.headers) config.headers = {} as any;
      config.headers["X-Session-Id"] = sessionId;
    }
    return {
      ...config,
      baseURL: API_HOST,
      timeout: config.timeout || 30000,
    };
  },
  onResponseError: handleResponseError,
});
