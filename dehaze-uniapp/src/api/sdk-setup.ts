import {
  configJavaAxios,
  configPythonAxios,
  javaService,
  pythonService,
  SESSION_KEY,
} from "dehaze-sdk-js";
import type { AxiosError, InternalAxiosRequestConfig } from "dehaze-sdk-js";
import { createUniRequestAdapter } from "./uni-adapter";
import { USER_INFO_KEY } from "./config";

const JAVA_HOST = "http://127.0.0.1:8989";
const PYTHON_HOST = "http://127.0.0.1:8991";
const REQUEST_TIMEOUT = 30000;

function getJavaBaseURL(): string {
  // #ifdef H5
  return "";
  // #endif
  // #ifndef H5
  return JAVA_HOST;
  // #endif
}

function getPythonBaseURL(): string {
  // #ifdef H5
  return "";
  // #endif
  // #ifndef H5
  return PYTHON_HOST;
  // #endif
}

export function clearAuth() {
  uni.removeStorageSync(SESSION_KEY);
  uni.removeStorageSync(USER_INFO_KEY);
}

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
javaService.defaults.adapter = uniAdapter;
pythonService.defaults.adapter = uniAdapter;

configJavaAxios({
  onRequest: (config: InternalAxiosRequestConfig) => {
    const sessionId = uni.getStorageSync(SESSION_KEY) || null;
    if (sessionId) {
      if (!config.headers) config.headers = {} as any;
      config.headers["X-Session-Id"] = sessionId;
    }
    return {
      ...config,
      baseURL: getJavaBaseURL(),
      timeout: config.timeout || REQUEST_TIMEOUT,
    };
  },
  onResponseError: handleResponseError,
});

configPythonAxios({
  onRequest: (config: InternalAxiosRequestConfig) => {
    const sessionId = uni.getStorageSync(SESSION_KEY) || null;
    if (sessionId) {
      if (!config.headers) config.headers = {} as any;
      config.headers["X-Session-Id"] = sessionId;
    }
    return {
      ...config,
      baseURL: getPythonBaseURL(),
      timeout: config.timeout || REQUEST_TIMEOUT,
    };
  },
  onResponseError: handleResponseError,
});

export const sdkReady = true;
