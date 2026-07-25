import Taro from "@tarojs/taro";
import type { AxiosError, InternalAxiosRequestConfig } from "axios";
import {
  configJavaAxios,
  configPythonAxios,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";
import { storage } from "@/utils/storage";
import { apiConfig } from "@/config/api";

function codeStartsWith(code: string | undefined, prefix: string): boolean {
  return !!code && code.startsWith(prefix);
}

let isRedirecting = false;
function redirectToLogin(): void {
  if (isRedirecting) return;
  isRedirecting = true;

  const pages = Taro.getCurrentPages();
  const currentRoute =
    pages.length > 0 ? "/" + (pages[pages.length - 1].route || "") : "";
  if (currentRoute === "/pages/login/index") {
    isRedirecting = false;
    return;
  }

  storage.clearAuth();

  Taro.redirectTo({
    url: "/pages/login/index",
    complete: () => {
      isRedirecting = false;
    },
  });
}

function onResponseError(error: AxiosError): void {
  if (error.response?.data) {
    const { code, msg } = error.response.data as ResponseData;

    if (codeStartsWith(code, "A02")) {
      if (
        code === ResultEnum.TOKEN_INVALID ||
        code === ResultEnum.TOKEN_ACCESS_FORBIDDEN ||
        code === ResultEnum.CLIENT_AUTHENTICATION_FAILED ||
        code === ResultEnum.USER_NOT_EXIST ||
        code === ResultEnum.USER_ACCOUNT_LOCKED ||
        code === ResultEnum.USER_ACCOUNT_INVALID
      ) {
        Taro.showModal({
          title: "提示",
          content: msg || "登录已失效，请重新登录",
          showCancel: false,
          confirmText: "重新登录",
          success: (res) => {
            if (res.confirm) {
              redirectToLogin();
            }
          },
        });
        return;
      }
      Taro.showToast({ title: msg || "登录异常", icon: "none" });
      return;
    }

    if (codeStartsWith(code, "A03")) {
      if (code === ResultEnum.ACCESS_UNAUTHORIZED) {
        redirectToLogin();
        return;
      }
      Taro.showToast({ title: msg || "无访问权限", icon: "none" });
      return;
    }

    if (codeStartsWith(code, "A04")) {
      Taro.showToast({ title: msg || "请求参数错误", icon: "none" });
      return;
    }

    if (codeStartsWith(code, "B")) {
      Taro.showToast({ title: msg || "系统异常，请稍后重试", icon: "none" });
      return;
    }

    Taro.showToast({ title: msg || "系统出错啦", icon: "none" });
    return;
  }

  if (error.request) {
    Taro.showToast({ title: "网络异常，请检查网络连接", icon: "none" });
    return;
  }

  Taro.showToast({ title: error.message || "请求发送失败", icon: "none" });
}

export default function configRequest(): void {
  configJavaAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      const sessionId = storage.getSessionId();
      if (sessionId) {
        if (!config.headers) config.headers = {} as any;
        config.headers["X-Session-Id"] = sessionId;
      }
      return {
        ...config,
        baseURL: apiConfig.java,
      };
    },
    onResponseError,
  });

  configPythonAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      const sessionId = storage.getSessionId();
      if (sessionId) {
        if (!config.headers) config.headers = {} as any;
        config.headers["X-Session-Id"] = sessionId;
      }
      return {
        ...config,
        baseURL: apiConfig.python,
      };
    },
    onResponseError,
  });
}
