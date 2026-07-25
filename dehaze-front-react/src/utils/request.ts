import {
  configJavaAxios,
  configPythonAxios,
  javaService,
  pythonService,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";
import { getAccessToken } from "@/utils/auth";

import type { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from "axios";

import store from "@/store";
import { refreshAccessToken, resetToken } from "@/store/modules/userSlice";

import { message, Modal } from "antd";

let isRefreshing = false;
let pendingQueue: Array<{
  service: AxiosInstance;
  config: InternalAxiosRequestConfig;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}> = [];

function showReloginModal() {
  Modal.confirm({
    title: "提示",
    content: "登录已失效，请重新登录",
    onOk() {
      store.dispatch(resetToken());
      window.location.href = `/login?redirect=${encodeURIComponent(window.location.href)}`;
    },
    onCancel() {
      store.dispatch(resetToken());
    },
  });
}

function handleTokenInvalid(
  error: AxiosError,
  service: AxiosInstance
): Promise<any> {
  if (isRefreshing) {
    return new Promise((resolve, reject) => {
      pendingQueue.push({
        service,
        config: error.config!,
        resolve,
        reject,
      });
    });
  }

  isRefreshing = true;
  return store
    .dispatch(refreshAccessToken())
    .unwrap()
    .then(() => {
      pendingQueue.forEach(({ service: svc, config, resolve, reject }) => {
        svc.request(config).then(resolve).catch(reject);
      });
      pendingQueue = [];
      return service.request(error.config!);
    })
    .catch((err) => {
      pendingQueue.forEach(({ reject }) => reject(err));
      pendingQueue = [];
      showReloginModal();
      return Promise.reject(error);
    })
    .finally(() => {
      isRefreshing = false;
    });
}

function createOnResponseError(service: AxiosInstance) {
  return (error: AxiosError) => {
    if (error.response?.data) {
      const { code, msg } = error.response.data as ResponseData;
      if (code === ResultEnum.TOKEN_INVALID) {
        return handleTokenInvalid(error, service);
      }
      message.error(msg || "系统出错");
    } else if (error.request) {
      message.error("网络异常，请检查网络连接");
    } else {
      message.error(error.message || "请求发送失败");
    }
    return Promise.reject(error);
  };
}

export default function configRequest() {
  configJavaAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_JAVA_BASE_API,
      };
    },
    onResponseError: createOnResponseError(javaService),
    getToken: () => getAccessToken(),
  });
  configPythonAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_PYTHON_BASE_API,
      };
    },
    onResponseError: createOnResponseError(pythonService),
    getToken: () => getAccessToken(),
  });
}
