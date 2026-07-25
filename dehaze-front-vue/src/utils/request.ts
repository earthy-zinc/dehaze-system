import { useUserStoreHook } from "@/store/modules/user";
import { clearAccessToken, getAccessToken } from "@/utils/auth";

import type { AxiosInstance, InternalAxiosRequestConfig } from "axios";
import {
  configJavaAxios,
  configPythonAxios,
  javaService,
  pythonService,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";

let isRefreshing = false;
let pendingQueue: Array<{
  service: AxiosInstance;
  config: InternalAxiosRequestConfig;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}> = [];

function showReloginDialog() {
  ElMessageBox.confirm("当前页面已失效，请重新登录", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  }).then(() => {
    const userStore = useUserStoreHook();
    userStore.resetToken();
    location.reload();
  });
}

function handleTokenInvalid(error: any, service: AxiosInstance): Promise<any> {
  const userStore = useUserStoreHook();

  if (isRefreshing) {
    return new Promise((resolve, reject) => {
      pendingQueue.push({ service, config: error.config!, resolve, reject });
    });
  }

  isRefreshing = true;
  return userStore
    .refreshAccessToken()
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
      showReloginDialog();
      return Promise.reject(error);
    })
    .finally(() => {
      isRefreshing = false;
    });
}

function createOnResponseError(service: AxiosInstance) {
  return (error: any) => {
    if (error.response?.data) {
      const { code, msg } = error.response.data as ResponseData;
      if (code === ResultEnum.TOKEN_INVALID) {
        return handleTokenInvalid(error, service);
      }
      ElMessage.error(msg || "系统出错");
    } else if (error.request) {
      ElMessage.error("网络异常，请检查网络连接");
    } else {
      ElMessage.error(error.message || "请求发送失败");
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
