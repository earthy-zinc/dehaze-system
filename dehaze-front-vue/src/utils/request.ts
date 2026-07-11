import { REFRESH_TOKEN_KEY } from "@/enums/CacheEnum";
import { useUserStoreHook } from "@/store/modules/user";

import type { AxiosInstance, InternalAxiosRequestConfig } from "axios";
import {
  configJavaAxios,
  configPythonAxios,
  javaService,
  pythonService,
  ResponseData,
  ResultEnum,
} from "dehaze-sdk-js";

// token 刷新状态：是否正在刷新中
let isRefreshing = false;
// 等待 token 刷新完成后重发的请求队列
let pendingQueue: Array<{
  service: AxiosInstance;
  config: InternalAxiosRequestConfig;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}> = [];

/**
 * 显示重新登录弹框
 */
function showReloginDialog() {
  ElMessageBox.confirm("当前页面已失效，请重新登录", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  }).then(() => {
    const userStore = useUserStoreHook();
    userStore.resetToken().then(() => {
      location.reload();
    });
  });
}

/**
 * 处理 token 失效：尝试刷新 token 后重发请求，刷新失败则弹框重新登录
 *
 * @param error 原始错误
 * @param service 对应的 axios 实例（用于重发请求）
 */
function handleTokenInvalid(
  error: any,
  service: AxiosInstance
): Promise<any> {
  const userStore = useUserStoreHook();
  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);

  // 无 refreshToken，直接弹框重新登录
  if (!refreshToken) {
    showReloginDialog();
    return Promise.reject(error.message);
  }

  // 正在刷新中，将当前请求加入队列等待
  if (isRefreshing) {
    return new Promise((resolve, reject) => {
      pendingQueue.push({ service, config: error.config!, resolve, reject });
    });
  }

  // 发起刷新
  isRefreshing = true;
  return userStore
    .refreshAccessToken(refreshToken)
    .then(() => {
      // 刷新成功，重发队列中等待的请求
      pendingQueue.forEach(({ service: svc, config, resolve, reject }) => {
        svc.request(config).then(resolve).catch(reject);
      });
      pendingQueue = [];
      // 重发原始请求
      return service.request(error.config!);
    })
    .catch((err) => {
      // 刷新失败，清空队列并弹框重新登录
      pendingQueue.forEach(({ reject }) => reject(err));
      pendingQueue = [];
      showReloginDialog();
      return Promise.reject(error.message);
    })
    .finally(() => {
      isRefreshing = false;
    });
}

/**
 * 创建响应错误处理函数（绑定到具体的 axios 实例，用于刷新后重发）
 */
function createOnResponseError(service: AxiosInstance) {
  return (error: any) => {
    if (error.response?.data) {
      const { code, msg } = error.response.data as ResponseData;
      if (code === ResultEnum.TOKEN_INVALID) {
        return handleTokenInvalid(error, service);
      }
      ElMessage.error(msg || "系统出错");
    }
    return Promise.reject(error.message);
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
  });
  configPythonAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      return {
        ...config,
        baseURL: import.meta.env.VITE_PYTHON_BASE_API,
      };
    },
    onResponseError: createOnResponseError(pythonService),
  });
}
