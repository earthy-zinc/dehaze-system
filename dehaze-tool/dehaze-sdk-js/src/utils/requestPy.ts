import { pythonConfigManager } from "@/config";
import { ResultEnum } from "@/enums/ResultEnum";
import type { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from "axios";
import axios from "axios";

const service = axios.create({
  baseURL: "",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json;charset=utf-8",
  },
});

service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const interceptors = pythonConfigManager.getInterceptors();
    const accessToken = interceptors.getToken?.();
    if (accessToken) {
      config.headers.Authorization = accessToken;
    }
    const otherConfig = interceptors.onRequest?.(config) || {};
    return { ...config, ...otherConfig };
  },
  (error: AxiosError) => {
    const interceptors = pythonConfigManager.getInterceptors();
    return Promise.reject(interceptors.onRequestError?.(error) || error);
  }
);

service.interceptors.response.use(
  async (response: AxiosResponse) => {
    const interceptors = pythonConfigManager.getInterceptors();
    const { code, data } = response.data;
    if (code !== ResultEnum.SUCCESS) {
      // 构造模拟 AxiosError，让 onResponseError 能访问 response.data
      const error = new Error(response.data?.msg || "Business error") as AxiosError;
      error.response = response;
      error.config = response.config;
      error.name = "AxiosError";
      error.isAxiosError = true;
      return Promise.reject(error);
    }
    return (await interceptors.onResponse?.(response)) || data;
  },
  (error: AxiosError) => {
    const interceptors = pythonConfigManager.getInterceptors();
    const result = interceptors.onResponseError?.(error);
    return result !== undefined ? result : Promise.reject(error);
  }
);

// 导出 axios 实例
export const pythonService = service;
export default service;
