import { pythonConfigManager } from "@/config";
import { TOKEN_KEY } from "@/enums/CacheEnum";
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
    const accessToken = localStorage.getItem(TOKEN_KEY);
    if (accessToken) {
      config.headers.Authorization = accessToken;
    }
    const interceptors = pythonConfigManager.getInterceptors();
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
    try {
      const interceptors = pythonConfigManager.getInterceptors();
      const { code, data } = response.data;
      if (code !== ResultEnum.SUCCESS) {
        return Promise.reject(response.data);
      }
      return (await interceptors.onResponse?.(response)) || data;
    } catch (error) {
      return Promise.reject(error);
    }
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
