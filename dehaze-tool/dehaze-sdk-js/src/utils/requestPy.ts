import { pythonConfigManager } from "@/config";
import { TOKEN_KEY } from "@/enums/CacheEnum";
import { ResultEnum } from "@/enums/ResultEnum";
import axios from "axios";
import type { AxiosResponse, InternalAxiosRequestConfig } from "axios";

const service = axios.create({
  baseURL: "http://localhost:5000",
  timeout: 5000,
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
  (error: any) => {
    const interceptors = pythonConfigManager.getInterceptors();
    return Promise.reject(interceptors.onRequestError?.(error) || error);
  }
);

service.interceptors.response.use(
  async (response: AxiosResponse) => {
    try {
      const interceptors = pythonConfigManager.getInterceptors();
      return (await interceptors.onResponse?.(response.data)) || response.data;
    } catch (error) {
      return Promise.reject(error);
    }
  },
  (error: any) => {
    const interceptors = pythonConfigManager.getInterceptors();
    return Promise.reject(interceptors.onResponseError?.(error) || error);
  }
);

// 导出 axios 实例
export default service;
