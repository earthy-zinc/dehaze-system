import { configManager } from "@/config";
import { TOKEN_KEY } from "@/enums/CacheEnum";
import { ResultEnum } from "@/enums/ResultEnum";
import axios from "axios";
import type { AxiosResponse, InternalAxiosRequestConfig } from "axios";

const service = axios.create({
  baseURL: "http://localhost:8989",
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
    const interceptors = configManager.getInterceptors();
    const otherConfig = interceptors.onRequest?.(config) || {};
    return { ...config, ...otherConfig };
  },
  (error: any) => {
    const interceptors = configManager.getInterceptors();
    return Promise.reject(interceptors.onRequestError?.(error) || error);
  }
);

service.interceptors.response.use(
  async (response: AxiosResponse) => {
    try {
      const interceptors = configManager.getInterceptors();
      return (await interceptors.onResponse?.(response.data)) || response.data;
    } catch (error) {
      return Promise.reject(error);
    }
  },
  (error: any) => {
    const interceptors = configManager.getInterceptors();
    return Promise.reject(interceptors.onResponseError?.(error) || error);
  }
);

export default service;
