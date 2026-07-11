import { configManager } from "@/config";
import { TOKEN_KEY } from "@/enums/CacheEnum";
import { ResultEnum } from "@/enums/ResultEnum";
import type {
  AxiosError,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
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
      // 如果 token 不包含 Bearer 前缀，则添加
      config.headers.Authorization = accessToken.startsWith("Bearer ")
        ? accessToken
        : `Bearer ${accessToken}`;
    }

    const interceptors = configManager.getInterceptors();
    // 调用自定义拦截器，直接使用返回的 config（如果有的话）
    const modifiedConfig = interceptors.onRequest?.(config);
    return modifiedConfig || config;
  },
  (error: AxiosError) => {
    const interceptors = configManager.getInterceptors();
    return Promise.reject(interceptors.onRequestError?.(error) || error);
  }
);

service.interceptors.response.use(
  async (response: AxiosResponse) => {
    try {
      const interceptors = configManager.getInterceptors();

      // 处理 arraybuffer 响应类型（如文件下载、导出等）
      if (response.config.responseType === "arraybuffer") {
        const result = (await interceptors.onResponse?.(response)) || response.data;
        return result;
      }

      const { code, data } = response.data;
      if (code !== ResultEnum.SUCCESS) {
        return Promise.reject(response.data);
      }
      const result = (await interceptors.onResponse?.(response)) || data;
      return result;
    } catch (error) {
      return Promise.reject(error);
    }
  },
  (error: AxiosError) => {
    const interceptors = configManager.getInterceptors();
    const result = interceptors.onResponseError?.(error);
    return result !== undefined ? result : Promise.reject(error);
  }
);

// 封装请求函数，正确处理泛型类型
// 响应拦截器已经返回 response.data.data，所以这里直接返回 R 类型
function request<T = any, R = any>(config: AxiosRequestConfig): Promise<R> {
  return service.request(config) as Promise<R>;
}

export const javaService = service;
export default request;
