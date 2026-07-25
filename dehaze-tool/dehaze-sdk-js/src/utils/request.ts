import { configManager } from "@/config";
import { ResultEnum } from "@/enums/ResultEnum";
import type {
  AxiosError,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
import axios from "axios";

export const service = axios.create({
  baseURL: "",
  timeout: 30000,
  withCredentials: true,
  headers: {
    "Content-Type": "application/json;charset=utf-8",
  },
});

service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const interceptors = configManager.getInterceptors();
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
    const interceptors = configManager.getInterceptors();

    if (response.config.responseType === "arraybuffer" || response.config.responseType === "blob") {
      const result = (await interceptors.onResponse?.(response)) || response.data;
      return result;
    }

    const { code, data } = response.data;
    if (code !== ResultEnum.SUCCESS) {
      const error = new Error(response.data?.msg || "Business error") as AxiosError;
      error.response = response;
      error.config = response.config;
      error.name = "AxiosError";
      error.isAxiosError = true;
      return Promise.reject(error);
    }
    const result = (await interceptors.onResponse?.(response)) || data;
    return result === null ? undefined : result;
  },
  (error: AxiosError) => {
    const interceptors = configManager.getInterceptors();
    const result = interceptors.onResponseError?.(error);
    return result !== undefined ? result : Promise.reject(error);
  }
);

export default function <T = any, R = any>(config: AxiosRequestConfig): Promise<R> {
  return service.request(config) as Promise<R>;
}
