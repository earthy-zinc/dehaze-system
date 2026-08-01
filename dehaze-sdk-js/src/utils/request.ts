import { configManager } from "@/config";
import { ResultEnum } from "@/enums/ResultEnum";
import { ResponseData } from "@/types";
import type {
  AxiosError,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
import axios from "axios";

export const service = axios.create({
  baseURL: "",
  timeout: 120000,
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
      const contentTypeRaw = response.headers["content-type"];
      const contentType =
        typeof contentTypeRaw === "string"
          ? contentTypeRaw
          : Array.isArray(contentTypeRaw)
            ? (contentTypeRaw[0] ?? "")
            : "";
      if (contentType.includes("application/json")) {
        let text: string;
        if (response.data instanceof Blob) {
          text = await response.data.text();
        } else if (typeof response.data === "string") {
          text = response.data;
        } else if (Buffer.isBuffer(response.data)) {
          text = response.data.toString("utf-8");
        } else {
          text = JSON.stringify(response.data);
        }
        const parsed = JSON.parse(text);
        if (parsed.code !== ResultEnum.SUCCESS) {
          const error = new Error(parsed?.msg || "Business error") as AxiosError;
          error.response = response;
          error.config = response.config;
          error.name = "AxiosError";
          error.isAxiosError = true;
          return Promise.reject(error);
        }
        const result =
          (await interceptors.onResponse?.({ ...response, data: parsed })) || parsed.data;
        return result === null ? undefined : result;
      }
      const data =
        response.data instanceof Blob
          ? response.data
          : new Blob([response.data], { type: contentType });
      const result = (await interceptors.onResponse?.({ ...response, data })) || data;
      return result;
    }

    const { code, data } = response.data as ResponseData<any>;
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

export default function <R = any>(config: AxiosRequestConfig): Promise<R> {
  return service.request(config) as Promise<R>;
}
