import { configManager } from "@/config";
import { ResultEnum } from "@/enums/ResultEnum";
import { Logger } from "@/logger";
import type { LogEntry } from "@/logger";
import { ResponseData } from "@/types";
import type {
  AxiosError,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
import axios from "axios";

const TRACE_HEADER = "X-Trace-Id";

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
    // 生成/复用 trace_id 注入请求头，供后端 TraceIdFilter 透传，端到端串联
    const logger = Logger.getInstance();
    if (logger) {
      config.headers.set(TRACE_HEADER, logger.ensureTraceId());
    }
    const interceptors = configManager.getInterceptors();
    const modifiedConfig = interceptors.onRequest?.(config);
    return modifiedConfig || config;
  },
  (error: AxiosError) => {
    const interceptors = configManager.getInterceptors();
    return Promise.reject(interceptors.onRequestError?.(error) || error);
  }
);

const SLOW_REQUEST_THRESHOLD_MS = 3000;

service.interceptors.response.use(
  async (response: AxiosResponse) => {
    const interceptors = configManager.getInterceptors();

    // 读取后端回写的 X-Trace-Id，与本地 trace_id 对齐
    alignResponseTraceId(response);
    // 慢请求告警：成功路径耗时超过阈值时 WARN 上报（失败路径已由 reportApiError 上报）
    reportSlowRequest(response.config);

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
    // 读取失败响应头 X-Trace-Id 对齐（对齐逻辑在下方 onResponseError 中统一完成）
    const interceptors = configManager.getInterceptors();
    const result = interceptors.onResponseError?.(error);
    reportApiError(error);
    return result !== undefined ? result : Promise.reject(error);
  }
);

/**
 * 读取响应头 X-Trace-Id，与本地 trace_id 对齐（成功与失败响应均调用）。
 */
function alignResponseTraceId(response: AxiosResponse): void {
  const logger = Logger.getInstance();
  if (!logger) return;
  const header = response.headers?.["x-trace-id"];
  logger.alignTraceId(
    typeof header === "string" ? header : Array.isArray(header) ? (header[0] ?? "") : undefined
  );
}

/**
 * API 失败自动构造日志条目（method/path/status/duration/code）交 Logger 上报。
 * 触发时机：响应拦截器 reject 路径（业务码错误或 HTTP 错误）。
 */
function reportApiError(error: AxiosError): void {
  const logger = Logger.getInstance();
  if (!logger) return;

  const config = error.config;
  const response = error.response;
  // 对齐失败响应的 trace_id（若成功路径已对齐则此处为空更新）
  if (response) {
    alignResponseTraceId(response);
  }

  const method = (config?.method ?? "GET").toUpperCase();
  const path = config?.url ? config.url.split("?")[0]! : "";
  const status = response?.status;
  const code = (response?.data as ResponseData<any> | undefined)?.code;
  const duration = requestDuration(config);

  const fields: Partial<LogEntry> = { method, path };
  if (status !== undefined) fields.status = status;
  if (code !== undefined) fields.code = code;
  if (duration !== undefined) fields.duration = duration;

  // error.message 携带关键诊断：业务错误为后端 msg，网络错误为 Network Error / timeout 描述
  logger.error(`API_ERROR: ${error.message}`, fields);
}

/** 从请求拦截器记录的 startTime 计算请求耗时（毫秒），无记录时返回 undefined */
function requestDuration(config: InternalAxiosRequestConfig | undefined): number | undefined {
  const startTime = (config as unknown as { metadata?: { startTime?: number } })?.metadata
    ?.startTime;
  return startTime !== undefined ? Date.now() - startTime : undefined;
}

/**
 * 成功路径慢请求告警：耗时超过阈值时 WARN 上报（method/path/duration）。
 * 失败路径的耗时告警统一由 reportApiError 的 ERROR 日志覆盖，不重复上报。
 */
function reportSlowRequest(config: InternalAxiosRequestConfig): void {
  const logger = Logger.getInstance();
  if (!logger) return;
  const duration = requestDuration(config);
  if (duration === undefined || duration <= SLOW_REQUEST_THRESHOLD_MS) return;
  const method = (config.method ?? "GET").toUpperCase();
  const path = config.url ? config.url.split("?")[0]! : "";
  logger.warn("SLOW_REQUEST", { method, path, duration });
}

/**
 * 记录请求开始时间到 config.metadata，供错误拦截器计算 duration。
 * 在请求拦截器链中尽早注入。
 */
service.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  (config as unknown as { metadata: { startTime: number } }).metadata = {
    startTime: Date.now(),
  };
  return config;
});

export default function <R = any>(config: AxiosRequestConfig): Promise<R> {
  return service.request(config) as Promise<R>;
}
