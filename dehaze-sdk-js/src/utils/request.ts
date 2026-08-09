import { configManager } from "@/config";
import { ResultEnum } from "@/enums/ResultEnum";
import { generateTraceId, Logger } from "@/logger";
import type { LogEntry } from "@/logger";
import { ResponseData } from "@/types";
import type { RequestMetadata } from "@/types/global";
import type {
  AxiosError,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
import axios from "axios";

const TRACE_HEADER = "X-Trace-Id";
const SLOW_REQUEST_THRESHOLD_MS = 3000;

/** 向 config.metadata 合并写入字段（响应阶段覆盖 traceId 时必须合并，否则会丢失请求阶段写入的 startTime） */
function setMetadata(config: InternalAxiosRequestConfig, patch: Partial<RequestMetadata>): void {
  config.metadata = { ...config.metadata, ...patch };
}

/** 构造业务码错误（非 HTTP 错误），伪装成 AxiosError 以便统一进入 reportApiError 上报链路 */
function buildBizError(message: string | undefined, response: AxiosResponse): AxiosError {
  const error = new Error(message || "Business error") as AxiosError;
  error.response = response;
  error.config = response.config;
  error.name = "AxiosError";
  error.isAxiosError = true;
  return error;
}

/**
 * 读取响应头 X-Trace-Id，对齐到本次请求上下文（config.metadata.traceId）。
 * 后端可能生成新的 trace_id 透传，此时以响应头为准（后端日志用的是后端生成的 trace_id）。
 * 不再写入全局变量，避免并发请求间互相覆盖。
 */
function alignResponseTraceId(response: AxiosResponse): void {
  const header = response.headers?.["x-trace-id"];
  let traceId: string;
  if (typeof header === "string") {
    traceId = header;
  } else if (Array.isArray(header)) {
    traceId = header[0] ?? "";
  } else {
    traceId = "";
  }
  if (traceId) {
    setMetadata(response.config, { traceId });
  }
}

/**
 * API 失败自动构造日志条目（method/path/status/duration/code）交 Logger 上报。
 * trace_id 从 config.metadata.traceId 读取，显式传入 fields.trace_id。
 * 触发时机：响应拦截器 reject 路径（业务码错误或 HTTP 错误）。
 */
function reportApiError(error: AxiosError): void {
  const logger = Logger.getInstance();
  if (!logger) return;

  const config = error.config;
  const response = error.response;
  // 对齐失败响应的 trace_id 到本次请求上下文
  if (response) {
    alignResponseTraceId(response);
  }

  const method = (config?.method ?? "GET").toUpperCase();
  const path = config?.url ? config.url.split("?")[0]! : "";
  const status = response?.status;
  const code = (response?.data as ResponseData<any> | undefined)?.code;
  const duration = requestDuration(config);
  const traceId = config?.metadata?.traceId ?? "";

  const fields: Partial<LogEntry> = { method, path, trace_id: traceId };
  if (status !== undefined) fields.status = status;
  if (code !== undefined) fields.code = code;
  if (duration !== undefined) fields.duration = duration;

  // error.message 携带关键诊断：业务错误为后端 msg，网络错误为 Network Error / timeout 描述
  logger.error(`API_ERROR: ${error.message}`, fields);
}

/** 从请求拦截器记录的 startTime 计算请求耗时（毫秒），无记录时返回 undefined */
function requestDuration(config: InternalAxiosRequestConfig | undefined): number | undefined {
  const startTime = config?.metadata?.startTime;
  return startTime !== undefined ? Date.now() - startTime : undefined;
}

/**
 * 成功路径慢请求告警：耗时超过阈值时 WARN 上报（method/path/duration）。
 * trace_id 从 config.metadata.traceId 读取，显式传入 fields.trace_id。
 * 失败路径的耗时告警统一由 reportApiError 的 ERROR 日志覆盖，不重复上报。
 */
function reportSlowRequest(config: InternalAxiosRequestConfig): void {
  const logger = Logger.getInstance();
  if (!logger) return;
  const duration = requestDuration(config);
  if (duration === undefined || duration <= SLOW_REQUEST_THRESHOLD_MS) return;
  const method = (config.method ?? "GET").toUpperCase();
  const path = config.url ? config.url.split("?")[0]! : "";
  const traceId = config?.metadata?.traceId ?? "";
  logger.warn("SLOW_REQUEST", { method, path, duration, trace_id: traceId });
}

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
    // 生成请求级 trace_id，写入请求头（供后端 TraceIdFilter 透传）与 config.metadata（供日志串联）。
    // 不写入全局变量：并发请求时全局变量会被覆盖，导致非请求日志 trace_id 串号。
    // startTime 同步注入：供响应拦截器计算 duration，越早记录越准确。
    const traceId = generateTraceId();
    config.headers.set(TRACE_HEADER, traceId);
    setMetadata(config, { traceId, startTime: Date.now() });

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

    // 读取后端回写的 X-Trace-Id，对齐到本次请求上下文（config.metadata.traceId）
    alignResponseTraceId(response);
    // 慢请求告警：成功路径耗时超过阈值时 WARN 上报（失败路径已由 reportApiError 上报）
    reportSlowRequest(response.config);

    if (response.config.responseType === "arraybuffer" || response.config.responseType === "blob") {
      const contentTypeRaw = response.headers["content-type"];
      let contentType: string;
      if (typeof contentTypeRaw === "string") {
        contentType = contentTypeRaw;
      } else if (Array.isArray(contentTypeRaw)) {
        contentType = contentTypeRaw[0] ?? "";
      } else {
        contentType = "";
      }
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
          const error = buildBizError(parsed?.msg, response);
          // 业务码错误在 fulfilled handler 中 reject，不会触发同一拦截器的 rejected handler，
          // 需在此显式上报，否则 API 失败日志丢失
          reportApiError(error);
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
      const error = buildBizError(response.data?.msg, response);
      // 业务码错误在 fulfilled handler 中 reject，不会触发同一拦截器的 rejected handler，
      // 需在此显式上报，否则 API 失败日志丢失
      reportApiError(error);
      return Promise.reject(error);
    }
    const result = (await interceptors.onResponse?.(response)) || data;
    return result === null ? undefined : result;
  },
  (error: AxiosError) => {
    const interceptors = configManager.getInterceptors();
    const result = interceptors.onResponseError?.(error);
    reportApiError(error);
    return result !== undefined ? result : Promise.reject(error);
  }
);

export default function <R = any>(config: AxiosRequestConfig): Promise<R> {
  return service.request(config) as Promise<R>;
}
