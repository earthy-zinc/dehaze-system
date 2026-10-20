import "axios";

/**
 * AxiosConfig.metadata 中由 SDK 注入的请求级上下文（trace_id / startTime）。
 * 字段均为可选：traceId 与 startTime 在不同请求拦截器中分两次写入，需支持增量合并。
 */
export interface RequestMetadata {
  traceId?: string;
  startTime?: number;
}

declare module "axios" {
  interface InternalAxiosRequestConfig {
    metadata?: RequestMetadata;
  }
}
