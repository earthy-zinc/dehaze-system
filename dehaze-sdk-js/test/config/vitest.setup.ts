/**
 * Vitest 测试文件级 setup（每个测试文件执行前运行）
 *
 * 职责：polyfill localStorage、配置 baseURL、触发首次登录
 * .env 由 #/config/constant 统一加载（经 utils 间接导入）；全局 Redis 清理在 globalSetup 中一次性执行
 * 每个测试文件在独立进程运行，sessionStore 仅在单文件内复用登录结果
 */
import { afterAll, beforeAll, beforeEach, onTestFinished } from "vitest";
import { service } from "@/utils/request";
import type { AxiosError, InternalAxiosRequestConfig } from "axios";
import type { CapturedRequest, CapturedResponse } from "#/config/compact-reporter";
import { clearLoginRateLimit, login } from "#/utils/auth";
import { disconnectRedis } from "#/utils/redis";
import { LocalStorageMock } from "#/utils/localstorage";
import { BACKEND_URL } from "#/config/constant";

Object.defineProperty(globalThis, "localStorage", {
  value: new LocalStorageMock(),
  writable: true,
  configurable: true,
});

service.defaults.baseURL = BACKEND_URL;

/**
 * 安全序列化请求参数/响应体，过滤函数、FormData、Stream 等不可 structuredClone 的值。
 * 避免含 FormData 的上传请求写入 task.meta 后触发 "could not be cloned" 错误。
 */
function safeClone(value: unknown): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === "function") return "[Function]";
  if (typeof value !== "object") return value;
  // FormData / ReadStream / Blob 等不可克隆对象 → 描述性字符串
  const ctorName = (value as object).constructor?.name;
  if (
    ctorName === "FormData" ||
    typeof (value as any).pipe === "function" ||
    ctorName === "ReadStream"
  ) {
    return `[${ctorName}]`;
  }
  if (typeof (value as any).byteLength === "number") return `[${ctorName ?? "Buffer"}]`;
  // 普通对象/数组 → JSON 序列化去函数
  try {
    return JSON.parse(
      JSON.stringify(value, (_k, v) => (typeof v === "function" ? "[Function]" : v))
    );
  } catch {
    return String(value);
  }
}

/**
 * 请求拦截器捕获每个请求的 method/url/params/body + 时间戳，与 transformResponse 捕获的响应
 * 按调用顺序一一对应，失败时连同响应一起写入 task.meta 供 reporter 输出，便于还原
 * 「发了什么请求、收到什么响应」。仅做只读捕获，不调用业务 onRequest，避免重复副作用。
 */
let capturedRequests: CapturedRequest[] = [];
service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    capturedRequests.push({
      method: (config.method ?? "GET").toUpperCase(),
      url: config.url ?? "",
      params: safeClone(config.params),
      body: safeClone(config.data),
      timestamp: new Date().toISOString(),
    });
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

/**
 * 通过自定义 transformResponse 在响应拦截器之前捕获完整响应体（含 code/msg/data/traceId）+ 时间戳
 * transformResponse 是 axios 管道中最早能拿到解析后 JSON 的位置，先于 request.ts 中的 response 拦截器
 * 捕获测试内全部响应（不设上限），保证断言失败涉及的那个响应不会被后续调用覆盖
 */
let capturedResponses: CapturedResponse<any>[] = [];
service.defaults.transformResponse = (rawData: any) => {
  const ts = new Date().toISOString();
  if (typeof rawData !== "string") {
    const type = rawData?.constructor?.name ?? typeof rawData;
    const size = rawData?.byteLength ?? rawData?.size ?? null;
    capturedResponses.push({
      code: "NON_TEXT",
      msg: `非文本响应: ${type}${size !== null ? `, ${size} bytes` : ""}`,
      data: null,
      traceId: "",
      timestamp: ts,
    });
    return rawData;
  }
  try {
    const parsed = JSON.parse(rawData);
    if (parsed && typeof parsed === "object" && "code" in parsed) {
      capturedResponses.push({ ...parsed, timestamp: ts } as CapturedResponse<any>);
    }
    return parsed;
  } catch {
    capturedResponses.push({
      code: "NON_JSON",
      msg: rawData.slice(0, 500),
      data: null,
      traceId: "",
      timestamp: ts,
    });
    return rawData;
  }
};

beforeAll(async () => {
  await clearLoginRateLimit();
  await login();
});

// 每个测试前清空缓存的响应体与请求体，并在测试结束时（无论通过/失败）将测试内全部 API 请求/响应
// 写入 task.meta 供 reporter 输出。用 onTestFinished 而非 onTestFailed，使 detail.log 能展示
// 全部用例（含通过）的请求/响应，用于性能分析与疑难排查
beforeEach(() => {
  capturedResponses = [];
  capturedRequests = [];
  onTestFinished((ctx) => {
    const meta = ctx.task.meta as Record<string, unknown>;
    if (capturedRequests.length > 0) {
      meta.requests = [...capturedRequests];
    }
    if (capturedResponses.length > 0) {
      meta.responses = [...capturedResponses];
    }
  });
});

afterAll(async () => {
  globalThis.localStorage.clear();
  await disconnectRedis();
});

export {};
