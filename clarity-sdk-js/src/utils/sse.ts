import { generateTraceId } from "@/logger";

/** SSE 事件结构 */
export interface SSEEvent {
  /** 事件 ID（用于断线重连 Last-Event-ID） */
  id?: string;
  /** 事件类型 */
  event?: string;
  /** 事件数据（JSON 字符串） */
  data: string;
}

/**
 * 从 ReadableStream 中解析 SSE 事件。
 *
 * 遵循 SSE 协议：以空行分隔事件，每行 `field: value` 格式。
 * 支持 `id:` / `event:` / `data:` 三种字段，`data:` 多行合并。
 */
async function* parseSSEStream(
  reader: ReadableStreamDefaultReader<Uint8Array>
): AsyncGenerator<SSEEvent> {
  const decoder = new TextDecoder();
  let buffer = "";
  let currentId: string | undefined;
  let currentEvent: string | undefined;
  let dataLines: string[] = [];

  const flush = function* (): Generator<SSEEvent> {
    if (dataLines.length > 0) {
      const event: SSEEvent = { data: dataLines.join("\n") };
      if (currentId !== undefined) event.id = currentId;
      if (currentEvent !== undefined) event.event = currentEvent;
      yield event;
    }
    dataLines = [];
    currentEvent = undefined;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.replace(/\r$/, "");
      if (trimmed === "") {
        yield* flush();
      } else if (trimmed.startsWith("id:")) {
        currentId = trimmed.slice(3).trim();
      } else if (trimmed.startsWith("event:")) {
        currentEvent = trimmed.slice(6).trim();
      } else if (trimmed.startsWith("data:")) {
        dataLines.push(trimmed.slice(5).trim());
      }
    }
  }
  yield* flush();
}

/** SSE 流式请求配置 */
export interface SSERequestConfig {
  url: string;
  method: "POST" | "PUT" | "GET";
  body?: unknown;
  /** 自定义请求头 */
  headers?: Record<string, string>;
  /** 断线重连时携带的 Last-Event-ID */
  lastEventId?: string;
  /** 外部 AbortSignal，用于中断流式 */
  signal?: AbortSignal;
  /** fetch 实现（缺省用全局 fetch；供调用方注入，便于测试不触碰全局 fetch） */
  fetchImpl?: typeof fetch;
}

/** 流式入口的注入选项（当前仅 fetch 实现，供测试注入假 fetch） */
export type SSEStreamOptions = Pick<SSERequestConfig, "fetchImpl">;

/** SSE 事件回调 */
export interface SSEHandlers {
  /** 收到 SSE 事件时触发 */
  onEvent: (event: SSEEvent) => void;
  /** 流式错误（网络/HTTP 错误） */
  onError?: (error: Error) => void;
  /** 流式正常结束 */
  onClose?: () => void;
}

/**
 * SSE 请求的 HTTP 错误（响应非 2xx）。
 *
 * 后端失败响应为标准业务信封 `{code, msg, data}`（与 axios 路径同源）。仅抛
 * `status` 会丢掉 body，上层只能笼统提示「网络连接中断」，与真实失败原因
 * （如参数超长 A0400）不符；故解析出的业务码/文案随错误携带，供上层映射真实提示。
 */
export class SSEHttpError extends Error {
  /** HTTP 状态码 */
  readonly status: number;
  /** 业务错误码（响应体非业务信封时缺省） */
  readonly code?: string;
  /** 业务错误文案（响应体非业务信封时缺省） */
  readonly msg?: string;

  constructor(status: number, statusText: string, code?: string, msg?: string) {
    super(msg || `SSE request failed: ${status} ${statusText}`);
    this.name = "SSEHttpError";
    this.status = status;
    if (code !== undefined) this.code = code;
    if (msg !== undefined) this.msg = msg;
  }
}

/** 从失败响应体中提取业务码与文案（非 JSON / 非业务信封时返回空对象） */
async function readErrorEnvelope(response: Response): Promise<{ code?: string; msg?: string }> {
  try {
    const parsed: unknown = JSON.parse(await response.text());
    if (parsed && typeof parsed === "object") {
      const { code, msg } = parsed as { code?: unknown; msg?: unknown };
      return {
        ...(typeof code === "string" ? { code } : {}),
        ...(typeof msg === "string" ? { msg } : {}),
      };
    }
  } catch {
    // 网关错误页等非 JSON 响应体：无业务码，按网络错误语义向上传递
  }
  return {};
}

/**
 * 发起 SSE 流式请求并逐事件回调。
 *
 * 用于 AI 对话消息发送（POST 返回 SSE 流）和断线重连（GET）。
 * 通过 `signal` 支持外部中断（对齐后端 `/messages/{id}/stop`）。
 */
export async function fetchSSE(config: SSERequestConfig, handlers: SSEHandlers): Promise<void> {
  const traceId = generateTraceId();

  const headers: Record<string, string> = {
    Accept: "text/event-stream",
    "X-Trace-Id": traceId,
    ...(config.headers || {}),
  };

  if ((config.method === "POST" || config.method === "PUT") && config.body !== undefined) {
    headers["Content-Type"] = "application/json;charset=utf-8";
  }
  if (config.lastEventId) {
    headers["Last-Event-ID"] = config.lastEventId;
  }

  const init: RequestInit = {
    method: config.method,
    headers,
    credentials: "include",
  };
  if (config.signal) {
    init.signal = config.signal;
  }
  if ((config.method === "POST" || config.method === "PUT") && config.body !== undefined) {
    init.body = JSON.stringify(config.body);
  }

  try {
    const response = await (config.fetchImpl ?? fetch)(config.url, init);

    if (!response.ok) {
      // 失败响应体为标准业务信封，解析后随错误抛出，避免上层只能报「网络错误」
      const envelope = await readErrorEnvelope(response);
      throw new SSEHttpError(response.status, response.statusText, envelope.code, envelope.msg);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/event-stream")) {
      // 非流式响应（如非流式消息模式直接返回 JSON）
      const text = await response.text();
      handlers.onEvent({ event: "message", data: text });
      handlers.onClose?.();
      return;
    }

    const reader = response.body!.getReader();
    for await (const event of parseSSEStream(reader)) {
      if (config.signal?.aborted) break;
      handlers.onEvent(event);
    }
    handlers.onClose?.();
  } catch (error) {
    if (config.signal?.aborted) {
      handlers.onClose?.();
    } else {
      handlers.onError?.(error as Error);
    }
  }
}
