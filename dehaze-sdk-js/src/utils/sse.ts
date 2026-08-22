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
}

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
    const response = await fetch(config.url, init);

    if (!response.ok) {
      throw new Error(`SSE request failed: ${response.status} ${response.statusText}`);
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
