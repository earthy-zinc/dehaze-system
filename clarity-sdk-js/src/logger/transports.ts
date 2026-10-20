import type { LogEntry, LogTransport } from "./types";

const MAX_BATCH = 50;

/** 基础字段不重复输出到 extras（error_stack 单独作为第三参数） */
const BASE_KEYS = new Set([
  "timestamp",
  "level",
  "message",
  "service",
  "app",
  "app_version",
  "url",
  "user_agent",
  "trace_id",
  "error_stack",
]);

/** 开发环境 transport：逐条输出到 console（DevTools 接管，不落盘） */
export class ConsoleTransport implements LogTransport {
  log(entry: LogEntry): void {
    const tag = `[dehaze][${entry.level}]`;
    // 附加字段（method/path/status/duration 等）一并输出，否则网络故障时只剩 message 无法定位
    const extras = Object.entries(entry)
      .filter(([k, v]) => !BASE_KEYS.has(k) && v !== undefined && v !== "")
      .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join(" ");
    const message = `${entry.message}${extras ? ` ${extras}` : ""} trace_id=${entry.trace_id}`;
    if (entry.level === "ERROR") {
      console.error(tag, message, entry.error_stack ?? "");
    } else if (entry.level === "WARN") {
      console.warn(tag, message);
    } else {
      console.info(tag, message);
    }
  }

  async send(): Promise<void> {
    // ConsoleTransport 不走批量上报
  }
}

/** 生产环境 transport：批量 POST 上报后端接收 API */
export class RemoteTransport implements LogTransport {
  constructor(private readonly endpoint: string = "/api/v1/logs/client") {}

  log(): void {
    // 生产环境不在 console 逐条刷屏
  }

  async send(logs: LogEntry[]): Promise<void> {
    const body = { logs: logs.slice(0, MAX_BATCH) };
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json;charset=utf-8" },
      body: JSON.stringify(body),
      keepalive: true,
    });
    if (!response.ok) {
      throw new Error(`remote log upload failed: ${response.status}`);
    }
  }
}
