/**
 * Logger 模块测试公共工具：内存存储 + 捕获 transport。
 *
 * Logger/Performance 为纯前端逻辑，不依赖后端，故不复用集成测试的 setup 设施
 * （集成测试 vitest.setup.ts 会强制登录后端）。这里仅提供隔离用的 storage/transport。
 */
import type { LogEntry, LoggerStorage, LogTransport } from "@/logger";
import { Logger } from "@/logger";

/** 内存存储：隔离测试间状态，规避 globalThis.localStorage 共享导致串扰 */
export class MemoryStore implements LoggerStorage {
  private store = new Map<string, string>();
  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.store.set(key, value);
  }
  removeItem(key: string): void {
    this.store.delete(key);
  }
}

/** 捕获 send 调用的测试 transport；failTimes > 0 时模拟上报失败 */
export class MockTransport implements LogTransport {
  sentBatches: LogEntry[][] = [];

  constructor(public failTimes = 0) {}

  log(entry: LogEntry): void {
    this.sentBatches.push([entry]);
  }

  async send(logs: LogEntry[]): Promise<void> {
    if (this.failTimes > 0) {
      this.failTimes--;
      throw new Error("upload failed");
    }
    this.sentBatches.push(logs);
  }
}

/** 仅捕获 log 输出、不批量上报的 transport（用于观测采样前的本地输出） */
export class CaptureTransport implements LogTransport {
  logs: LogEntry[] = [];
  log(entry: LogEntry): void {
    this.logs.push(entry);
  }
}

/** 读取 Logger 持久化队列 */
export function persistedQueue(): LogEntry[] {
  const logger = Logger.getInstance();
  if (!logger) return [];
  const raw = (logger as any).storage.getItem("dehaze_logs");
  return raw ? (JSON.parse(raw) as LogEntry[]) : [];
}

/** 等待 in-flight 异步 flush 结算，避免断言时序竞态 */
export function settleFlush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
