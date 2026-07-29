/**
 * Vitest 测试文件级 setup（每个测试文件执行前运行）
 *
 * 职责：加载 .env、polyfill localStorage、配置 baseURL、触发首次登录
 * 全局 Redis 清理在 globalSetup 中一次性执行，此处不再重复
 * 每个测试文件在独立进程运行，sessionStore 仅在单文件内复用登录结果
 */
import dotenv from "dotenv";
import path from "path";
import { afterAll, beforeAll } from "vitest";
import { service } from "@/utils/request";
import { clearLoginRateLimit, login } from "#/utils/auth";
import { disconnectRedis } from "#/utils/redis";

// quiet: 抑制 dotenv v17 的 injected env 提示日志，保持测试输出为纯 JSON
dotenv.config({ path: path.resolve(__dirname, "../../.env"), quiet: true });

class LocalStorageMock {
  private store: Record<string, string> = {};

  getItem(key: string): string | null {
    return this.store[key] ?? null;
  }

  setItem(key: string, value: string): void {
    this.store[key] = value;
  }

  removeItem(key: string): void {
    delete this.store[key];
  }

  clear(): void {
    this.store = {};
  }

  get length(): number {
    return Object.keys(this.store).length;
  }

  key(index: number): string | null {
    const keys = Object.keys(this.store);
    return keys[index] ?? null;
  }
}

const localStorageInstance = new LocalStorageMock();

Object.defineProperty(globalThis, "localStorage", {
  value: localStorageInstance,
  writable: true,
  configurable: true,
});

service.defaults.baseURL = process.env.BACKEND_URL || "http://127.0.0.1:8989";

beforeAll(async () => {
  await clearLoginRateLimit();
  await login();
});

afterAll(async () => {
  globalThis.localStorage.clear();
  await disconnectRedis();
});

export {};
