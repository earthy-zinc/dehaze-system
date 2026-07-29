/**
 * Vitest 测试文件级 setup（每个测试文件执行前运行）
 *
 * 职责：加载 .env、polyfill localStorage、配置 baseURL、触发首次登录
 * 不再清理 Redis 缓存（已迁移到 vitest.globalSetup.ts 一次性执行）
 *
 * 设计说明：
 * - 全局 Redis 清理在 globalSetup 中完成，整个测试运行只清理一次
 * - sessionStore 在内存中跨文件复用，跨文件不再重复登录
 * - 单个测试文件内的登录切换会命中 sessionStore 缓存，不会触发限流
 */
import dotenv from "dotenv";
import path from "path";
import { afterAll, beforeAll } from "vitest";
import { service } from "@/utils/request";
import { clearLoginRateLimit, login } from "#/utils/auth";
import { disconnectRedis } from "#/utils/redis";

dotenv.config({ path: path.resolve(__dirname, "../../.env") });

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
