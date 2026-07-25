/**
 * Vitest 全局测试环境配置
 * 在所有测试之前加载，提供 Node.js 环境下缺失的浏览器 API polyfill
 */
import { afterAll, beforeAll } from "vitest";
import { javaService } from "./src/utils/request";
import { backendProfile } from "./test/config/backend";
import { disconnectRedis, getRedis } from "./test/utils/redis";

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

// 配置后端 baseURL（Node.js 环境无浏览器 origin，需显式指定）
// 通过 TEST_BACKEND 环境变量切换 java / python / go 后端
javaService.defaults.baseURL = process.env.TEST_BASE_URL || backendProfile.baseURL;

beforeAll(async () => {
  try {
    const redis = getRedis();
    const keys = await redis.keys("captcha*");
    if (keys.length > 0) {
      await redis.del(keys);
    }
  } catch {}
});

afterAll(async () => {
  globalThis.localStorage.clear();
  await disconnectRedis();
});

export {};
