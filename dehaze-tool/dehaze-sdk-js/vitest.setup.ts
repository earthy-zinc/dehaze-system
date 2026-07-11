/**
 * Vitest 全局测试环境配置
 * 在所有测试之前加载，提供 Node.js 环境下缺失的浏览器 API polyfill
 */
import { afterAll, beforeEach, vi } from "vitest";
import { javaService } from "./src/utils/request";

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

// 配置 Java 后端 baseURL（Node.js 环境无浏览器 origin 兜底，需显式指定）
javaService.defaults.baseURL = process.env.JAVA_BASE_URL || "http://127.0.0.1:8989";

// 清理所有 mock
beforeEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  globalThis.localStorage.clear();
});

export {};
