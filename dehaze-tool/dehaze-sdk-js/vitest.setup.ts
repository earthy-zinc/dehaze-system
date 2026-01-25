/**
 * Vitest 全局测试环境配置
 * 在所有测试之前加载，提供 Node.js 环境下缺失的浏览器 API polyfill
 */
import { afterAll, beforeEach, vi } from "vitest";

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

// 清理所有 mock
beforeEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  globalThis.localStorage.clear();
});

export {};
