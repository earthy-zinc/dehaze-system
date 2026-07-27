/**
 * Vitest 全局测试环境配置
 * 在所有测试之前加载，提供 Node.js 环境下缺失的浏览器 API polyfill
 */
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { afterAll, beforeAll } from "vitest";
import { service } from "./src/utils/request";
import { login } from "./test/utils/auth";
import { disconnectRedis, getRedis } from "./test/utils/redis";

// 根据 TEST_BACKEND 加载对应的 .env.{type} 文件
// 系统环境变量优先级高于 .env 文件（dotenv 默认 override:false）
const backend = process.env.TEST_BACKEND || "java";
const envFile = path.resolve(__dirname, `.env.${backend}`);
if (fs.existsSync(envFile)) {
  dotenv.config({ path: envFile });
}

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
service.defaults.baseURL = process.env.BACKEND_URL || "http://127.0.0.1:8989";

beforeAll(async () => {
  try {
    const redis = getRedis();
    const keys = await redis.keys("captcha*");
    if (keys.length > 0) {
      await redis.del(keys);
    }
  } catch {}
  await login();
});

afterAll(async () => {
  globalThis.localStorage.clear();
  await disconnectRedis();
});

export {};
