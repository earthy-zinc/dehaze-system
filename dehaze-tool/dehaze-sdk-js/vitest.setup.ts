/**
 * Vitest 全局测试环境配置
 * 在所有测试之前加载，提供 Node.js 环境下缺失的浏览器 API polyfill
 */
import { afterAll, beforeAll, beforeEach, vi } from "vitest";
import { execSync } from "child_process";
import { javaService } from "./src/utils/request";
import { backendProfile } from "./test/config/backend";

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

/**
 * 每个测试文件开始前清理 Redis 缓存，确保测试隔离性
 * 集成测试共享同一后端实例，前一个测试文件创建/删除的数据可能残留在 Redis 缓存中
 * （如 @Cacheable 的 dataset:all、dict:options:* 等），导致后续测试读到脏数据
 *
 * 注意: 三端验证码均存于 Redis db0，按后端选择对应 DB 清理
 */
beforeAll(async () => {
  try {
    execSync(
      `docker exec ${backendProfile.redisContainer} redis-cli -a ${backendProfile.redisPassword} -n ${backendProfile.captchaRedisDB} FLUSHDB`,
      { stdio: "pipe", timeout: 5000 }
    );
  } catch {
    // Redis 容器未运行时忽略，由各测试文件的 login() 单独处理连接错误
  }
});

// 清理所有 mock
beforeEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  globalThis.localStorage.clear();
});

export {};
