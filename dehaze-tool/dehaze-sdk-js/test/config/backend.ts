/**
 * 后端配置 - 支持在 Java / Python / Go 三个后端之间切换测试
 *
 * 通过环境变量 TEST_BACKEND 选择后端:
 *   TEST_BACKEND=java   (默认)  → http://127.0.0.1:8989
 *   TEST_BACKEND=python         → http://127.0.0.1:8991
 *   TEST_BACKEND=go             → http://127.0.0.1:8990
 *
 * 三个后端业务逻辑相同、API 路径统一为 /api/v1/...，
 * 验证码在三端均存于 Redis db0，key 前缀统一为 captcha_code:。
 */

export type BackendType = "java" | "python" | "go";

export interface BackendProfile {
  type: BackendType;
  baseURL: string;
}

export const REDIS_CONFIG = {
  host: "127.0.0.1",
  port: 6379,
  password: "12345678",
  db: 0,
};

export const CAPTCHA_KEY_PREFIX = "captcha_code:";

const PROFILES: Record<BackendType, BackendProfile> = {
  java: { type: "java", baseURL: "http://127.0.0.1:8989" },
  python: { type: "python", baseURL: "http://127.0.0.1:8991" },
  go: { type: "go", baseURL: "http://127.0.0.1:8990" },
};

const raw = (process.env.TEST_BACKEND || "java").toLowerCase().trim();
if (!(raw in PROFILES)) {
  throw new Error(`无效的 TEST_BACKEND 值: "${raw}"，应为 java | python | go`);
}
export const backendProfile: BackendProfile = PROFILES[raw as BackendType];
