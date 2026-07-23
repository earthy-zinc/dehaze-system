/**
 * 后端配置 - 支持在 Java / Python / Go 三个后端之间切换测试
 *
 * 通过环境变量 TEST_BACKEND 选择后端:
 *   TEST_BACKEND=java   (默认)  → http://127.0.0.1:8989
 *   TEST_BACKEND=python         → http://127.0.0.1:8991
 *   TEST_BACKEND=go             → http://127.0.0.1:8990
 *
 * 三个后端业务逻辑相同、API 路径统一为 /api/v1/...，
 * 验证码在三端均存于 Redis db0、纯文本形式，仅 key 前缀有差异。
 */

export type BackendType = "java" | "python" | "go";

export interface BackendProfile {
  /** 后端类型 */
  type: BackendType;
  /** 基础地址（不含路径） */
  baseURL: string;
  /** Redis 容器名 */
  redisContainer: string;
  /** Redis 密码 */
  redisPassword: string;
  /** 验证码所在的 Redis DB */
  captchaRedisDB: string;
  /** 验证码 key 前缀 */
  captchaKeyPrefix: string;
  /** 后端中文名 */
  name: string;
}

const REDIS_CONTAINER = "redis";
const REDIS_PASSWORD = "12345678";

const PROFILES: Record<BackendType, BackendProfile> = {
  java: {
    type: "java",
    baseURL: "http://127.0.0.1:8989",
    redisContainer: REDIS_CONTAINER,
    redisPassword: REDIS_PASSWORD,
    captchaRedisDB: "0",
    captchaKeyPrefix: "captcha_code:",
    name: "Java",
  },
  python: {
    type: "python",
    baseURL: "http://127.0.0.1:8991",
    redisContainer: REDIS_CONTAINER,
    redisPassword: REDIS_PASSWORD,
    captchaRedisDB: "0",
    captchaKeyPrefix: "captcha:",
    name: "Python",
  },
  go: {
    type: "go",
    baseURL: "http://127.0.0.1:8990",
    redisContainer: REDIS_CONTAINER,
    redisPassword: REDIS_PASSWORD,
    captchaRedisDB: "0",
    captchaKeyPrefix: "captcha_code:",
    name: "Go",
  },
};

function resolveBackend(): BackendType {
  const raw = (process.env.TEST_BACKEND || "java").toLowerCase().trim();
  if (raw === "java" || raw === "python" || raw === "go") {
    return raw;
  }
  throw new Error(`无效的 TEST_BACKEND 值: "${raw}"，应为 java | python | go`);
}

const backendType = resolveBackend();
export const backendProfile: BackendProfile = PROFILES[backendType];

export const isJava = backendType === "java";
export const isPython = backendType === "python";
export const isGo = backendType === "go";
