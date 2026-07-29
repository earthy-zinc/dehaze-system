/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 三端验证码统一存于 Redis db0，key 前缀为 captcha_code:
 *
 * 设计说明：
 * - session 仅在内存中复用（同一测试运行内按用户名缓存，切换用户不触发重复登录）
 * - 不做本地文件缓存，避免与后端 Redis 状态不同步导致需要手动清理
 */
import { SESSION_KEY } from "@/enums";
import { AuthAPI, configAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";
import { getRedis } from "./redis";

let currentSessionId: string = "";
let activeUser: string = "";

// 按用户名缓存 sessionId，切换用户时直接复用，避免触发登录限流
const sessionStore = new Map<string, string>();

async function getCaptchaCode(captchaKey: string): Promise<string> {
  const redisKey = `${process.env.CAPTCHA_KEY_PREFIX || "captcha_code:"}${captchaKey}`;
  const redis = getRedis();
  const code = await redis.get(redisKey);
  if (!code) {
    throw new Error(`验证码已过期或不存在: ${captchaKey}`);
  }
  return code;
}

function setupAxiosInterceptor() {
  configAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      config.headers["X-Session-Id"] = currentSessionId;
      if (config.data instanceof FormData) {
        delete config.headers["Content-Type"];
        const formHeaders = config.data.getHeaders();
        Object.assign(config.headers, formHeaders);
      }
      return config;
    },
  });
}

function applySession(sessionId: string, username: string) {
  currentSessionId = sessionId;
  activeUser = username;
  globalThis.localStorage.setItem(SESSION_KEY, sessionId);
  setupAxiosInterceptor();
}

async function clearLoginRateLimit(): Promise<void> {
  const redis = getRedis();
  // Java Redisson: rate_limit:login:{ip}:{class}#{method} + {rate_limit:login:...}:value + {rate_limit:login:...}:permits
  // Python: rate:limit:/api/v1/auth/login:{ip}
  // Go: rate:limit:{prefix}:{ip}（未挂载到登录路由，预防性清理）
  for (const pattern of ["*rate_limit:login:*", "*rate:limit:*/api/v1/auth/login:*"]) {
    const keys = await redis.keys(pattern);
    if (keys.length > 0) {
      await redis.del(keys);
    }
  }
}

async function doLogin(username: string) {
  const captcha = await AuthAPI.getCaptcha();
  const captchaCode = await getCaptchaCode(captcha.captchaKey);
  return await AuthAPI.login({
    username,
    password: process.env.TEST_PASSWORD || "12345678",
    captchaKey: captcha.captchaKey,
    captchaCode,
  });
}

export async function login(username: string = "admin"): Promise<string> {
  if (activeUser === username && currentSessionId) {
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    return currentSessionId;
  }

  const cached = sessionStore.get(username);
  if (cached) {
    applySession(cached, username);
    return currentSessionId;
  }

  let result;
  try {
    result = await doLogin(username);
  } catch (err: any) {
    const status = err?.response?.status;
    const bizCode = err?.response?.data?.code;
    // HTTP 429 或业务码 B0211 都视为限流
    if (status === 429 || bizCode === "B0211") {
      // 测试中切换多用户累积触发后端登录限流（Java @RateLimit 10次/60秒）
      // 清理限流计数后重试一次
      await clearLoginRateLimit();
      result = await doLogin(username);
    } else {
      throw err;
    }
  }

  if (!result.sessionId) {
    throw new Error("登录成功但 sessionId 为空");
  }

  sessionStore.set(username, result.sessionId);
  applySession(result.sessionId, username);
  return currentSessionId;
}

export async function logout(username?: string): Promise<void> {
  if (username && username !== activeUser) {
    return;
  }
  try {
    await AuthAPI.logout();
  } finally {
    if (activeUser) {
      sessionStore.delete(activeUser);
    }
    currentSessionId = "";
    activeUser = "";
    globalThis.localStorage.removeItem(SESSION_KEY);
  }
}
