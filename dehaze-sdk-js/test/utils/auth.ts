/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 三端验证码统一存于 Redis db0，key 前缀为 captcha_code:
 */
import fs from "fs";
import path from "path";
import { SESSION_KEY } from "@/enums";
import { AuthAPI, configAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";
import { getRedis } from "./redis";

const SESSION_CACHE_FILE = path.resolve(__dirname, "..", "..", ".session-cache.json");
const CACHE_MAX_AGE_MS = 30 * 60 * 1000;
const CAPTCHA_KEY_PREFIX = process.env.CAPTCHA_KEY_PREFIX || "captcha_code:";

let currentSessionId: string = "";
let activeUser: string = "";

const DEFAULT_PASSWORD = process.env.TEST_PASSWORD || "12345678";

function getCacheKey(username: string): string {
  return `${process.env.TEST_BACKEND || "java"}:${username}`;
}

async function getCaptchaCode(captchaKey: string): Promise<string> {
  const redisKey = `${CAPTCHA_KEY_PREFIX}${captchaKey}`;
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

function readSessionCache(): Record<string, { sessionId: string; createdAt: number }> {
  try {
    if (fs.existsSync(SESSION_CACHE_FILE)) {
      return JSON.parse(fs.readFileSync(SESSION_CACHE_FILE, "utf-8"));
    }
  } catch {}
  return {};
}

function writeSessionCache(cache: Record<string, { sessionId: string; createdAt: number }>): void {
  try {
    fs.writeFileSync(SESSION_CACHE_FILE, JSON.stringify(cache), "utf-8");
  } catch {}
}

export async function login(username: string = "admin"): Promise<string> {
  const cacheKey = getCacheKey(username);
  if (activeUser === cacheKey && currentSessionId) {
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    return currentSessionId;
  }

  const cache = readSessionCache();
  const cached = cache[cacheKey];
  if (cached?.sessionId && Date.now() - cached.createdAt < CACHE_MAX_AGE_MS) {
    currentSessionId = cached.sessionId;
    activeUser = cacheKey;
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    setupAxiosInterceptor();
    return currentSessionId;
  }

  try {
    const captcha = await AuthAPI.getCaptcha();
    const captchaCode = await getCaptchaCode(captcha.captchaKey);

    const result = await AuthAPI.login({
      username,
      password: DEFAULT_PASSWORD,
      captchaKey: captcha.captchaKey,
      captchaCode,
    });
    currentSessionId = result.sessionId;

    if (!currentSessionId) {
      throw new Error("登录成功但 sessionId 为空");
    }

    activeUser = cacheKey;
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    setupAxiosInterceptor();

    cache[cacheKey] = { sessionId: currentSessionId, createdAt: Date.now() };
    writeSessionCache(cache);

    return currentSessionId;
  } catch (error) {
    console.error("登录失败:", error);
    throw error;
  }
}

export async function logout(username?: string): Promise<void> {
  const cacheKey = username ? getCacheKey(username) : activeUser;
  await AuthAPI.logout();
  currentSessionId = "";
  if (cacheKey === activeUser) {
    activeUser = "";
  }
  globalThis.localStorage.removeItem(SESSION_KEY);

  const cache = readSessionCache();
  delete cache[cacheKey];
  if (Object.keys(cache).length === 0) {
    try {
      fs.unlinkSync(SESSION_CACHE_FILE);
    } catch {}
  } else {
    writeSessionCache(cache);
  }
}
