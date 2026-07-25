/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 支持多后端切换（Java/Python/Go），各后端验证码在 Redis 中的存储方式有差异
 */
import fs from "fs";
import path from "path";
import { SESSION_KEY } from "@/enums";
import { AuthAPI, configAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";
import { CAPTCHA_KEY_PREFIX } from "../config/backend";
import { getRedis } from "./redis";

const SESSION_CACHE_FILE = path.resolve(__dirname, "..", "..", ".session-cache.json");
const CACHE_MAX_AGE_MS = 30 * 60 * 1000;

let currentSessionId: string = "";

const TEST_CREDENTIALS = {
  username: process.env.TEST_USERNAME || "admin",
  password: process.env.TEST_PASSWORD || "123456",
};

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

export async function login(): Promise<string> {
  if (currentSessionId) {
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    return currentSessionId;
  }

  try {
    if (fs.existsSync(SESSION_CACHE_FILE)) {
      const cached = JSON.parse(fs.readFileSync(SESSION_CACHE_FILE, "utf-8"));
      if (
        cached.sessionId &&
        cached.createdAt &&
        Date.now() - cached.createdAt < CACHE_MAX_AGE_MS
      ) {
        currentSessionId = cached.sessionId;
        globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
        setupAxiosInterceptor();
        return currentSessionId;
      }
    }
  } catch {}

  try {
    const captcha = await AuthAPI.getCaptcha();

    const captchaCode = await getCaptchaCode(captcha.captchaKey);

    const result = await AuthAPI.login({
      ...TEST_CREDENTIALS,
      captchaKey: captcha.captchaKey,
      captchaCode,
    });
    currentSessionId = result.sessionId;

    if (!currentSessionId) {
      throw new Error("登录成功但 sessionId 为空");
    }

    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    setupAxiosInterceptor();

    try {
      fs.writeFileSync(
        SESSION_CACHE_FILE,
        JSON.stringify({ sessionId: currentSessionId, createdAt: Date.now() }),
        "utf-8"
      );
    } catch {}

    return currentSessionId;
  } catch (error) {
    console.error("登录失败:", error);
    throw error;
  }
}

export async function logout(): Promise<void> {
  await AuthAPI.logout();
  currentSessionId = "";
  globalThis.localStorage.removeItem(SESSION_KEY);
  try {
    fs.unlinkSync(SESSION_CACHE_FILE);
  } catch {}
}
