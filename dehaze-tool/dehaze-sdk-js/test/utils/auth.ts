/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 支持多后端切换（Java/Python/Go），各后端验证码在 Redis 中的存储方式有差异
 */
import { SESSION_KEY } from "@/enums";
import { AuthAPI, configJavaAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";
import { CAPTCHA_KEY_PREFIX } from "../config/backend";
import { getRedis } from "./redis";

let currentSessionId: string = "";

const TEST_CREDENTIALS = {
  username: process.env.TEST_USERNAME || "admin",
  password: process.env.TEST_PASSWORD || "123456",
};

async function getCaptchaCodeFromRedis(captchaKey: string): Promise<string> {
  const redisKey = `${CAPTCHA_KEY_PREFIX}${captchaKey}`;
  const redis = getRedis();
  const code = await redis.get(redisKey);
  return code || "";
}

export async function login(): Promise<string> {
  if (currentSessionId) {
    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);
    return currentSessionId;
  }

  try {
    const captcha = await AuthAPI.getCaptcha();

    const captchaCode = await getCaptchaCodeFromRedis(captcha.captchaKey);
    if (!captchaCode) {
      throw new Error(`验证码已过期或不存在: ${captcha.captchaKey}`);
    }

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

    configJavaAxios({
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

    return currentSessionId;
  } catch (error) {
    console.error("登录失败:", error);
    throw error;
  }
}

export async function logout(): Promise<void> {}
