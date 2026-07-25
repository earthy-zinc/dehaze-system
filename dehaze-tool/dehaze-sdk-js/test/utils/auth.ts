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
import { execSync } from "child_process";
import { backendProfile } from "../config/backend";

let currentSessionId: string = "";

const TEST_CREDENTIALS = {
  username: process.env.TEST_USERNAME || "admin",
  password: process.env.TEST_PASSWORD || "123456",
};

function getCaptchaCodeFromRedis(captchaKey: string): string {
  const redisKey = `${backendProfile.captchaKeyPrefix}${captchaKey}`;
  return execSync(
    `docker exec -i ${backendProfile.redisContainer} redis-cli -a ${backendProfile.redisPassword} -n ${backendProfile.captchaRedisDB} get ${redisKey}`,
    { encoding: "utf-8", stdio: ["pipe", "pipe", "ignore"] }
  ).trim();
}

export async function login(): Promise<string> {
  try {
    const captcha = await AuthAPI.getCaptcha();

    const captchaCode = getCaptchaCodeFromRedis(captcha.captchaKey);
    if (!captchaCode) {
      throw new Error(`验证码已过期或不存在: ${captcha.captchaKey}`);
    }

    const result = await AuthAPI.login({
      ...TEST_CREDENTIALS,
      captchaKey: captcha.captchaKey,
      captchaCode,
    });
    currentSessionId = result.sessionId;

    globalThis.localStorage.setItem(SESSION_KEY, currentSessionId);

    configJavaAxios({
      onRequest: (config: InternalAxiosRequestConfig) => {
        if (currentSessionId) {
          config.headers["X-Session-Id"] = currentSessionId;
        }

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

export async function logout(): Promise<void> {
  try {
    await AuthAPI.logout();
  } catch (error) {
    console.error("登出失败:", error);
  } finally {
    currentSessionId = "";
    globalThis.localStorage.removeItem(SESSION_KEY);
    configJavaAxios({});
  }
}
