/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 支持多后端切换（Java/Python/Go），各后端验证码在 Redis 中的存储方式有差异
 */
import { TOKEN_KEY } from "@/enums";
import { AuthAPI, configJavaAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";
import { execSync } from "child_process";
import { backendProfile } from "../config/backend";

let currentToken: string = "";

const TEST_CREDENTIALS = {
  username: process.env.TEST_USERNAME || "admin",
  password: process.env.TEST_PASSWORD || "123456",
};

/**
 * 从 Redis 读取验证码值
 * 三端统一: 存于 Redis db0、纯文本形式，仅 key 前缀有差异:
 *   - Java:   key=captcha_code:{captchaKey}
 *   - Python: key=captcha:{captchaKey}
 *   - Go:     key=captcha_code:{captchaKey}
 */
function getCaptchaCodeFromRedis(captchaKey: string): string {
  const redisKey = `${backendProfile.captchaKeyPrefix}${captchaKey}`;
  return execSync(
    `docker exec -i ${backendProfile.redisContainer} redis-cli -a ${backendProfile.redisPassword} -n ${backendProfile.captchaRedisDB} get ${redisKey}`,
    { encoding: "utf-8", stdio: ["pipe", "pipe", "ignore"] }
  ).trim();
}

/**
 * 执行登录操作，保存 token 并配置 axios 请求拦截器
 */
export async function login(): Promise<string> {
  try {
    // 1. 获取验证码
    const captcha = await AuthAPI.getCaptcha();

    // 2. 从 Redis 读取验证码值
    const captchaCode = getCaptchaCodeFromRedis(captcha.captchaKey);
    if (!captchaCode) {
      throw new Error(`验证码已过期或不存在: ${captcha.captchaKey}`);
    }

    // 3. 携带验证码登录
    const result = await AuthAPI.login({
      ...TEST_CREDENTIALS,
      captchaKey: captcha.captchaKey,
      captchaCode,
    });
    currentToken = result.accessToken!;

    const tokenValue = currentToken.startsWith("Bearer ") ? currentToken : `Bearer ${currentToken}`;
    globalThis.localStorage.setItem(TOKEN_KEY, tokenValue);

    configJavaAxios({
      onRequest: (config: InternalAxiosRequestConfig) => {
        if (currentToken) {
          config.headers.Authorization = currentToken.startsWith("Bearer ")
            ? currentToken
            : `Bearer ${currentToken}`;
        }

        if (config.data instanceof FormData) {
          delete config.headers["Content-Type"];
          const formHeaders = config.data.getHeaders();
          Object.assign(config.headers, formHeaders);
        }

        return config;
      },
    });

    return currentToken;
  } catch (error) {
    console.error("登录失败:", error);
    throw error;
  }
}

/**
 * 执行登出操作，清理 token 和 axios 配置
 */
export async function logout(): Promise<void> {
  try {
    await AuthAPI.logout();
  } catch (error) {
    console.error("登出失败:", error);
  } finally {
    currentToken = "";
    globalThis.localStorage.removeItem(TOKEN_KEY);
    configJavaAxios({});
  }
}
