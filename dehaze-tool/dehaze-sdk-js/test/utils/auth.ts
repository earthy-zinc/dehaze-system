/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 */
import { TOKEN_KEY } from "@/enums";
import { AuthAPI, configJavaAxios } from "../../index";
import type { InternalAxiosRequestConfig } from "axios";
import FormData from "form-data";

let currentToken: string = "";

const TEST_CREDENTIALS = {
  username: process.env.TEST_USERNAME || "admin",
  password: process.env.TEST_PASSWORD || "123456",
  captchaKey: "",
  captchaCode: "",
};

/**
 * 执行登录操作，保存 token 并配置 axios 请求拦截器
 */
export async function login(): Promise<string> {
  try {
    const result = await AuthAPI.login(TEST_CREDENTIALS);
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
