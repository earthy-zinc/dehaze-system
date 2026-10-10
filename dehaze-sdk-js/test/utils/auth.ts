/**
 * 认证相关的测试辅助工具
 * 提供登录、登出等功能
 *
 * 登录流程：获取验证码 → 从 Redis 读取验证码值 → 携带验证码登录
 * 三端验证码统一存于 REDIS_DATABASE 指定的库（dev db1 / pre db2，取值来自根 .env），key 前缀为 captcha_code:
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
import { SEED_PASSWORD } from "#/config/constant";

let currentSessionId: string = "";
let activeUser: string = "";

// 按用户名缓存 sessionId，切换用户时直接复用，避免触发登录限流
const sessionStore = new Map<string, string>();

/**
 * 默认登录账号：多个并行套件可用 SDK_TEST_USER 各登一个种子账号（admin/root/test 等，同口令），
 * 否则任一以 admin 登录的套件都会按设备数上限踢掉同账号的其他持有者会话（表现为成片 A0230）。
 */
export const DEFAULT_TEST_USER = process.env.SDK_TEST_USER || "admin";

/**
 * 验证码明文 → 实际要提交的验证码。
 *
 * Java 端生产 profile 的验证码字符是算式（captcha.code.type=math，Hutool MathGenerator）：
 * Redis 存的是算式本身（如 "2*4="，运算位数 >1 时数字后会补空格），校验方按算式求值再与
 * 提交值比较 ⇒ 必须提交计算结果，不能直接回填 Redis 原文。dev 的 random 与 Go/Python 端
 * 存的就是最终码，原样提交即可。
 */
function solveCaptcha(raw: string): string {
  const compact = raw.replace(/\s/g, "");
  const matched = /^(\d+)([+\-*])(\d+)=?$/.exec(compact);
  if (!matched) {
    return compact;
  }
  const left = Number(matched[1]);
  const right = Number(matched[3]);
  switch (matched[2]) {
    case "+":
      return String(left + right);
    case "-":
      return String(left - right);
    default:
      return String(left * right);
  }
}

/** 从 Redis 读取验证码并换算为可提交的值（登录/注册共用，勿在用例内重复实现） */
export async function getCaptchaCode(captchaKey: string): Promise<string> {
  const redisKey = `captcha_code:${captchaKey}`;
  const redis = getRedis();
  const code = await redis.get(redisKey);
  if (!code) {
    throw new Error(`验证码已过期或不存在: ${captchaKey}`);
  }
  return solveCaptcha(code);
}

/**
 * 构造一个保证校验不通过的错误验证码，供"验证码错误（A0214）"负向用例使用。
 *
 * 不能硬编码固定值：Java 生产 profile 的验证码是算式（captcha.code.type=math、code.length=1），
 * 正确答案有约 1/10 概率为 0（如 "0*3="、"5-5="），而 "0000" 经校验方 Integer.parseInt 同样是 0，
 * 会被判为正确、登录直接成功（用例 10% 概率假失败）。故先取真实答案，再避开它：
 * 答案为 0 时提交 "0001"，否则提交 "0000"——两种情况下都不会与正确答案相等。
 */
export async function getWrongCaptchaCode(captchaKey: string): Promise<string> {
  const correct = await getCaptchaCode(captchaKey);
  return Number(correct) === 0 ? "0001" : "0000";
}

function setupAxiosInterceptor() {
  configAxios({
    onRequest: (config: InternalAxiosRequestConfig) => {
      // 仅在调用方未显式指定时注入会话头，伪造/过期会话等负向用例需覆盖自定义值
      if (!config.headers["X-Session-Id"]) {
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
}

function applySession(sessionId: string, username: string) {
  currentSessionId = sessionId;
  activeUser = username;
  globalThis.localStorage.setItem(SESSION_KEY, sessionId);
  setupAxiosInterceptor();
}

/** 清理登录/注册限流计数：三端键名与结构不同，需逐族清理，否则重复运行会命中 10 次/分钟阈值 */
export async function clearAuthRateLimit(): Promise<void> {
  const redis = getRedis();
  // 只删计数子 key，不删 Redisson 配置 key（rate:limit:{scope}:{ip} 本身）
  // Redisson 计数: {rate:limit:login:{ip}}:value / :permits
  // Python 计数: rate:limit:/api/v1/auth/login:{ip}、rate:limit:auth:register:{ip}
  // Go 计数: rate:limit:ip:register:{ip}
  const patterns = [
    "{rate:limit:login:*",
    "rate:limit:/api/v1/auth/login:*",
    "{rate:limit:register:*",
    "rate:limit:/api/v1/auth/register:*",
    "rate:limit:auth:register:*",
    "rate:limit:ip:register:*",
  ];
  for (const pattern of patterns) {
    const keys = await redis.keys(pattern);
    if (keys.length > 0) {
      await redis.del(keys);
    }
  }
}

/**
 * 清理登录失败锁定计数（IP + 用户名双维度）。
 * 负向登录/脏语料用例会累积 login:fail:* 计数，达 5 次触发 30 分钟锁定
 * 连锁拒绝后续登录，因此失败类用例执行前需清零。
 */
export async function clearLoginFailCounters(): Promise<void> {
  const redis = getRedis();
  const keys = await redis.keys("login:fail:*");
  if (keys.length > 0) {
    await redis.del(keys);
  }
}

async function doLogin(username: string) {
  const captcha = await AuthAPI.getCaptcha();
  const captchaCode = await getCaptchaCode(captcha.captchaKey);
  return await AuthAPI.login({
    username,
    password: SEED_PASSWORD,
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
      await clearAuthRateLimit();
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

/**
 * 强制重新登录（忽略会话缓存）。
 * 并发登录/注销类用例可能在服务端踢掉缓存会话（单点互踢），
 * 后续用例复用失效会话会得到 401，需用此函数刷新缓存。
 */
export async function forceLogin(username: string = DEFAULT_TEST_USER): Promise<string> {
  sessionStore.delete(username);
  if (activeUser === username) {
    currentSessionId = "";
    activeUser = "";
  }
  return await login(username);
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
