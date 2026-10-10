/**
 * 页面测试的登录辅助。
 *
 * 与 SDK 契约测试（dehaze-sdk-js/test/utils/auth.ts）同口径：验证码不支持绕过，走
 * 「取验证码 → 从 Redis 读明文 → 换算为可提交值」，Redis 连接参数取自仓库根 .env。
 *
 * 会话凭证是 HttpOnly cookie，页面 JS 不可读写，因此这里用 BrowserContext 的 request
 * 走接口登录——它与页面共享 cookie jar，登录后 page 直接处于已登录态，无需再走登录 UI。
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect } from "@playwright/test";
import type { BrowserContext, Page, Route } from "@playwright/test";
import dotenv from "dotenv";
import Redis from "ioredis";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../.."
);
dotenv.config({ path: path.join(repoRoot, ".env"), quiet: true });

/** 种子账号口令明文，与 config/sql/data/sys_user.sql 的 bcrypt 对应 */
export const SEED_PASSWORD = "Dehaze2026";

const SUCCESS_CODE = "00000";

let redis: Redis | null = null;

function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`缺少环境变量 ${name}：请确认仓库根 .env 已按环境模板落地`);
  }
  return value;
}

function getRedis(): Redis {
  if (!redis) {
    redis = new Redis({
      host: required("REDIS_HOST"),
      port: Number(required("REDIS_PORT")),
      password: required("REDIS_PASSWORD"),
      // 验证码按环境分库（dev db1 / pre db2）：对已部署环境跑用例时用 E2E_REDIS_DATABASE 覆盖
      db: Number(process.env.E2E_REDIS_DATABASE ?? required("REDIS_DATABASE")),
      maxRetriesPerRequest: 3,
    });
    redis.on("error", () => {});
  }
  return redis;
}

/**
 * 验证码明文 → 实际要提交的值。
 *
 * math 型（Java 生产 profile）Redis 存的是算式，校验方按算式求值后比对，必须提交计算结果；
 * random 型（dev 与 Go/Python 端）存的就是最终码，原样提交。
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

/** 取验证码并换算为可提交值 */
export async function fetchCaptchaCode(
  context: BrowserContext
): Promise<{ captchaKey: string; captchaCode: string }> {
  const response = await context.request.get("/api/v1/auth/captcha");
  const body = await response.json();
  const captchaKey: string | undefined = body?.data?.captchaKey;
  if (!captchaKey) {
    throw new Error(
      `获取验证码失败：HTTP ${response.status()} ${JSON.stringify(body)}`
    );
  }

  const raw = await getRedis().get(`captcha_code:${captchaKey}`);
  if (!raw) {
    throw new Error(`验证码已过期或不存在: ${captchaKey}`);
  }
  return { captchaKey, captchaCode: solveCaptcha(raw) };
}

/**
 * 供「走登录 UI 表单」的用例获取验证码：表单提交的 captchaKey 是页面自己请求的那次，
 * 外部另取一份验证码无法与之配对，因此拦截页面自身的验证码请求，读出与其配对的验证码值。
 */
export async function capturePageCaptcha(page: Page): Promise<string> {
  let captchaKey = "";
  const captureCaptchaKey = async (route: Route) => {
    const response = await route.fetch();
    const body = await response.json();
    captchaKey = body?.data?.captchaKey ?? "";
    await route.fulfill({ response });
  };
  await page.route("**/api/v1/auth/captcha", captureCaptchaKey);
  await page.reload();
  await expect
    .poll(() => captchaKey, { message: "页面未发起验证码请求" })
    .toBeTruthy();
  // 拦截只服务于本函数取 key：留着会继续拦截页面后续的验证码刷新，并在用例收尾时对
  // 已关闭的页面发起请求，产生"不属于任何用例"的报告噪音
  await page.unroute("**/api/v1/auth/captcha", captureCaptchaKey);

  const raw = await getRedis().get(`captcha_code:${captchaKey}`);
  if (!raw) {
    throw new Error(`验证码已过期或不存在: ${captchaKey}`);
  }
  return solveCaptcha(raw);
}

/** 通过接口登录，登录态落进 context 的 cookie jar */
export async function loginByApi(
  context: BrowserContext,
  username = "admin"
): Promise<void> {
  const { captchaKey, captchaCode } = await fetchCaptchaCode(context);
  const response = await context.request.post("/api/v1/auth/login", {
    data: { username, password: SEED_PASSWORD, captchaKey, captchaCode },
  });
  const body = await response.json();
  if (body?.code !== SUCCESS_CODE) {
    throw new Error(`登录失败：${body?.code} ${body?.msg ?? ""}`);
  }
}
