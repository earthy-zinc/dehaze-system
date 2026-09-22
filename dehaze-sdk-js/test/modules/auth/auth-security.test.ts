/**
 * 认证安全链路测试套件（Session 伪造/过期、并发安全、API Key 越权、脏语料、性能烟测）
 *
 * 设计依据：dehaze-doc/docs/03-模块设计/基础模块/认证管理/测试用例.md
 * 对应用例：T-AM-015/066/082/083（会话失效与伪造）、T-AM-100/101（并发）、
 *          T-AM-064/067（API Key 过期与越权）、T-AM-004/005（脱敏）、§6.1 性能指标
 *
 * 注意：
 * - 负向登录用例会递增 login:fail 计数，文件级 afterAll 统一清理
 * - 并发用例直接用 AuthAPI 原生调用，不走 utils/auth 的会话缓存
 */
import { describe, test, expect, beforeAll, beforeEach, afterAll } from "vitest";
import axios from "axios";
import { AuthAPI, UserAPI, ApiKeyAPI, service } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { clearLoginFailCounters, forceLogin, login, logout } from "#/utils/auth";
import { getRedis, disconnectRedis } from "#/utils/redis";
import { disconnectMysql } from "#/utils/mysql";
import { uniqueName, uniqueUsername } from "#/factories/common";
import { USERS, ROLES } from "#/factories/constants";
import { SEED_PASSWORD } from "#/config/constant";

async function captchaPair(): Promise<{ key: string; code: string }> {
  const captcha = await AuthAPI.getCaptcha();
  const redis = getRedis();
  const code = await redis.get(`captcha_code:${captcha.captchaKey}`);
  if (!code) throw new Error(`验证码已过期或不存在: ${captcha.captchaKey}`);
  return { key: captcha.captchaKey, code };
}

/**
 * 生成合规用户名：RegisterForm 校验 ^[a-zA-Z0-9_]{3,32}$，
 * uniqueName 的 nanoid 片段可能含连字符，注册场景不可用
 */
let usernameCounter = 1;
function compliantUsername(prefix: string): string {
  return `${prefix}${Date.now().toString(36)}${usernameCounter++}`.slice(0, 32);
}

beforeAll(async () => {
  await login(USERS.ADMIN.username);
});

describe("Session 安全（T-AM-066/082/083）", () => {
  test("边界：伪造 Session ID 应返回 401", async () => {
    const forged = "00000000-0000-4000-8000-000000000000";
    await expect(
      service.get("/api/v1/auth/me", { headers: { "X-Session-Id": forged } })
    ).rejects.toSatisfy((error: any) => error.response?.status === 401);
  });

  test("边界：格式畸形的 Session ID 应返回 401 而非 500", async () => {
    for (const malformed of ["' OR '1'='1", "%00%00", "a".repeat(512)]) {
      await expect(
        service.get("/api/v1/auth/me", { headers: { "X-Session-Id": malformed } })
      ).rejects.toSatisfy((error: any) => error.response?.status === 401);
    }
  });

  test("边界：注销后 Session 立即失效，重复注销返回 401", async () => {
    await login(USERS.ADMIN.username);
    const sessionId = globalThis.localStorage.getItem("sessionId") || "";
    expect(sessionId).toBeTruthy();

    await AuthAPI.logout();

    // 注销后原 Session 立即失效（T-AM-083）
    await expect(
      service.get("/api/v1/auth/me", { headers: { "X-Session-Id": sessionId } })
    ).rejects.toSatisfy((error: any) => error.response?.status === 401);

    // 重复注销同一 Session 返回 401（T-AM-015）
    await expect(
      service.post("/api/v1/auth/logout", undefined, { headers: { "X-Session-Id": sessionId } })
    ).rejects.toSatisfy((error: any) => error.response?.status === 401);

    await login(USERS.ADMIN.username);
  });
});

describe("登录错误信息脱敏（T-AM-004/005）", () => {
  // 失败登录会计入 login:fail:* 计数，先清零避免触发 30 分钟锁定连锁
  beforeEach(async () => {
    await clearLoginFailCounters();
  });

  test("验证：用户名不存在与密码错误返回同一业务码 A0210", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: "ghost_user_not_exist",
        password: SEED_PASSWORD,
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0210"]
    );

    const pair2 = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: "Definitely-Wrong-Pass",
        captchaKey: pair2.key,
        captchaCode: pair2.code,
      }),
      ["A0210"]
    );
  });
});

describe("对抗性脏语料（登录/验证码字段）", () => {
  // 每个脏语料用例都是一次失败登录，先清零失败计数避免触发 IP/账号锁定连锁
  beforeEach(async () => {
    await clearLoginFailCounters();
  });

  // 脏语料矩阵：CRLF / 零宽字符 / emoji / 全角 / BOM / 超长
  const dirtyValues = [
    "abcd\r\nEVIL",
    "ab\u200bcd",
    "验证码😀",
    "ａｂｃｄ",
    "﻿abcd",
    "a".repeat(2048),
  ];

  test.each(dirtyValues)("边界：captchaCode 脏语料 %j 应返回验证码错误而非 500", async (dirty) => {
    const captcha = await AuthAPI.getCaptcha();
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: SEED_PASSWORD,
        captchaKey: captcha.captchaKey,
        captchaCode: dirty,
      }),
      ["A0214", "A0213"]
    );
  });

  test("边界：captchaKey 超长/畸形应返回验证码过期而非 500", async () => {
    for (const dirtyKey of ["a".repeat(2048), "../../etc/passwd", "\u0000\u0000"]) {
      await expectBizError(
        AuthAPI.login({
          username: USERS.ADMIN.username,
          password: SEED_PASSWORD,
          captchaKey: dirtyKey,
          captchaCode: "0000",
        }),
        ["A0213"]
      );
    }
  });
});

describe("并发安全（T-AM-100/101）", () => {
  test("并发：同一 captchaKey 并发提交登录仅一个成功", async () => {
    const { key, code } = await captchaPair();
    const payload = {
      username: USERS.ADMIN.username,
      password: SEED_PASSWORD,
      captchaKey: key,
      captchaCode: code,
    };
    const settled = await Promise.allSettled([
      AuthAPI.login({ ...payload }),
      AuthAPI.login({ ...payload }),
    ]);
    const succeeded = settled.filter((r) => r.status === "fulfilled");
    const failed = settled.filter((r) => r.status === "rejected") as PromiseRejectedResult[];
    expect(succeeded.length).toBe(1);
    // 失败方必须是验证码已被消费（A0213），而非其他未定义错误
    for (const f of failed) {
      const bizCode = (f.reason as any)?.response?.data?.code;
      expect(bizCode).toBe("A0213");
    }

    // 单点模式下并发登录的胜出会话会踢掉拦截器缓存的旧会话，强制刷新登录态
    await forceLogin(USERS.ADMIN.username);
  });

  test("并发：同一用户名并发注册仅一个成功", async () => {
    const createdUserIds: number[] = [];
    const username = uniqueUsername("concreg");
    const [{ key: key1, code: code1 }, { key: key2, code: code2 }] = await Promise.all([
      captchaPair(),
      captchaPair(),
    ]);
    const settled = await Promise.allSettled([
      AuthAPI.register({
        username,
        password: SEED_PASSWORD,
        nickname: "并发A",
        captchaKey: key1,
        captchaCode: code1,
      }),
      AuthAPI.register({
        username,
        password: SEED_PASSWORD,
        nickname: "并发B",
        captchaKey: key2,
        captchaCode: code2,
      }),
    ]);
    const succeeded = settled.filter((r) => r.status === "fulfilled");
    const failed = settled.filter((r) => r.status === "rejected") as PromiseRejectedResult[];
    expect(succeeded.length).toBe(1);
    // 败方错误码按 python 基准实际行为（A0501）：并发竞态穿透 countByUsername 预检后，
    // 唯一键冲突被捕获并映射为 A0501"用户名已被注册"，三端（python/go/java）口径一致。
    for (const f of failed) {
      const bizCode = (f.reason as any)?.response?.data?.code;
      expect(bizCode).toBe("A0501");
    }

    // 清理注册产生的用户（并发缺陷可能产生 0/2 个额外账号，一并按名单清理）
    await login(USERS.ADMIN.username);
    const pageResult = await UserAPI.getPage({ pageNum: 1, pageSize: 100, keywords: username });
    for (const u of pageResult.list) {
      if (u.id) createdUserIds.push(u.id);
    }
    for (const userId of createdUserIds) {
      try {
        await UserAPI.deleteByIds(userId.toString());
      } catch {
        /* 忽略 */
      }
    }
  });
});

describe("API Key 安全（T-AM-064/067）", () => {
  test("边界：已过期 API Key 认证应被拒绝", async () => {
    await login(USERS.ADMIN.username);
    const pastDate = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
    const expiredKey = await ApiKeyAPI.create({
      name: uniqueName("expired"),
      expiresAt: pastDate,
    });
    try {
      await expect(
        service.get("/api/v1/auth/me", {
          headers: { Authorization: `Bearer ${expiredKey.apiKey}` },
        })
      ).rejects.toSatisfy((error: any) => error.response?.status === 401);
    } finally {
      await ApiKeyAPI.delete(expiredKey.id);
    }
  });

  test("边界：跨用户越权删除 API Key 应返回 A0401 且原 Key 不受影响", async () => {
    // admin 创建 Key
    await login(USERS.ADMIN.username);
    const adminKey = await ApiKeyAPI.create({ name: uniqueName("own") });

    try {
      // 切换到普通用户尝试越权删除
      await login(USERS.USER.username);
      await expectBizError(ApiKeyAPI.delete(adminKey.id), ["A0401"]);

      // 原 Key 仍可正常使用
      await login(USERS.ADMIN.username);
      const userInfo = (await service.get("/api/v1/auth/me", {
        headers: { Authorization: `Bearer ${adminKey.apiKey}` },
      })) as any;
      expect(userInfo.userId).toBe(USERS.ADMIN.id);
    } finally {
      await ApiKeyAPI.delete(adminKey.id);
    }
  });
});

describe("性能烟测（测试用例.md §6.1）", () => {
  test("性能：验证码生成响应时间 < 200ms", async () => {
    // 取 3 次最短耗时：单次采样会被连接建立与并行用例抢占 CPU 干扰，阈值仍按 §6.1 的
    // 200ms 判定，真实劣化会在每次采样中一致体现
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < 3; i += 1) {
      const start = Date.now();
      await AuthAPI.getCaptcha();
      best = Math.min(best, Date.now() - start);
    }
    expect(best).toBeLessThan(200);
  });

  test("性能：完整登录流程（含验证码+Session 创建）< 1s", async () => {
    const start = Date.now();
    const { key, code } = await captchaPair();
    const result = await AuthAPI.login({
      username: USERS.ADMIN.username,
      password: SEED_PASSWORD,
      captchaKey: key,
      captchaCode: code,
    });
    const elapsed = Date.now() - start;
    expect(result.sessionId).toBeTruthy();
    expect(elapsed).toBeLessThan(1000);
  });
});

describe("rememberMe Cookie 行为（T-AM-012a/012b）", () => {
  // SDK service 的响应拦截器会解包业务信封，拿不到原始响应头，
  // 这里用独立的 axios 实例直连后端读取 Set-Cookie
  const rawHttp = axios.create({ baseURL: process.env.BACKEND_URL ?? "http://127.0.0.1:8991" });

  test("验证：rememberMe=true 设置持久化 Cookie（Max-Age=604800）", async () => {
    const { key, code } = await captchaPair();
    const response = await rawHttp.post("/api/v1/auth/login", {
      username: USERS.ADMIN.username,
      password: SEED_PASSWORD,
      captchaKey: key,
      captchaCode: code,
      rememberMe: true,
    });
    const setCookies: string[] = response.headers["set-cookie"] ?? [];
    const sessionCookie = setCookies.find((c) => c.startsWith("X-Session-Id="));
    expect(sessionCookie).toBeDefined();
    expect(sessionCookie).toContain("Max-Age=604800");
  });

  test("验证：rememberMe=false 为会话级 Cookie（无 Max-Age）", async () => {
    const { key, code } = await captchaPair();
    const response = await rawHttp.post("/api/v1/auth/login", {
      username: USERS.ADMIN.username,
      password: SEED_PASSWORD,
      captchaKey: key,
      captchaCode: code,
      rememberMe: false,
    });
    const setCookies: string[] = response.headers["set-cookie"] ?? [];
    const sessionCookie = setCookies.find((c) => c.startsWith("X-Session-Id="));
    expect(sessionCookie).toBeDefined();
    expect(sessionCookie?.toLowerCase()).not.toContain("max-age");
  });
});

// 文件级 afterAll：先清登录失败计数（IP 锁定期内 login 会被拒）再恢复登录态
afterAll(async () => {
  await clearLoginFailCounters();
  await login(USERS.ADMIN.username);
  await disconnectRedis();
  await disconnectMysql();
});
