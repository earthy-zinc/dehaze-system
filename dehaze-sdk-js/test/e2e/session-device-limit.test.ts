/**
 * 同时在线设备数上限端到端用例（F-AM-011）
 *
 * 行为受后端开关 `USE_MULTI_POINT` 控制，dev/测试实例默认 false（"多端共存"现状），
 * 故本文件放在 `test/e2e/`（默认套件只收 test/modules 目录下的用例，不会带上它），
 * 只在开启开关的窗口内显式执行：
 *
 *   # ① 开启窗口：环境变量优先于 .env（pydantic-settings），无需改共享 .env 文件
 *   USE_MULTI_POINT=true python scripts/run.py restart python
 *   cd dehaze-sdk-js && SESSION_MULTI_POINT=true pnpm test:e2e
 *   # ② 关闭窗口并验证恢复
 *   python scripts/run.py restart python
 *   cd dehaze-sdk-js && pnpm test:e2e     # 未设 SESSION_MULTI_POINT 时按 false 断言现状
 *
 * `SESSION_MULTI_POINT` 必须与服务端开关一致：不一致时断言必然失败（不做静默跳过）。
 */
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { AuthAPI, service } from "../../index";
import { USERS } from "#/factories/constants";
import { SEED_PASSWORD } from "#/config/constant";
import { clearLoginFailCounters } from "#/utils/auth";
import { getRedis, disconnectRedis } from "#/utils/redis";

const SWITCH_ON = process.env.SESSION_MULTI_POINT === "true";

/** 直连登录（不走 utils/auth 的会话缓存，便于同时持有多个会话） */
async function rawLogin(username: string): Promise<string> {
  const captcha = await AuthAPI.getCaptcha();
  const redis = getRedis();
  const code = await redis.get(`captcha_code:${captcha.captchaKey}`);
  if (!code) throw new Error(`验证码已过期或不存在: ${captcha.captchaKey}`);
  const result = await AuthAPI.login({
    username,
    password: SEED_PASSWORD,
    captchaKey: captcha.captchaKey,
    captchaCode: code,
  });
  if (!result.sessionId) throw new Error("登录成功但 sessionId 为空");
  return result.sessionId;
}

/** 会话是否仍可鉴权：401 = 已被踢出；其他异常必须抛出（不得被当成"已踢出"） */
async function sessionAlive(sessionId: string): Promise<boolean> {
  try {
    await service.get("/api/v1/auth/me", { headers: { "X-Session-Id": sessionId } });
    return true;
  } catch (error: any) {
    if (error?.response?.status === 401) return false;
    throw error;
  }
}

const issuedSessions: string[] = [];

beforeAll(async () => {
  // 失败登录会累积 login:fail:* 计数并触发 30 分钟锁定，跑前清零（本文件登录均为成功，仅防御）
  await clearLoginFailCounters();
});

afterAll(async () => {
  // 清理本文件签发的会话，避免影响其他套件（逐条幂等，已踢出的会话注销返回 401 属正常）
  for (const sessionId of issuedSessions) {
    try {
      await service.post("/api/v1/auth/logout", undefined, {
        headers: { "X-Session-Id": sessionId },
      });
    } catch {
      /* 已被踢出 */
    }
  }
  await disconnectRedis();
});

describe("同时在线设备数上限（F-AM-011）", () => {
  test("普通用户（level_0，上限 1）：两次登录后在线会话数与开关口径一致", async () => {
    const first = await rawLogin(USERS.USER.username);
    const second = await rawLogin(USERS.USER.username);
    issuedSessions.push(first, second);

    const alive: string[] = [];
    if (await sessionAlive(first)) alive.push("first");
    if (await sessionAlive(second)) alive.push("second");

    // 开关开启：第 2 台登录踢掉最早的第 1 台（新会话保留）；关闭：两端共存
    expect(alive).toEqual(SWITCH_ON ? ["second"] : ["first", "second"]);

    if (SWITCH_ON) {
      // 被踢会话的错误口径：HTTP 401 + 业务码 A0230（token 无效/已过期）
      await expect(
        service.get("/api/v1/auth/me", { headers: { "X-Session-Id": first } })
      ).rejects.toSatisfy(
        (error: any) => error?.response?.status === 401 && error?.response?.data?.code === "A0230"
      );
    }

    const indexKey = `session:user:${USERS.USER.id}`;
    const redis = getRedis();
    if (SWITCH_ON) {
      // 索引 session:user:{userId} 为 ZSet，member=sessionId，基数=同时在线数
      expect(await redis.type(indexKey)).toBe("zset");
      expect(await redis.zcard(indexKey)).toBe(1);
      expect(await redis.zrange(indexKey, 0, -1)).toEqual([second]);
    } else {
      // 开关关闭时不维护索引（多点登录控制未启用）
      expect(await redis.exists(indexKey)).toBe(0);
    }
  });

  test("被踢会话的下一次请求返回 401，新会话正常", async () => {
    const first = await rawLogin(USERS.VIP1.username);
    const second = await rawLogin(USERS.VIP1.username);
    issuedSessions.push(first, second);

    // level_1 上限 3：仅两次登录，两台都在限内 → 两端均可用（开关开关两态一致）
    expect(await sessionAlive(first)).toBe(true);
    expect(await sessionAlive(second)).toBe(true);
    const redis = getRedis();
    expect(await redis.zcard(`session:user:${USERS.VIP1.id}`)).toBe(SWITCH_ON ? 2 : 0);
  });

  test("管理员不受 level_0 的 1 台限制（固定 10 台）", async () => {
    const first = await rawLogin(USERS.ADMIN.username);
    const second = await rawLogin(USERS.ADMIN.username);
    issuedSessions.push(first, second);

    // admin 账号等级为 level_0（上限 1），但角色为 ADMIN → 固定 10 台，两台均在线
    // （若错按等级权益取 1 台，first 必被踢出 → 本条失败）
    expect(await sessionAlive(first)).toBe(true);
    expect(await sessionAlive(second)).toBe(true);

    const redis = getRedis();
    const indexKey = `session:user:${USERS.ADMIN.id}`;
    if (SWITCH_ON) {
      // 索引中除本用例两台外还有 vitest.setup 的 admin 会话，故只断言包含关系
      expect(await redis.zrange(indexKey, 0, -1)).toEqual(expect.arrayContaining([first, second]));
    } else {
      expect(await redis.exists(indexKey)).toBe(0);
    }
  });
});
