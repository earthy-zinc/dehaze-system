/**
 * 认证管理测试套件
 *
 * 覆盖认证模块核心 P0/P1 用例：验证码、登录、注册、注销、当前用户权限、登录日志、会话管理。
 * 设计依据：dehaze-doc/docs/03-模块设计/基础模块/认证管理/测试用例.md
 *
 * 注意：
 * - 不测试登录失败锁定（T-AM-050 系列）；负向登录用例会递增 login:fail 计数，文件级 afterAll 会清理避免触发 30 分钟锁定
 * - 未认证访问（401）已由 security.test.ts 覆盖，此处不重复
 * - vitest.setup.ts 在 beforeAll 已自动 login("admin")，涉及登录/注销切换的块在 afterAll 恢复 admin 登录态
 */
import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AuthAPI } from "../../../index";
import UserAPI from "@/api/user";
import { expectBizError } from "#/utils/assertion";
import { login, logout } from "#/utils/auth";
import { getRedis, disconnectRedis } from "#/utils/redis";
import { uniqueName } from "#/factories/common";
import { USERS, ROLES } from "#/factories/constants";
import { ADMIN_PASSWORD } from "#/config/constant";

/**
 * 从 Redis 读取验证码
 * 三端验证码统一存于 Redis db0，key 前缀为 captcha_code:
 */
async function getCaptchaCode(captchaKey: string): Promise<string> {
  const redis = getRedis();
  const code = await redis.get(`captcha_code:${captchaKey}`);
  if (!code) {
    throw new Error(`验证码已过期或不存在: ${captchaKey}`);
  }
  return code;
}

/** 获取验证码 Key 及对应 Redis 中的验证码值，用于携带正确验证码的登录/注册场景 */
async function captchaPair(): Promise<{ key: string; code: string }> {
  const captcha = await AuthAPI.getCaptcha();
  const code = await getCaptchaCode(captcha.captchaKey);
  return { key: captcha.captchaKey, code };
}

describe("GET /api/v1/auth/captcha - 获取验证码", () => {
  test("正向测试：获取验证码并验证返回结构", async () => {
    const result = await AuthAPI.getCaptcha();
    expect(typeof result.captchaKey).toBe("string");
    expect(typeof result.captchaBase64).toBe("string");
  });

  test("验证：验证码图片为 Base64 格式", async () => {
    const result = await AuthAPI.getCaptcha();
    expect(result.captchaBase64.length).toBeGreaterThan(100);
  });

  test("验证：每次获取的验证码 Key 不同", async () => {
    const result1 = await AuthAPI.getCaptcha();
    const result2 = await AuthAPI.getCaptcha();
    expect(result1.captchaKey).not.toBe(result2.captchaKey);
  });
});

describe("POST /api/v1/auth/login - 用户登录", () => {
  test("正向测试：用户名密码登录成功", async () => {
    const { key, code } = await captchaPair();
    const result = await AuthAPI.login({
      username: USERS.ADMIN.username,
      password: ADMIN_PASSWORD,
      captchaKey: key,
      captchaCode: code,
    });
    expect(result.sessionId).toBeTruthy();
    expect(result.user.username).toBe(USERS.ADMIN.username);
  });

  test("边界：用户名不存在应返回认证失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: "notexist_user_xyz",
        password: ADMIN_PASSWORD,
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0210", "A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });

  test("边界：密码错误应返回认证失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: "WrongPassword123",
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0210", "A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });

  // 后端已按文档区分验证码错误（A0214）与验证码过期/不存在（A0213），详见 verify_captcha_status。
  test("边界：验证码错误应返回验证码错误码 A0214", async () => {
    const captcha = await AuthAPI.getCaptcha();
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: ADMIN_PASSWORD,
        captchaKey: captcha.captchaKey,
        captchaCode: "0000",
      }),
      ["A0214"]
    );
  });

  test("边界：验证码不存在/过期应返回验证码过期码 A0213", async () => {
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: ADMIN_PASSWORD,
        captchaKey: "nonexistent_captcha_key",
        captchaCode: "1234",
      }),
      ["A0213"]
    );
  });

  test("异常：缺少用户名应返回参数校验失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: "",
        password: ADMIN_PASSWORD,
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });

  test("异常：缺少密码应返回参数校验失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.login({
        username: USERS.ADMIN.username,
        password: "",
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });
});

describe("POST /api/v1/auth/register - 用户注册", () => {
  const createdUserIds: number[] = [];

  afterAll(async () => {
    // 切回 admin 清理注册创建的用户
    await login(USERS.ADMIN.username);
    for (const userId of createdUserIds.reverse()) {
      try {
        await UserAPI.deleteByIds(userId.toString());
      } catch {
        /* 忽略 */
      }
    }
  });

  test("正向测试：注册新用户并自动登录", async () => {
    const { key, code } = await captchaPair();
    const username = uniqueName("testreg");
    const result = await AuthAPI.register({
      username,
      password: ADMIN_PASSWORD,
      nickname: "测试注册用户",
      captchaKey: key,
      captchaCode: code,
    });
    expect(result.sessionId).toBeTruthy();
    // 后端注册将用户名规范化为小写，返回可能与传入大小写不同
    expect(result.user.username).toBe(username.toLowerCase());

    // 通过 admin 查找用户 ID 用于清理
    await login(USERS.ADMIN.username);
    const pageResult = await UserAPI.getPage({
      pageNum: 1,
      pageSize: 100,
      keywords: username.toLowerCase(),
    });
    const createdUser = pageResult.list.find((u) => u.username === username.toLowerCase());
    if (createdUser?.id) {
      createdUserIds.push(createdUser.id);
    }
  });

  test("边界：用户名已存在应返回注册失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.register({
        username: USERS.ADMIN.username,
        password: ADMIN_PASSWORD,
        nickname: "重复用户",
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });

  test("边界：验证码错误应返回验证码错误", async () => {
    const captcha = await AuthAPI.getCaptcha();
    await expectBizError(
      AuthAPI.register({
        username: uniqueName("testcaptcha"),
        password: ADMIN_PASSWORD,
        nickname: "验证码测试",
        captchaKey: captcha.captchaKey,
        captchaCode: "0000",
      }),
      ["A0214", "A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });

  test("异常：缺少用户名应返回参数校验失败", async () => {
    const { key, code } = await captchaPair();
    await expectBizError(
      AuthAPI.register({
        username: "",
        password: ADMIN_PASSWORD,
        nickname: "测试",
        captchaKey: key,
        captchaCode: code,
      }),
      ["A0400", "B0001", "ERR_BAD_REQUEST"]
    );
  });
});

describe("POST /api/v1/auth/logout - 用户注销", () => {
  afterAll(async () => {
    // 恢复 admin 登录态
    await login(USERS.ADMIN.username);
  });

  test("正向测试：注销后 Session 失效", async () => {
    await login(USERS.ADMIN.username);
    // logout() 同时清空内存缓存的 sessionId，避免后续 login() 复用已失效会话
    await logout();
    await expect(AuthAPI.getCurrentUser()).rejects.toThrow();
  });
});

describe("GET /api/v1/auth/me - 获取权限信息", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  test("正向测试：获取当前用户权限信息并验证数据完整性", async () => {
    const result = await AuthAPI.getCurrentUser();
    expect(result.userId).toBe(USERS.ADMIN.id);
    expect(result.username).toBe(USERS.ADMIN.username);
    expect(result.nickname).toBe(USERS.ADMIN.nickname);
    expect(Array.isArray(result.roles)).toBe(true);
    expect(result.roles.length).toBeGreaterThan(0);
    expect(result.roles).toContain(ROLES.ADMIN.code);
    expect(Array.isArray(result.perms)).toBe(true);
    expect(result.perms.length).toBeGreaterThan(0);
  });
});

// 对应文档 T-AM-110~120：登录日志分页、按用户名筛选、普通用户仅本人。
describe("GET /api/v1/auth/login-logs - 登录日志查询", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  test("正向测试：管理员查询登录日志分页列表", async () => {
    const result = await AuthAPI.getLoginLogs({ pageNum: 1, pageSize: 10 });
    expect(Array.isArray(result.list)).toBe(true);
    expect(typeof result.total).toBe("number");
    expect(result.total).toBeGreaterThan(0);

    if (result.list.length > 0) {
      const log = result.list[0]!;
      expect(log.username).toBeTruthy();
      expect(log.ip).toBeTruthy();
      expect(log.loginTime).toBeTruthy();
      expect(typeof log.status).toBe("number");
    }
  });

  test("正向测试：按用户名筛选登录日志", async () => {
    const result = await AuthAPI.getLoginLogs({
      pageNum: 1,
      pageSize: 10,
      username: USERS.ADMIN.username,
    });
    expect(result.list.length).toBeGreaterThan(0);
    result.list.forEach((log) => {
      expect(log.username).toBe(USERS.ADMIN.username);
    });
  });

  test("验证：普通用户仅查看本人日志", async () => {
    await login(USERS.USER.username);
    const result = await AuthAPI.getLoginLogs({ pageNum: 1, pageSize: 100 });
    result.list.forEach((log) => {
      expect(log.username).toBe(USERS.USER.username);
    });
    await login(USERS.ADMIN.username);
  });
});

// 规划中：文档 T-AM-090~097 定义多端会话共存（按 deviceType 区分存储、多 Session 并存、管理员踢出）架构，
// 当前 python 后端为单点登录模型（USE_MULTI_POINT=False，无 deviceType 追踪），属超前规划，未实现。保持 skip。
describe.skip("GET /api/v1/auth/sessions - 会话管理 [规划中]", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  test("正向测试：管理员查询用户在线会话", async () => {
    const result = await AuthAPI.getSessions(USERS.ADMIN.username);
    expect(Array.isArray(result)).toBe(true);
    // admin 当前应该有至少一个在线会话
    expect(result.length).toBeGreaterThan(0);
    const session = result[0]!;
    expect(session.sessionId).toBeTruthy();
    expect(session.deviceType).toBeTruthy();
  });

  test("边界：查询不存在用户的会话应返回空", async () => {
    const result = await AuthAPI.getSessions("nonexistent_user_xyz");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBe(0);
  });
});

// 文件级 afterAll：清理登录失败锁定计数，避免污染共享环境
// 本套件的负向登录用例（密码/验证码错误等）会递增 Python 端 login:fail:*（用户+IP 双维度），
// 若不清理，累计 5 次会触发 30 分钟锁定，影响后续所有模块测试。
afterAll(async () => {
  const redis = getRedis();
  const failKeys = await redis.keys("login:fail:*");
  if (failKeys.length > 0) {
    await redis.del(failKeys);
  }
  await disconnectRedis();
});
