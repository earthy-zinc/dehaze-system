// 登录页守卫回归（D12）：内存 roles 可能是已失效会话的残留，
// 访问 /login 必须由服务端确认后才能弹回首页，否则失效用户被永久挡在登录页外。
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UserAPI } from "clarity-sdk-js";
import router from "@/router";
import { useUserStore } from "@/store/modules/user";
import { setupPermission } from "../permission";

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return {
    ...actual,
    UserAPI: { ...actual.UserAPI, getInfo: vi.fn() },
    MenuAPI: { ...actual.MenuAPI, getRoutes: vi.fn() },
  };
});

vi.mock("@/router", () => ({
  default: {
    beforeEach: vi.fn(),
    afterEach: vi.fn(),
    hasRoute: vi.fn(() => true),
    addRoute: vi.fn(),
    removeRoute: vi.fn(),
  },
}));

vi.mock("@/utils/nprogress", () => ({
  default: { start: vi.fn(), done: vi.fn() },
}));

/** 取出 setupPermission 注册的全局前置守卫 */
function guard() {
  setupPermission();
  const calls = vi.mocked(router.beforeEach).mock.calls;
  return calls[calls.length - 1]![0] as (
    to: { path: string },
    from: unknown,
    next: (arg?: unknown) => void
  ) => Promise<void>;
}

beforeEach(() => {
  vi.clearAllMocks();
  useUserStore().user.roles = ["ROOT"];
});

describe("登录页守卫", () => {
  it("会话仍有效：服务端确认后弹回首页", async () => {
    vi.mocked(UserAPI.getInfo).mockResolvedValue({
      userId: 1,
      username: "admin",
      roles: ["ROOT"],
    } as never);
    const next = vi.fn();

    await guard()({ path: "/login" }, {}, next);

    expect(next).toHaveBeenCalledWith({ path: "/" });
    expect(useUserStore().user.roles).toEqual(["ROOT"]);
  });

  it("内存 roles 为失效残留：放行登录页并清空用户态", async () => {
    vi.mocked(UserAPI.getInfo).mockRejectedValue(
      Object.assign(new Error("token无效或已过期"), {
        response: { status: 401, data: { code: "A0230" } },
      })
    );
    const next = vi.fn();

    await guard()({ path: "/login" }, {}, next);

    expect(next).toHaveBeenCalledWith();
    expect(useUserStore().user.roles).toEqual([]);
  });

  it("未登录：直接放行（不发确认请求）", async () => {
    useUserStore().user.roles = [];
    const next = vi.fn();

    await guard()({ path: "/login" }, {}, next);

    expect(next).toHaveBeenCalledWith();
    expect(UserAPI.getInfo).not.toHaveBeenCalled();
  });
});
