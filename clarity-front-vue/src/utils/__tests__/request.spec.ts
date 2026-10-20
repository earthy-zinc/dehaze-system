// 会话失效兜底（D12）：HTTP 401 / 业务码 A0230·A0231 时清用户态并送回登录页，
// 失效风暴下只跳一次；已在登录页时不跳转，原样提示后端文案。
import { flushPromises } from "@vue/test-utils";
import type { AxiosError } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { configAxios, ResultEnum } from "clarity-sdk-js";
import { useUserStore } from "@/store/modules/user";
import router from "@/router";
import configRequest from "../request";

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return { ...actual, configAxios: vi.fn() };
});

vi.mock("element-plus", async (importOriginal) => {
  const actual = await importOriginal<typeof import("element-plus")>();
  return {
    ...actual,
    ElMessage: { error: vi.fn(), warning: vi.fn() },
  };
});

vi.mock("@/router", () => ({
  default: {
    currentRoute: { value: { path: "/home", fullPath: "/home?a=1" } },
    replace: vi.fn(() => Promise.resolve()),
  },
}));

/** 取最近一次 configRequest 注册的拦截器 */
function interceptors() {
  configRequest();
  const calls = vi.mocked(configAxios).mock.calls;
  return calls[calls.length - 1]![0];
}

beforeEach(async () => {
  vi.mocked(router.replace).mockClear();
  router.currentRoute.value.path = "/home";
  router.currentRoute.value.fullPath = "/home?a=1";
  const store = useUserStore();
  store.user.roles = ["ROOT"];
  await flushPromises();
});

describe("会话失效兜底", () => {
  it("业务码 A0230：清用户态并带 redirect 跳登录页", async () => {
    const hooks = interceptors();

    hooks.onBizError?.(ResultEnum.TOKEN_INVALID, "token无效或已过期");
    await flushPromises();

    expect(useUserStore().user.roles).toEqual([]);
    expect(router.replace).toHaveBeenCalledWith({
      path: "/login",
      query: { redirect: "/home?a=1" },
    });
  });

  it("HTTP 401：同样接管（用户名下的会话 cookie 已失效）", async () => {
    const hooks = interceptors();

    // 夹具只需 status/data，其余 Axios 响应字段非本用例关注点
    const unauthorized = {
      response: { status: 401, data: {} },
    } as unknown as AxiosError;
    await hooks.onResponseError?.(unauthorized).catch(() => {});
    await flushPromises();

    expect(useUserStore().user.roles).toEqual([]);
    expect(router.replace).toHaveBeenCalledTimes(1);
  });

  it("失效风暴：并发失效只跳一次", async () => {
    const hooks = interceptors();

    hooks.onBizError?.(ResultEnum.TOKEN_INVALID, "token无效或已过期");
    hooks.onBizError?.(ResultEnum.TOKEN_INVALID, "token无效或已过期");
    hooks.onBizError?.(ResultEnum.TOKEN_ACCESS_FORBIDDEN, "token已被禁止访问");
    await flushPromises();

    expect(router.replace).toHaveBeenCalledTimes(1);
  });

  it("已在登录页：不跳转、不重复清理，原样提示后端文案", async () => {
    router.currentRoute.value.path = "/login";
    router.currentRoute.value.fullPath = "/login";
    const hooks = interceptors();
    const { ElMessage } = await import("element-plus");

    hooks.onBizError?.(ResultEnum.TOKEN_INVALID, "token无效或已过期");
    await flushPromises();

    expect(router.replace).not.toHaveBeenCalled();
    expect(ElMessage.error).toHaveBeenCalledWith("token无效或已过期");
  });

  it("非会话类业务码：只提示，不动用户态", async () => {
    const hooks = interceptors();
    const { ElMessage } = await import("element-plus");

    hooks.onBizError?.("A0400", "图片地址不允许访问内网资源");
    await flushPromises();

    expect(ElMessage.error).toHaveBeenCalledWith("图片地址不允许访问内网资源");
    expect(router.replace).not.toHaveBeenCalled();
    expect(useUserStore().user.roles).toEqual(["ROOT"]);
  });
});
