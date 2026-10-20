/**
 * 小程序环境兼容测试。
 *
 * 覆盖两类运行时：
 * - 微信系（globalThis.wx）：url/user_agent/onError/storage 全部走 wx 同步 API
 * - 裸小程序（无 wx/uni/window/navigator，如支付宝/抖音）：全部降级不崩溃
 *
 * vitest unit 默认 node 环境（无 window），天然符合小程序"无 DOM"特征。
 */
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Logger } from "@/logger";
import { CaptureTransport, MemoryStore } from "#/unit/logger/helpers";

beforeEach(() => {
  Logger.reset();
});

afterEach(() => {
  Logger.reset();
  vi.unstubAllGlobals();
});

describe("微信小程序环境（globalThis.wx）", () => {
  function stubWx() {
    const onErrorHandlers: Array<(error: string) => void> = [];
    vi.stubGlobal("wx", {
      getCurrentPages: () => [{ route: "pages/index/index" }],
      getSystemInfoSync: () => ({ model: "iPhone 15" }),
      onError: (h: (error: string) => void) => onErrorHandlers.push(h),
    });
    return onErrorHandlers;
  }

  test("install 不抛错，url/user_agent 走 wx API", () => {
    stubWx();
    const transport = new CaptureTransport();
    expect(() =>
      Logger.install({ app: "taro", storage: new MemoryStore(), transports: [transport] })
    ).not.toThrow();

    Logger.getInstance()!.error("测试错误");
    const entry = transport.logs.find((l) => l.message === "测试错误")!;
    expect(entry.url).toBe("/pages/index/index");
    expect(entry.user_agent).toBe("wx iPhone 15");
  });

  test("wx.onError 注册并转发脚本异常", () => {
    const onErrorHandlers = stubWx();
    const transport = new CaptureTransport();
    Logger.install({ app: "taro", storage: new MemoryStore(), transports: [transport] });

    expect(onErrorHandlers).toHaveLength(1);
    onErrorHandlers[0]!("ReferenceError: xxx is not defined");
    const entry = transport.logs.find((l) => l.error_source === "wx_on_error");
    expect(entry).toBeDefined();
    expect(entry!.message).toContain("小程序脚本异常");
  });
});

describe("裸小程序环境（无 wx/uni/window，如支付宝/抖音）", () => {
  test("install 不抛错，环境字段降级为空字符串", () => {
    // node 22 自带 navigator，stub 掉以贴近真实裸小程序（无 window/navigator/wx/uni）
    vi.stubGlobal("navigator", undefined);
    const transport = new CaptureTransport();
    expect(() =>
      Logger.install({ app: "miniapp", storage: new MemoryStore(), transports: [transport] })
    ).not.toThrow();

    Logger.getInstance()!.error("测试错误");
    const entry = transport.logs.find((l) => l.message === "测试错误")!;
    expect(entry.url).toBe("");
    expect(entry.user_agent).toBe("");
  });
});
