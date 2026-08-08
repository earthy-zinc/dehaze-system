/**
 * React Native 环境兼容测试。
 *
 * RN 特征：window === globalThis（无 DOM API：location/addEventListener/PerformanceObserver）、
 * navigator.product === "ReactNative"、ErrorUtils 全局错误处理。
 * 验证：install 不抛错、ErrorUtils handler 链式调用原 handler（不吞 RN 红屏）、日志字段不读 DOM API。
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import { CaptureTransport, MemoryStore } from "#/unit/logger/helpers";

describe("React Native 环境兼容", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  /** 模拟 RN 全局特征后重新加载模块（env 常量在模块加载时求值，需 resetModules） */
  async function importLoggerInRnEnv() {
    vi.stubGlobal("navigator", { product: "ReactNative" });
    vi.stubGlobal("window", globalThis); // RN: window 是 globalThis 别名，无 DOM API
    vi.resetModules();
    const mod = await import("@/logger");
    return mod.Logger;
  }

  test("install 在 RN 环境不抛错，日志字段不读 DOM API", async () => {
    const Logger = await importLoggerInRnEnv();
    const transport = new CaptureTransport();
    expect(() =>
      Logger.install({ app: "rn", storage: new MemoryStore(), transports: [transport] })
    ).not.toThrow();

    Logger.getInstance()!.error("测试错误");
    const entry = transport.logs.find((l) => l.message === "测试错误")!;
    expect(entry.url).toBe(""); // 不读 window.location
    expect(entry.user_agent).toBe("ReactNative"); // 不读 navigator.userAgent
    Logger.reset();
  });

  test("ErrorUtils handler 链式调用原 handler，不吞 RN 红屏/崩溃上报", async () => {
    const originalHandler = vi.fn();
    let registered: ((error: Error, isFatal: boolean) => void) | undefined;
    vi.stubGlobal("ErrorUtils", {
      getGlobalHandler: () => originalHandler,
      setGlobalHandler: (h: typeof registered) => {
        registered = h;
      },
    });
    const Logger = await importLoggerInRnEnv();
    Logger.install({
      app: "rn",
      storage: new MemoryStore(),
      transports: [new CaptureTransport()],
    });

    expect(registered).toBeTypeOf("function");
    const error = new Error("render boom");
    registered!(error, true);
    expect(originalHandler).toHaveBeenCalledWith(error, true);
    Logger.reset();
  });
});
