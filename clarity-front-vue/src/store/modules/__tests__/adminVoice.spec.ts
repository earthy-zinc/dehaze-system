// 语音服务监控轮询契约（B-4）：重复 start 不得叠加定时器，stop 后必须彻底停轮询。
// 组件侧由 VoiceServiceMonitor 的 onActivated/onDeactivated 驱动（页面 keepAlive 缓存）。
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VoiceAPI } from "clarity-sdk-js";
import { useAdminVoiceStore } from "../adminVoice";

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return {
    ...actual,
    VoiceAPI: { ...actual.VoiceAPI, getServiceStatus: vi.fn() },
  };
});

describe("adminVoice 服务状态轮询", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(VoiceAPI.getServiceStatus).mockResolvedValue({
      asr: { status: "online" },
    } as never);
  });

  it("重复 startMonitor 不叠加定时器，stopMonitor 后停止", async () => {
    const store = useAdminVoiceStore();

    store.startMonitor();
    store.startMonitor();
    await vi.advanceTimersByTimeAsync(0);
    expect(VoiceAPI.getServiceStatus).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(30_000);
    expect(VoiceAPI.getServiceStatus).toHaveBeenCalledTimes(2);

    store.stopMonitor();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(VoiceAPI.getServiceStatus).toHaveBeenCalledTimes(2);

    // 重新激活可再次启动（KeepAlive 复访场景）
    store.startMonitor();
    await vi.advanceTimersByTimeAsync(0);
    expect(VoiceAPI.getServiceStatus).toHaveBeenCalledTimes(3);

    store.stopMonitor();
    vi.useRealTimers();
  });
});
