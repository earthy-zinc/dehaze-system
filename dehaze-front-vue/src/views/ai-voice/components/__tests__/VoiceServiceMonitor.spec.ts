// 语音监控轮询泄漏回归（B-4）：页面被 KeepAlive 缓存，切走时只失活不卸载，
// 组件必须在 deactivated 停轮询、activated 恢复，否则切走后仍持续请求 /voice/service/status。
import { mount } from "@vue/test-utils";
import { defineComponent, nextTick, ref } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { elStubs } from "@/components/ai-chat/__tests__/stubs";
import VoiceServiceMonitor from "../VoiceServiceMonitor.vue";

const startMonitor = vi.fn();
const stopMonitor = vi.fn();

vi.mock("@/store/modules/adminVoice", () => ({
  useAdminVoiceStore: () => ({
    startMonitor,
    stopMonitor,
    fetchServiceStatus: vi.fn(),
    statusLoading: false,
    serviceStatus: null,
  }),
}));

/** 宿主：把被缓存的监控组件放进 KeepAlive 内，用 v-if 切换来触发失活/激活 */
const Host = defineComponent({
  components: { VoiceServiceMonitor },
  setup() {
    const visible = ref(true);
    return { visible };
  },
  template: `<KeepAlive><VoiceServiceMonitor v-if="visible" /></KeepAlive>`,
});

describe("VoiceServiceMonitor 轮询生命周期", () => {
  beforeEach(() => {
    startMonitor.mockClear();
    stopMonitor.mockClear();
  });

  it("首次进入启动轮询，KeepAlive 失活停止、复访恢复", async () => {
    const wrapper = mount(Host, {
      global: { stubs: { ...elStubs }, directives: { loading: {} } },
    });
    await nextTick();
    // 首次挂载 mounted 与 activated 都会触发（store 侧有 monitorTimer 去重，不会叠加定时器）
    const started = startMonitor.mock.calls.length;
    expect(started).toBeGreaterThan(0);
    expect(stopMonitor).not.toHaveBeenCalled();

    // 切走：组件被缓存（不卸载），必须停轮询
    wrapper.vm.visible = false;
    await nextTick();
    expect(stopMonitor).toHaveBeenCalledTimes(1);

    // 复访：恢复轮询
    wrapper.vm.visible = true;
    await nextTick();
    expect(startMonitor.mock.calls.length).toBeGreaterThan(started);
  });
});
