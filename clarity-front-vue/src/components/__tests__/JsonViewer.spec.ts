import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import JsonViewer from "../JsonViewer.vue";

// el-alert/el-button 以轻量桩渲染（与 ImportExportToolbar spec 同模式）
const elStubs = {
  "el-alert": {
    template: '<div class="alert">{{ title }}</div>',
    props: ["title", "type", "closable"],
  },
  "el-button": {
    template: "<button @click=\"$emit('click')\"><slot /></button>",
    props: ["link", "size", "icon"],
    emits: ["click"],
  },
};

const mountViewer = (props: Record<string, unknown> = {}) =>
  mount(JsonViewer, {
    props,
    global: { stubs: elStubs },
  });

describe("JsonViewer", () => {
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;
  const clipboardWriteText = vi.fn();

  beforeEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: clipboardWriteText },
      configurable: true,
    });
    URL.createObjectURL = vi.fn(() => "blob:mock");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
    vi.clearAllMocks();
  });

  describe("空态降级", () => {
    it("data 为 null 时显示空态提示", () => {
      const wrapper = mountViewer({ data: null });

      expect(wrapper.find(".json-viewer--empty").exists()).toBe(true);
      expect(wrapper.text()).toContain("无原始报文记录");
      expect(wrapper.find(".json-viewer__body").exists()).toBe(false);
    });

    it("data 为 undefined 时显示自定义空态文案", () => {
      const wrapper = mountViewer({ data: undefined, emptyText: "无详情数据" });

      expect(wrapper.text()).toContain("无详情数据");
    });

    it("data 为空对象时正常渲染（非空态）", () => {
      const wrapper = mountViewer({ data: {} });

      expect(wrapper.find(".json-viewer--empty").exists()).toBe(false);
      expect(wrapper.find(".json-node--container").exists()).toBe(true);
    });
  });

  describe("渲染与折叠", () => {
    it("渲染对象与嵌套数组，默认展开首层", () => {
      const wrapper = mountViewer({
        data: { model: "qwen3-0.6b", messages: [{ role: "user" }] },
      });

      // 首层键可见
      expect(wrapper.text()).toContain("model");
      expect(wrapper.text()).toContain("qwen3-0.6b");
      // 深层默认折叠（messages 数组内容为折叠占位）
      expect(wrapper.text()).toContain("…]");
    });

    it("expandDepth=0 时全层折叠，点击切换展开", async () => {
      const wrapper = mountViewer({
        data: { model: "qwen3-0.6b" },
        expandDepth: 0,
      });

      expect(wrapper.text()).toContain("…}");
      expect(wrapper.text()).not.toContain("qwen3-0.6b");

      await wrapper.find(".json-node__toggle").trigger("click");
      expect(wrapper.text()).toContain("qwen3-0.6b");
    });

    it("字符串值按 JSON.stringify 渲染", () => {
      const wrapper = mountViewer({ data: "plain" });

      expect(wrapper.text()).toContain('"plain"');
    });
  });

  describe("复制", () => {
    it("点击复制按钮写入剪贴板", async () => {
      const wrapper = mountViewer({ data: { a: 1 } });

      await wrapper.find(".json-viewer__toolbar button").trigger("click");

      expect(clipboardWriteText).toHaveBeenCalledWith(
        JSON.stringify({ a: 1 }, null, 2)
      );
    });
  });

  describe("下载", () => {
    it("点击下载按钮生成 JSON blob 链接", async () => {
      const wrapper = mountViewer({ data: { a: 1 }, filename: "req.json" });
      const clickSpy = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(() => {});

      const buttons = wrapper.findAll(".json-viewer__toolbar button");
      await buttons[1].trigger("click");

      expect(URL.createObjectURL).toHaveBeenCalled();
      expect(clickSpy).toHaveBeenCalled();
      const link = clickSpy.mock.instances[0] as HTMLAnchorElement;
      expect(link.download).toBe("req.json");
    });
  });
});
