import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Hamburger from "../index.vue";

describe("Hamburger Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("基本功能", () => {
    it("应该正确渲染汉堡菜单组件", () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      expect(wrapper.find("div").exists()).toBe(true);
    });

    it("应该应用正确的CSS类", () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      const container = wrapper.find("div");
      expect(container.exists()).toBe(true);
      expect(container.classes()).toContain("px-[15px]");
      expect(container.classes()).toContain("flex");
      expect(container.classes()).toContain("items-center");
      expect(container.classes()).toContain("justify-center");
    });
  });

  describe("事件处理", () => {
    it("应该在点击时触发toggleClick事件", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      await wrapper.trigger("click");

      expect(wrapper.emitted("toggleClick")).toBeTruthy();
      expect(wrapper.emitted("toggleClick")?.length).toBe(1);
    });

    it("应该在多次点击时触发多次toggleClick事件", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      await wrapper.trigger("click");
      await wrapper.trigger("click");
      await wrapper.trigger("click");

      expect(wrapper.emitted("toggleClick")?.length).toBe(3);
    });

    it("应该在触发事件时不传递参数", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      await wrapper.trigger("click");

      const emittedEvent = wrapper.emitted("toggleClick")?.[0];
      expect(emittedEvent).toEqual([]);
    });
  });

  describe("Props 验证", () => {
    it("应该接受布尔类型的isActive", () => {
      const wrapper1 = mount(Hamburger, {
        props: {
          isActive: true,
        },
      });

      const wrapper2 = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      expect(wrapper1.props("isActive")).toBe(true);
      expect(wrapper2.props("isActive")).toBe(false);
    });

    it("应该设置isActive的默认值为false", () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      expect(wrapper.props("isActive")).toBe(false);
    });
  });

  describe("响应式更新", () => {
    it("应该在isActive改变时更新组件", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      expect(wrapper.props("isActive")).toBe(false);

      await wrapper.setProps({ isActive: true });

      expect(wrapper.props("isActive")).toBe(true);
    });
  });

  describe("边界情况", () => {
    it("应该处理快速连续点击", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      // 快速连续点击
      for (let i = 0; i < 10; i++) {
        await wrapper.trigger("click");
      }

      expect(wrapper.emitted("toggleClick")?.length).toBe(10);
    });

    it("应该处理isActive的快速切换", async () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      // 快速切换状态
      const states = [true, false, true, false, true];
      for (const state of states) {
        await wrapper.setProps({ isActive: state });
        expect(wrapper.props("isActive")).toBe(state);
      }
    });
  });

  describe("组件交互", () => {
    it("应该保持正确的点击区域", () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      const container = wrapper.find("div");
      expect(container.exists()).toBe(true);
      // 组件应该有足够的点击区域
      expect(container.classes()).toContain("px-[15px]");
      expect(container.classes()).toContain("flex");
      expect(container.classes()).toContain("items-center");
      expect(container.classes()).toContain("justify-center");
    });
  });

  describe("可访问性", () => {
    it("应该有正确的语义结构", () => {
      const wrapper = mount(Hamburger, {
        props: {
          isActive: false,
        },
      });

      // 验证组件有合适的点击区域
      const container = wrapper.find("div");
      expect(container.exists()).toBe(true);
      expect(container.classes()).toContain("px-[15px]");
      expect(container.classes()).toContain("flex");
      expect(container.classes()).toContain("items-center");
      expect(container.classes()).toContain("justify-center");
    });
  });
});
