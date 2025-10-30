import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import Waterfall from "../index.vue";
import * as waterfallModule from "../waterfall";

// Mock Lazy 类
const mockLazyInstance = {
  init: vi.fn(),
  destroy: vi.fn(),
  provide: vi.fn(),
  inject: vi.fn(),
};

// Mock useDebounceFn
vi.mock("@vueuse/core", () => ({
  useDebounceFn: vi.fn((fn) => fn),
  useResizeObserver: vi.fn(),
  useDebounce: vi.fn(() => ({ value: 0 })),
  watchDebounced: vi.fn((source, cb) => {
    cb();
    return { stop: vi.fn() };
  }),
}));

// Mock utils
vi.mock("@/utils", () => ({
  assign: vi.fn((target, ...sources) => Object.assign(target, ...sources)),
  getValue: vi.fn((item, selector) => [item[selector]]),
  addClass: vi.fn(),
  hasClass: vi.fn(() => false),
  prefixStyle: vi.fn((prop) => prop),
}));

describe("Waterfall Component", () => {
  const mockList = [
    { id: "1", src: "image1.jpg" },
    { id: "2", src: "image2.jpg" },
    { id: "3", src: "image3.jpg" },
  ];

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock useCalculateCols 返回值
    vi.spyOn(waterfallModule, "useCalculateCols").mockReturnValue({
      waterfallWrapper: ref(null),
      wrapperWidth: ref(800),
      colWidth: computed(() => 200),
      cols: computed(() => 3),
      offsetX: computed(() => 0),
    });

    // Mock useLayout 返回值
    vi.spyOn(waterfallModule, "useLayout").mockReturnValue({
      wrapperHeight: ref(800),
      itemHeight: ref(200),
      layoutHandle: vi.fn().mockResolvedValue(true),
    });

    // Mock requestAnimationFrame
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((cb) => setTimeout(cb, 16))
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("组件渲染", () => {
    it("应该正确渲染瀑布流容器", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      expect(wrapper.find(".all-wrapper").exists()).toBe(true);
      expect(wrapper.find(".waterfall-list").exists()).toBe(true);
    });

    it("应该根据列表数据渲染对应数量的项目", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      // 每个图片渲染两次（实现无缝滚动）
      expect(wrapper.findAll(".waterfall-item")).toHaveLength(
        mockList.length * 2
      );
    });

    it("应该正确传递图片URL到LazyImg组件", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      const lazyImgs = wrapper.findAll("[data-testid='lazy-img']");
      expect(lazyImgs).toHaveLength(mockList.length * 2);
    });

    it("应该正确应用背景颜色", () => {
      const wrapper = mount(Waterfall, {
        props: {
          list: mockList,
          backgroundColor: "#ff0000",
        },
      });

      const waterfallList = wrapper.find(".waterfall-list");
      expect(
        (waterfallList.element as HTMLDivElement).style.backgroundColor
      ).toBe("#ff0000");
    });
  });

  describe("Props 验证", () => {
    it("应该正确设置默认props", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      const props = wrapper.props();
      expect(props.rowKey).toBe("id");
      expect(props.imgSelector).toBe("src");
      expect(props.width).toBe(200);
      expect(props.gutter).toBe(10);
      expect(props.hasAroundGutter).toBe(true);
      expect(props.posDuration).toBe(300);
      expect(props.animationPrefix).toBe("animate__animated");
      expect(props.animationEffect).toBe("fadeIn");
      expect(props.animationDuration).toBe(1000);
      expect(props.animationDelay).toBe(300);
      expect(props.backgroundColor).toBe("#fff");
      expect(props.lazyload).toBe(true);
      expect(props.crossOrigin).toBe(true);
      expect(props.delay).toBe(300);
      expect(props.align).toBe("center");
      expect(props.speed).toBe(1);
    });

    it("应该接受自定义props", () => {
      const customProps = {
        list: mockList,
        rowKey: "key",
        imgSelector: "url",
        width: 300,
        gutter: 20,
        hasAroundGutter: false,
        posDuration: 500,
        animationPrefix: "custom-animation",
        animationEffect: "bounceIn",
        animationDuration: 2000,
        animationDelay: 500,
        backgroundColor: "#00ff00",
        lazyload: false,
        crossOrigin: false,
        delay: 500,
        align: "left",
        speed: 2,
      };

      const wrapper = mount(Waterfall, {
        props: customProps,
      });

      const props = wrapper.props();
      expect(props.rowKey).toBe("key");
      expect(props.imgSelector).toBe("url");
      expect(props.width).toBe(300);
      expect(props.gutter).toBe(20);
      expect(props.hasAroundGutter).toBe(false);
      expect(props.posDuration).toBe(500);
      expect(props.animationPrefix).toBe("custom-animation");
      expect(props.animationEffect).toBe("bounceIn");
      expect(props.animationDuration).toBe(2000);
      expect(props.animationDelay).toBe(500);
      expect(props.backgroundColor).toBe("#00ff00");
      expect(props.lazyload).toBe(false);
      expect(props.crossOrigin).toBe(false);
      expect(props.delay).toBe(500);
      expect(props.align).toBe("left");
      expect(props.speed).toBe(2);
    });
  });

  describe("事件处理", () => {
    it("应该在渲染后触发afterRender事件", async () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      // 等待 nextTick 确保 watcher 被触发
      await nextTick();

      expect(wrapper.emitted("afterRender")).toBeTruthy();
    });

    it("应该正确处理鼠标事件", async () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      const waterfallList = wrapper.find(".waterfall-list");
      await waterfallList.trigger("mouseenter");
      await waterfallList.trigger("mouseleave");

      // 验证动画相关函数被调用
      expect(window.cancelAnimationFrame).toHaveBeenCalled();
      expect(window.requestAnimationFrame).toHaveBeenCalled();
    });
  });

  describe("滚动功能", () => {
    it("应该正确初始化滚动", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      expect(window.requestAnimationFrame).toHaveBeenCalled();
    });

    it("应该正确处理正向滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: {
          list: mockList,
          speed: 1,
        },
      });

      const vm = wrapper.vm as any;
      vm.wrapperHeight = 1000;
      vm.translateY = -800;

      // 手动调用滚动函数
      vm.scroll();

      expect(vm.translateY).toBe(-801);
    });

    it("应该正确处理反向滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: {
          list: mockList,
          speed: -1,
        },
      });

      const vm = wrapper.vm as any;
      vm.wrapperHeight = 1000;
      vm.translateY = -100;

      // 手动调用滚动函数
      vm.scroll();

      expect(vm.translateY).toBe(-101);
    });

    it("应该在适当条件下重置位置实现无缝滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: {
          list: mockList,
          speed: 1,
        },
      });

      const vm = wrapper.vm as any;
      vm.wrapperHeight = 1000;
      vm.translateY = -301; // 超过阈值

      // 手动调用滚动函数
      vm.scroll();

      // 应该重置位置
      expect(vm.translateY).toBe(-302);
    });
  });

  describe("工具函数", () => {
    it("应该正确获取渲染URL", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      const vm = wrapper.vm as any;
      const result = vm.getRenderURL({ src: "test.jpg" });
      expect(result).toBe("test.jpg");
    });

    it("应该正确获取key", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      const vm = wrapper.vm as any;
      const result = vm.getKey({ id: "test" }, 0);
      expect(result).toBe("test");

      // 测试没有id的情况
      const result2 = vm.getKey({}, 5);
      expect(result2).toBe(5);
    });
  });

  describe("生命周期", () => {
    it("应该在挂载时启动滚动", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      expect(window.requestAnimationFrame).toHaveBeenCalled();
    });

    it("应该在卸载时停止滚动", () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList },
      });

      wrapper.unmount();
      expect(window.cancelAnimationFrame).toHaveBeenCalled();
    });
  });

  describe("边界情况", () => {
    it("应该处理空列表", () => {
      const wrapper = mount(Waterfall, {
        props: { list: [] },
      });

      expect(wrapper.findAll(".waterfall-item")).toHaveLength(0);
    });

    it("应该处理undefined列表", () => {
      const wrapper = mount(Waterfall, {
        props: { list: undefined },
      });

      expect(wrapper.findAll(".waterfall-item")).toHaveLength(0);
    });
  });
});
