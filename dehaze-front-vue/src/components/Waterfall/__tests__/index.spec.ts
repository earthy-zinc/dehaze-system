import { mount, type VueWrapper } from "@vue/test-utils";
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

// Mock utils：局部 mock（保留其余真实导出，防止源码新增依赖后 mock 缺导出而全文件报错）。
// hasIntersectionObserver 在 utils 模块导入时求值（jsdom 下为 false，会走 Lazy
// "不支持 IntersectionObserver" 的抛错分支），强制为 true 并 stub 全局 IO，
// 对齐浏览器的真实懒加载路径。
vi.mock("@/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/utils")>();
  return {
    ...actual,
    hasIntersectionObserver: true,
    assign: vi.fn((target, ...sources) => Object.assign(target, ...sources)),
    getValue: vi.fn((item, selector) => [item[selector]]),
    addClass: vi.fn(),
    hasClass: vi.fn(() => false),
    prefixStyle: vi.fn((prop) => prop),
  };
});

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

    // Mock requestAnimationFrame（帧驱动改为定时器，cancel 对应清理，卸载后可停止循环）
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn((cb) => setTimeout(cb, 16))
    );
    vi.stubGlobal(
      "cancelAnimationFrame",
      vi.fn((id) => clearTimeout(id))
    );

    // jsdom 无 IntersectionObserver，stub 空实现使 Lazy 懒加载挂载路径不抛错
    vi.stubGlobal(
      "IntersectionObserver",
      class {
        disconnect() {}
        observe() {}
        unobserve() {}
      }
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
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

      const lazyImgs = wrapper.findAll(".lazy__box");
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
      // jsdom 会把 #ff0000 归一化为 rgb() 形式
      expect(
        (waterfallList.element as HTMLDivElement).style.backgroundColor
      ).toBe("rgb(255, 0, 0)");
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

      // mock 的 useCalculateCols 返回值恒定，watch 不会因宽度变化触发，
      // 靠 list 变化驱动 watch → renderer(useDebounceFn 已 mock 为直调) → afterRender
      await wrapper.setProps({
        list: [...mockList, { id: "4", src: "image4.jpg" }],
      });
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
    // rAF 被 stub 为 setTimeout(16)，假时钟下按帧推进断言 transform 可观测行为
    // （useLayout mock 高度 800，正向阈值 -(800-700)=-100）
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    function listTransform(wrapper: VueWrapper): string {
      return (wrapper.find(".waterfall-list").element as HTMLDivElement).style
        .transform;
    }

    it("应该正确处理正向滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList, speed: 1 },
      });

      expect(listTransform(wrapper)).toContain("translateY(0px)");
      await vi.advanceTimersByTimeAsync(32);
      expect(listTransform(wrapper)).toContain("translateY(-2px)");
      wrapper.unmount();
    });

    it("应该正确处理反向滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList, speed: -1 },
      });

      // 反向速度 translateY 递增，触及 0 即重置到 -(高度-700)
      await vi.advanceTimersByTimeAsync(16);
      expect(listTransform(wrapper)).toContain("translateY(-100px)");
      wrapper.unmount();
    });

    it("应该在适当条件下重置位置实现无缝滚动", async () => {
      const wrapper = mount(Waterfall, {
        props: { list: mockList, speed: 1 },
      });

      // 连续推进 100 帧越过 -100 阈值后重置回顶部
      await vi.advanceTimersByTimeAsync(16 * 100);
      expect(listTransform(wrapper)).toContain("translateY(0px)");
      wrapper.unmount();
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
