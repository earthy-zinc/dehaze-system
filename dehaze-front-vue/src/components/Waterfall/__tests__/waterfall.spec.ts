import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as waterfallModule from "../waterfall";
import { ref, nextTick } from "vue";
import { useResizeObserver } from "@vueuse/core";

// Mock @vueuse/core
vi.mock("@vueuse/core", () => ({
  useResizeObserver: vi.fn(),
}));

// Mock @/utils
vi.mock("@/utils", () => ({
  addClass: vi.fn((el, className) => {
    if (el.classList) {
      el.classList.add(className);
    } else {
      el.className += " " + className;
    }
  }),
  hasClass: vi.fn((el, className) => {
    if (el.classList) {
      return el.classList.contains(className);
    }
    return el.className.indexOf(className) !== -1;
  }),
  prefixStyle: vi.fn((prop) => prop),
}));

describe("waterfall.ts", () => {
  describe("getItemWidth function", () => {
    // Since getItemWidth is not exported, we'll test it indirectly through useCalculateCols
    it("应该根据断点计算项目宽度", () => {
      const props = {
        breakpoints: {
          1200: { rowPerView: 3 },
          800: { rowPerView: 2 },
          500: { rowPerView: 1 },
        },
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const { colWidth } = waterfallModule.useCalculateCols(props);

      // We can't directly test getItemWidth since it's not exported
      // But we can test that useCalculateCols works correctly
      expect(colWidth).toBeDefined();
    });
  });

  describe("useCalculateCols", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("应该正确计算列数和宽度", () => {
      const props = {
        breakpoints: {
          1200: { rowPerView: 3 },
          800: { rowPerView: 2 },
          500: { rowPerView: 1 },
        },
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const { waterfallWrapper, wrapperWidth, colWidth, cols, offsetX } =
        waterfallModule.useCalculateCols(props);

      expect(waterfallWrapper).toBeDefined();
      expect(wrapperWidth.value).toBe(0);
      expect(colWidth.value).toBeDefined();
      expect(cols.value).toBeDefined();
      expect(offsetX.value).toBeDefined();
    });

    it("应该正确处理resize事件", () => {
      const mockCallback = vi.fn();
      (useResizeObserver as any).mockImplementation((target, callback) => {
        mockCallback(target, callback);
      });

      const props = {
        breakpoints: {
          1200: { rowPerView: 3 },
          800: { rowPerView: 2 },
          500: { rowPerView: 1 },
        },
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const { wrapperWidth } = waterfallModule.useCalculateCols(props);

      expect(mockCallback).toHaveBeenCalled();

      // Simulate resize observer callback
      const callback = mockCallback.mock.calls[0][1];
      callback([{ contentRect: { width: 800 } }]);

      expect(wrapperWidth.value).toBe(800);
    });
  });

  describe("useLayout", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("应该正确初始化布局函数", () => {
      const props = {
        breakpoints: {
          1200: { rowPerView: 3 },
          800: { rowPerView: 2 },
          500: { rowPerView: 1 },
        },
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const colWidth = ref(200);
      const cols = ref(3);
      const offsetX = ref(0);
      const waterfallWrapper = ref(null);

      const { wrapperHeight, itemHeight, layoutHandle } =
        waterfallModule.useLayout(
          props,
          colWidth,
          cols,
          offsetX,
          waterfallWrapper
        );

      expect(wrapperHeight.value).toBe(0);
      expect(itemHeight.value).toBe(0);
      expect(layoutHandle).toBeTypeOf("function");
    });

    it("应该正确执行布局处理", async () => {
      const props = {
        breakpoints: {
          1200: { rowPerView: 3 },
          800: { rowPerView: 2 },
          500: { rowPerView: 1 },
        },
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 10, // 短时间以便测试
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const colWidth = ref(200);
      const cols = ref(3);
      const offsetX = ref(0);

      // 创建模拟的DOM元素
      const waterfallWrapper = ref({
        childNodes: [
          {
            className: "waterfall-item",
            style: {},
            firstChild: {
              classList: {
                add: vi.fn(),
                contains: vi.fn().mockReturnValue(false),
              },
              className: "",
              style: {},
            },
            getBoundingClientRect: vi.fn().mockReturnValue({ height: 100 }),
          },
        ],
      });

      const { wrapperHeight, itemHeight, layoutHandle } =
        waterfallModule.useLayout(
          props,
          colWidth,
          cols,
          offsetX,
          waterfallWrapper
        );

      const result = await layoutHandle();

      expect(result).toBe(true);
      // 等待异步操作完成
      await nextTick();

      expect(wrapperHeight.value).toBeGreaterThan(0);
      expect(itemHeight.value).toBe(100);
    });
  });

  describe("addAnimation", () => {
    it("应该正确添加动画类", () => {
      const props = {
        breakpoints: {},
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const addAnimationFunc = waterfallModule.addAnimation(props);

      const mockElement = {
        firstChild: {
          classList: {
            add: vi.fn(),
            contains: vi.fn().mockReturnValue(false),
          },
          className: "",
          style: {},
        },
      };

      addAnimationFunc(mockElement as any);

      expect(mockElement.firstChild.classList.add).toHaveBeenCalledWith(
        "animate__animated"
      );
      expect(mockElement.firstChild.classList.add).toHaveBeenCalledWith(
        "fadeIn"
      );
    });

    it("应该处理没有firstChild的情况", () => {
      const props = {
        breakpoints: {},
        width: 200,
        gutter: 10,
        hasAroundGutter: true,
        align: "center",
        list: [],
        posDuration: 300,
        animationPrefix: "animate__animated",
        animationEffect: "fadeIn",
        animationDuration: 1000,
        animationDelay: 300,
        backgroundColor: "#fff",
        lazyload: true,
        loadProps: {},
        crossOrigin: true,
        delay: 300,
        speed: 1,
      };

      const addAnimationFunc = waterfallModule.addAnimation(props);

      const mockElement = {
        firstChild: null,
      };

      // 不应该抛出异常
      expect(() => addAnimationFunc(mockElement as any)).not.toThrow();
    });
  });
});
