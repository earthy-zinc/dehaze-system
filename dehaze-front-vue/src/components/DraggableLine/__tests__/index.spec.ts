import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import DraggableLine from "../index.vue";

// Mock useSettingsStore
vi.mock("@/store", () => ({
  useSettingsStore: vi.fn(() => ({
    themeColor: "#409eff",
  })),
}));

// Mock hexToRGBA utility
vi.mock("@/utils", () => ({
  hexToRGBA: vi.fn(
    (hex: string, alpha: number) => `rgba(64, 158, 255, ${alpha})`
  ),
}));

// Mock useWindowSize
vi.mock("@vueuse/core", () => ({
  useWindowSize: vi.fn(() => ({
    width: 1920,
    height: 1080,
  })),
}));

// Mock SVGIcon component
vi.mock("@/components/SvgIcon/index.vue", () => ({
  default: {
    template: "<div data-testid='svg-icon'></div>",
    props: ["iconClass", "color", "size"],
  },
}));

describe("DraggableLine Component", () => {
  describe("组件渲染", () => {
    it("应该成功渲染容器元素", () => {
      const wrapper = mount(DraggableLine);

      expect(wrapper.find(".container").exists()).toBe(true);
      expect(wrapper.find("[data-testid='svg-icon']").exists()).toBe(true);
    });

    it("应该正确显示默认标签文本", () => {
      const wrapper = mount(DraggableLine);

      const labels = wrapper.findAll(".drag-label");
      expect(labels[0].text()).toContain("原图");
      expect(labels[1].text()).toContain("对比图");
    });

    it("应该正确应用自定义标签文本", () => {
      const wrapper = mount(DraggableLine, {
        props: {
          leftLabel: "左侧图",
          rightLabel: "右侧图",
        },
      });

      const labels = wrapper.findAll(".drag-label");
      expect(labels[0].text()).toContain("左侧图");
      expect(labels[1].text()).toContain("右侧图");
    });

    it("应该正确设置主题颜色", async () => {
      const wrapper = mount(DraggableLine);
      await nextTick();

      const lineEl = wrapper.find(".line").element as HTMLElement;
      const rightLabelEl = wrapper.findAll(".drag-label")[1]
        .element as HTMLElement;

      expect(lineEl.style.backgroundColor).toBe("rgb(64, 158, 255)"); // 浏览器会将十六进制颜色转换为rgb
      expect(rightLabelEl.style.backgroundColor).toBe(
        "rgba(64, 158, 255, 0.5)"
      );
    });
  });

  describe("拖拽功能", () => {
    beforeEach(() => {
      // Mock getBoundingClientRect
      Element.prototype.getBoundingClientRect = vi.fn(() => ({
        left: 100,
        right: 500,
        width: 400,
        height: 300,
        top: 0,
        bottom: 0,
        x: 100,
        y: 0,
        toJSON: () => {},
      })) as any;
    });

    afterEach(() => {
      vi.clearAllMocks();
    });

    it("应该在鼠标按下时开始拖拽", async () => {
      const wrapper = mount(DraggableLine);
      const container = wrapper.find(".container");

      await container.trigger("mousedown");

      const vm = wrapper.vm as any;
      expect(vm.isDragging).toBe(true);
    });

    it("应该在鼠标释放时停止拖拽", async () => {
      const wrapper = mount(DraggableLine);
      const container = wrapper.find(".container");

      await container.trigger("mousedown");
      await container.trigger("mouseup");

      const vm = wrapper.vm as any;
      expect(vm.isDragging).toBe(false);
    });

    it("应该在拖拽时更新位置", async () => {
      const wrapper = mount(DraggableLine);
      const container = wrapper.find(".container");

      // 开始拖拽
      await container.trigger("mousedown");

      // 模拟鼠标移动事件
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: 200,
      }) as any;

      // 调用 drag 方法
      const vm = wrapper.vm as any;
      vm.drag(mouseEvent);

      expect(vm.parentOffsetLeft).toBe(100); // 200 - 100
    });

    it("应该在非拖拽状态下不更新位置", async () => {
      const wrapper = mount(DraggableLine);
      const vm = wrapper.vm as any;

      const initialOffset = vm.parentOffsetLeft;

      // 模拟鼠标移动事件但未按下鼠标
      const mouseEvent = new MouseEvent("mousemove", {
        clientX: 200,
      }) as any;

      vm.drag(mouseEvent);

      expect(vm.parentOffsetLeft).toBe(initialOffset);
    });
  });

  describe("事件发射", () => {
    it("应该在位置变化时发射 update:offset 事件", async () => {
      const wrapper = mount(DraggableLine);
      const vm = wrapper.vm as any;

      // 修改 parentOffsetLeft 触发 watch
      vm.parentOffsetLeft = 150;
      await nextTick();

      expect(wrapper.emitted("update:offset")).toBeTruthy();
      expect(wrapper.emitted("update:offset")![0]).toEqual([150]);
    });
  });

  describe("动画功能", () => {
    beforeEach(() => {
      // Mock requestAnimationFrame
      vi.useFakeTimers();

      // Mock getBoundingClientRect
      Element.prototype.getBoundingClientRect = vi.fn(() => ({
        left: 0,
        right: 400,
        width: 400,
        height: 300,
        top: 0,
        bottom: 0,
        x: 0,
        y: 0,
        toJSON: () => {},
      })) as any;
    });

    afterEach(() => {
      vi.useRealTimers();
      vi.restoreAllMocks();
    });

    it("应该在挂载时启动动画", async () => {
      const wrapper = mount(DraggableLine);

      // 快进动画
      vi.advanceTimersByTime(3000);
      await nextTick();

      // 动画结束后应该接近0
      const vm = wrapper.vm as any;
      expect(vm.parentOffsetLeft).toBeCloseTo(0, -1); // 允许较大的误差范围
    });

    it("应该正确执行动画函数", async () => {
      const wrapper = mount(DraggableLine);
      const vm = wrapper.vm as any;

      // 设置初始值
      vm.parentOffsetLeft = 100;

      // 调用动画函数，允许一定的误差
      vm.animateOffsetLeft(100, 50, 100);

      // 快进时间
      vi.advanceTimersByTime(100);
      await nextTick();

      // 应该更新 parentOffsetLeft 接近目标值
      expect(vm.parentOffsetLeft).toBeCloseTo(50, -1); // 允许较大的误差范围
    });
  });

  describe("边界情况", () => {
    it("应该隐藏空标签", () => {
      const wrapper = mount(DraggableLine, {
        props: {
          leftLabel: "",
          rightLabel: "",
        },
      });

      // 即使标签文本为空，元素仍然存在，只是不显示文本
      const labels = wrapper.findAll(".drag-label");
      expect(labels[0].text()).toBe("");
      expect(labels[1].text()).toBe("");
    });

    it("应该处理 undefined 标签", () => {
      const wrapper = mount(DraggableLine, {
        props: {
          leftLabel: undefined,
          rightLabel: undefined,
        },
      });

      // 应该显示默认标签
      const labels = wrapper.findAll(".drag-label");
      expect(labels[0].text()).toContain("原图");
      expect(labels[1].text()).toContain("对比图");
    });
  });

  describe("Props 验证", () => {
    it("应该正确设置默认 props", () => {
      const wrapper = mount(DraggableLine);

      expect(wrapper.props().leftLabel).toBe("原图");
      expect(wrapper.props().rightLabel).toBe("对比图");
    });

    it("应该接受字符串类型的标签", () => {
      const wrapper = mount(DraggableLine, {
        props: {
          leftLabel: "Test Left",
          rightLabel: "Test Right",
        },
      });

      expect(wrapper.props().leftLabel).toBe("Test Left");
      expect(wrapper.props().rightLabel).toBe("Test Right");
    });
  });
});
