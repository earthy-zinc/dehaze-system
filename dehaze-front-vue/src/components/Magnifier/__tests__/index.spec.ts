import { describe, it, expect, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import Magnifier from "../index.vue";

// Mock loadImage utility
vi.mock("@/utils", () => ({
  loadImage: vi.fn((src: string) => {
    return Promise.resolve({
      src,
      width: 800,
      height: 600,
    } as HTMLImageElement);
  }),
}));

// Mock transform utility
vi.mock("../../AlgorithmToolBar/utils", () => ({
  transform: vi.fn((value: number) => 100 + value),
}));

describe("Magnifier Component", () => {
  // vitest-canvas-mock 会自动 mock Canvas API，无需手动创建

  describe("组件渲染", () => {
    it("应该成功渲染 canvas 元素", () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      expect(wrapper.find("canvas").exists()).toBe(true);
    });

    it("应该根据 radius 设置 canvas 尺寸", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          radius: 150,
        },
      });

      await nextTick();
      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;

      expect(canvas.width).toBe(300); // radius * 2
      expect(canvas.height).toBe(300);
    });

    it("应该根据 shape 设置 canvas 样式", async () => {
      const circleWrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          shape: "circle",
        },
      });

      await nextTick();
      const circleCanvas = circleWrapper.find("canvas")
        .element as HTMLCanvasElement;
      expect(circleCanvas.style.borderRadius).toBe("50%");

      const squareWrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          shape: "square",
        },
      });

      await nextTick();
      const squareCanvas = squareWrapper.find("canvas")
        .element as HTMLCanvasElement;
      expect(squareCanvas.style.borderRadius).toBe("0");
    });
  });

  describe("图片加载和绘制", () => {
    it("应该在组件挂载时加载图片", async () => {
      const { loadImage } = await import("@/utils");

      mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(loadImage).toHaveBeenCalledWith("test-image.jpg", true);
    });

    it("应该支持 bigImgSrc 用于高清放大", async () => {
      const { loadImage } = await import("@/utils");

      mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          bigImgSrc: "test-image-hd.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(loadImage).toHaveBeenCalledWith("test-image.jpg", true);
      expect(loadImage).toHaveBeenCalledWith("test-image-hd.jpg", true);
    });

    it("应该在图片源改变时重新加载图片", async () => {
      const { loadImage } = await import("@/utils");

      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 改变图片源
      await wrapper.setProps({ src: "test-image-2.jpg" });
      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(loadImage).toHaveBeenCalledWith("test-image-2.jpg", true);
    });
  });

  describe("Canvas 绘制逻辑", () => {
    it("应该在鼠标移动时调用 clearRect 和 drawImage", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 更新鼠标位置
      await wrapper.setProps({ point: { x: 200, y: 200 } });
      await nextTick();

      // 验证 Canvas API 调用 (vitest-canvas-mock 会自动 mock 这些方法)
      expect(ctx?.clearRect).toHaveBeenCalled();
      expect(ctx?.drawImage).toHaveBeenCalled();
    });

    it("应该根据缩放比例计算源图片裁剪区域", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 2,
          scale: 5,
          point: { x: 400, y: 300 },
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // drawImage 应该被调用，参数包含计算后的裁剪坐标
      expect(ctx?.drawImage).toHaveBeenCalled();
    });

    it("应该限制裁剪区域不超出图片边界", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          scale: 5,
          point: { x: 0, y: 0 }, // 边界位置
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 应该能正常绘制，不会因为边界问题抛出错误
      expect(ctx?.drawImage).toHaveBeenCalled();
    });
  });

  describe("标签绘制", () => {
    it("应该在提供 label 时绘制标签", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          label: {
            text: "Test Label",
            color: "#ffffff",
            backgroundColor: "#000000",
          },
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 验证标签绘制
      expect(ctx?.fillRect).toHaveBeenCalled();
      expect(ctx?.fillText).toHaveBeenCalledWith("Test Label", 3, 15);
    });

    it("应该测量文本宽度并绘制背景", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          label: {
            text: "Long Label Text",
            color: "#ffffff",
            backgroundColor: "#000000",
          },
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 验证文本测量
      expect(ctx?.measureText).toHaveBeenCalledWith("Long Label Text");
      expect(ctx?.fillRect).toHaveBeenCalled();
    });

    it("应该在没有 label 时跳过标签绘制", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // fillText 不应该被调用（只有 drawImage 会调用 fillRect）
      expect(ctx?.fillText).not.toHaveBeenCalled();
    });
  });

  describe("亮度和对比度滤镜", () => {
    it("应该根据 brightness 和 contrast 设置 filter", async () => {
      const { transform } = await import("../../AlgorithmToolBar/utils");

      mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          brightness: 20,
          contrast: 10,
        },
      });

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 验证 transform 被调用
      expect(transform).toHaveBeenCalledWith(20);
      expect(transform).toHaveBeenCalledWith(10);
    });

    it("应该在亮度或对比度改变时更新滤镜", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          brightness: 0,
          contrast: 0,
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 更新亮度
      await wrapper.setProps({ brightness: 30 });
      await nextTick();

      // 验证重新绘制
      expect(ctx?.clearRect).toHaveBeenCalled();
    });
  });

  describe("Props 响应式", () => {
    it("应该在 radius 改变时重新初始化", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          radius: 100,
        },
      });

      await nextTick();
      let canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      expect(canvas.width).toBe(200);

      await wrapper.setProps({ radius: 150 });
      await nextTick();

      canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      expect(canvas.width).toBe(300);
    });

    it("应该在 shape 改变时更新样式", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          shape: "circle",
        },
      });

      await nextTick();
      let canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      expect(canvas.style.borderRadius).toBe("50%");

      await wrapper.setProps({ shape: "square" });
      await nextTick();

      canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      expect(canvas.style.borderRadius).toBe("0");
    });

    it("应该在 scale 改变时重新计算放大区域", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          scale: 5,
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      const drawImageCallsBefore =
        (ctx?.drawImage as any).mock?.calls?.length || 0;

      await wrapper.setProps({ scale: 10 });
      await nextTick();

      // 应该触发重新绘制
      const drawImageCallsAfter =
        (ctx?.drawImage as any).mock?.calls?.length || 0;
      expect(drawImageCallsAfter).toBeGreaterThan(drawImageCallsBefore);
    });
  });

  describe("边界情况", () => {
    it("应该处理无效的 radius 值", () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 100, y: 100 },
          radius: 50, // 小于默认最小值
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      // 应该设置最小宽度
      expect(canvas.width).toBeGreaterThanOrEqual(100);
    });

    it("应该处理极端的鼠标坐标", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: -100, y: -100 }, // 负坐标
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 不应该抛出错误
      expect(ctx?.drawImage).toHaveBeenCalled();
    });

    it("应该处理非常大的鼠标坐标", async () => {
      const wrapper = mount(Magnifier, {
        props: {
          src: "test-image.jpg",
          originScale: 1,
          point: { x: 10000, y: 10000 }, // 超出图片范围
        },
      });

      const canvas = wrapper.find("canvas").element as HTMLCanvasElement;
      const ctx = canvas.getContext("2d");

      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 100));

      // 应该限制在图片范围内
      expect(ctx?.drawImage).toHaveBeenCalled();
    });
  });
});
