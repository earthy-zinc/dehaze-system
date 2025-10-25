import { describe, it, expect } from "vitest";
import type {
  ViewCard,
  Breakpoints,
  WaterfallProps,
  ItemWidthProps,
  ItemWidthByBreakpointProps,
} from "../types";

describe("Waterfall Types", () => {
  describe("ViewCard", () => {
    it("应该正确定义ViewCard接口", () => {
      const card: ViewCard = {
        src: "image.jpg",
        id: "1",
        name: "Test Image",
        star: true,
        backgroundColor: "#ffffff",
      };

      expect(card.src).toBe("image.jpg");
      expect(card.id).toBe("1");
      expect(card.name).toBe("Test Image");
      expect(card.star).toBe(true);
      expect(card.backgroundColor).toBe("#ffffff");
    });

    it("应该允许ViewCard具有任意附加属性", () => {
      const card: ViewCard = {
        src: "image.jpg",
        customProp1: "value1",
        customProp2: 123,
        customProp3: true,
      };

      expect(card.customProp1).toBe("value1");
      expect(card.customProp2).toBe(123);
      expect(card.customProp3).toBe(true);
    });
  });

  describe("Breakpoints", () => {
    it("应该正确定义Breakpoints类型", () => {
      const breakpoints: Breakpoints = {
        1200: { rowPerView: 3 },
        800: { rowPerView: 2 },
        500: { rowPerView: 1 },
      };

      expect(breakpoints[1200]).toEqual({ rowPerView: 3 });
      expect(breakpoints[800]).toEqual({ rowPerView: 2 });
      expect(breakpoints[500]).toEqual({ rowPerView: 1 });
    });

    it("应该允许动态添加断点", () => {
      const breakpoints: Breakpoints = {
        1200: { rowPerView: 3 },
      };

      breakpoints[600] = { rowPerView: 2 };

      expect(breakpoints[600]).toEqual({ rowPerView: 2 });
    });
  });

  describe("WaterfallProps", () => {
    it("应该正确定义WaterfallProps接口", () => {
      const props: WaterfallProps = {
        breakpoints: {
          1200: { rowPerView: 3 },
        },
        width: 200,
        posDuration: 300,
        animationDuration: 1000,
        animationDelay: 300,
        animationEffect: "fadeIn",
        hasAroundGutter: true,
        gutter: 10,
        list: [],
        animationPrefix: "animate__animated",
        align: "center",
      };

      expect(props.width).toBe(200);
      expect(props.posDuration).toBe(300);
      expect(props.animationDuration).toBe(1000);
      expect(props.animationDelay).toBe(300);
      expect(props.animationEffect).toBe("fadeIn");
      expect(props.hasAroundGutter).toBe(true);
      expect(props.gutter).toBe(10);
      expect(props.list).toEqual([]);
      expect(props.animationPrefix).toBe("animate__animated");
      expect(props.align).toBe("center");
    });
  });

  describe("ItemWidthProps", () => {
    it("应该正确定义ItemWidthProps接口", () => {
      const itemWidthProps: ItemWidthProps = {
        breakpoints: {
          1200: { rowPerView: 3 },
        },
        wrapperWidth: 800,
        gutter: 10,
        hasAroundGutter: true,
        initWidth: 200,
      };

      expect(itemWidthProps.wrapperWidth).toBe(800);
      expect(itemWidthProps.gutter).toBe(10);
      expect(itemWidthProps.hasAroundGutter).toBe(true);
      expect(itemWidthProps.initWidth).toBe(200);
    });
  });

  describe("ItemWidthByBreakpointProps", () => {
    it("应该正确定义ItemWidthByBreakpointProps接口", () => {
      const itemWidthByBreakpointProps: ItemWidthByBreakpointProps = {
        breakpoints: {
          1200: { rowPerView: 3 },
        },
        wrapperWidth: 800,
        gutter: 10,
        hasAroundGutter: true,
        initWidth: 200,
        size: 1200,
      };

      expect(itemWidthByBreakpointProps.size).toBe(1200);
    });
  });
});
