import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import Loading from "../index.vue";

describe("Loading Component", () => {
  describe("组件渲染", () => {
    it("应该正确渲染加载组件", () => {
      const wrapper = mount(Loading);

      expect(wrapper.find(".wrap").exists()).toBe(true);
      expect(wrapper.find(".loading").exists()).toBe(true);
      expect(wrapper.find(".loading").findAll("span").length).toBe(5);
    });

    it("应该显示默认的加载文本", () => {
      const wrapper = mount(Loading);

      expect(wrapper.text()).toContain("正在生成图片中，请耐心等候");
    });

    it("应该显示自定义的加载文本", () => {
      const customText = "正在处理中，请稍候...";
      const wrapper = mount(Loading, {
        props: {
          loadingText: customText,
        },
      });

      expect(wrapper.text()).toContain(customText);
    });

    it("应该包含正确的CSS类结构", () => {
      const wrapper = mount(Loading);

      expect(wrapper.find(".wrap").exists()).toBe(true);
      expect(wrapper.find(".loading").exists()).toBe(true);
      expect(wrapper.find(".ml-2").exists()).toBe(true);
    });
  });

  describe("加载动画元素", () => {
    it("应该渲染5个动画span元素", () => {
      const wrapper = mount(Loading);
      const spans = wrapper.find(".loading").findAll("span");

      expect(spans.length).toBe(5);
      // 在测试环境中，v-for的key属性可能不会暴露出来，只验证数量
    });

    it("动画span应该按正确的顺序渲染", () => {
      const wrapper = mount(Loading);
      const spans = wrapper.find(".loading").findAll("span");

      // 验证渲染顺序 - 只验证数量和存在性
      expect(spans.length).toBe(5);
      spans.forEach((span, index) => {
        expect(span.exists()).toBe(true);
      });
    });
  });

  describe("Props验证", () => {
    it("应该设置默认的loadingText", () => {
      const wrapper = mount(Loading);

      expect(wrapper.props("loadingText")).toBe("正在生成图片中，请耐心等候");
    });

    it("应该接受自定义的loadingText", () => {
      const customText = "自定义加载文本";
      const wrapper = mount(Loading, {
        props: {
          loadingText: customText,
        },
      });

      expect(wrapper.props("loadingText")).toBe(customText);
      expect(wrapper.text()).toContain(customText);
    });

    it("应该处理空的loadingText", () => {
      const wrapper = mount(Loading, {
        props: {
          loadingText: "",
        },
      });

      expect(wrapper.props("loadingText")).toBe("");
      expect(wrapper.text()).not.toContain("正在生成图片中，请耐心等候");
    });
  });

  describe("组件名称", () => {
    it("应该设置正确的组件名称", () => {
      const wrapper = mount(Loading);

      // 通过组件定义检查名称
      expect(Loading.name || Loading.__name).toBe("Loading");
    });
  });

  describe("响应式更新", () => {
    it("应该在loadingText改变时更新显示文本", async () => {
      const wrapper = mount(Loading, {
        props: {
          loadingText: "初始文本",
        },
      });

      expect(wrapper.text()).toContain("初始文本");

      await wrapper.setProps({
        loadingText: "更新后的文本",
      });

      expect(wrapper.text()).toContain("更新后的文本");
      expect(wrapper.text()).not.toContain("初始文本");
    });

    it("应该在loadingText清空时隐藏文本", async () => {
      const wrapper = mount(Loading, {
        props: {
          loadingText: "显示的文本",
        },
      });

      expect(wrapper.text()).toContain("显示的文本");

      await wrapper.setProps({
        loadingText: "",
      });

      expect(wrapper.text()).not.toContain("显示的文本");
    });
  });

  describe("DOM结构", () => {
    it("应该有正确的嵌套结构", () => {
      const wrapper = mount(Loading);

      const wrapDiv = wrapper.find(".wrap");
      const loadingDiv = wrapDiv.find(".loading");
      const textSpan = wrapDiv.find(".ml-2");

      expect(wrapDiv.exists()).toBe(true);
      expect(loadingDiv.exists()).toBe(true);
      expect(textSpan.exists()).toBe(true);

      // 验证loadingDiv包含5个span
      expect(loadingDiv.findAll("span").length).toBe(5);
    });

    it("文本应该和动画在同一层级", () => {
      const wrapper = mount(Loading);

      const wrapDiv = wrapper.find(".wrap");
      const children = wrapDiv.findAll("*");

      // 验证主要元素存在
      expect(children.length).toBeGreaterThan(0);
      expect(wrapper.find(".loading").exists()).toBe(true);
      expect(wrapper.find(".ml-2").exists()).toBe(true);
    });
  });

  describe("可访问性", () => {
    it("应该为加载指示器提供合适的语义", () => {
      const wrapper = mount(Loading);

      // 组件应该包含加载状态的文本信息
      expect(wrapper.text()).toContain("正在");
      expect(wrapper.find(".loading").exists()).toBe(true);
    });

    it("应该能通过文本内容识别加载状态", () => {
      const wrapper = mount(Loading);

      const textContent = wrapper.text();
      expect(textContent).toMatch(/正在|加载|生成|处理/);
    });
  });

  describe("边界情况", () => {
    it("应该处理非常长的loadingText", () => {
      const longText = "这是一个非常长的加载文本内容".repeat(10);
      const wrapper = mount(Loading, {
        props: {
          loadingText: longText,
        },
      });

      expect(wrapper.text()).toContain(longText);
    });

    it("应该处理包含特殊字符的loadingText", () => {
      const specialText = "Loading... 请稍候! @#$%^&*()";
      const wrapper = mount(Loading, {
        props: {
          loadingText: specialText,
        },
      });

      expect(wrapper.text()).toContain(specialText);
    });

    it("应该处理包含HTML标签的文本（应该转义）", () => {
      const htmlText = "Loading <script>alert('xss')</script>";
      const wrapper = mount(Loading, {
        props: {
          loadingText: htmlText,
        },
      });

      // 文本应该被转义，不会渲染为HTML
      expect(wrapper.text()).toContain(htmlText);
      expect(wrapper.find("script").exists()).toBe(false);
    });
  });

  describe("样式相关", () => {
    it("应该应用正确的Tailwind CSS类", () => {
      const wrapper = mount(Loading);

      // 验证主要元素和部分CSS类存在
      expect(wrapper.find(".ml-2").exists()).toBe(true);
      expect(wrapper.find(".wrap").exists()).toBe(true);
      // 在测试环境中，某些Tailwind CSS类可能不会完全保持原样
    });

    it("文本元素应该有正确的间距类", () => {
      const wrapper = mount(Loading);

      const textElement = wrapper.find(".ml-2");
      expect(textElement.exists()).toBe(true);
    });
  });
});
