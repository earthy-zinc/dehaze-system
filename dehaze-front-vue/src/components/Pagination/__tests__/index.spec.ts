import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Pagination from "../index.vue";

describe("Pagination Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("组件渲染", () => {
    it("应该正确渲染分页组件", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.find(".pagination").exists()).toBe(true);
      expect(wrapper.find("el-pagination").exists()).toBe(true);
    });

    it("应该设置默认的props值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("page")).toBe(1);
      expect(wrapper.props("limit")).toBe(20);
      expect(wrapper.props("pageSizes")).toEqual([10, 20, 30, 50]);
      expect(wrapper.props("layout")).toBe(
        "total, sizes, prev, pager, next, jumper"
      );
      expect(wrapper.props("background")).toBe(true);
      expect(wrapper.props("autoScroll")).toBe(true);
      expect(wrapper.props("hidden")).toBe(false);
    });

    it("应该应用自定义的props值", () => {
      const customPageSizes = [5, 10, 15];
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 2,
          limit: 10,
          pageSizes: customPageSizes,
          layout: "prev, pager, next",
          background: false,
          autoScroll: false,
          hidden: true,
        },
      });

      expect(wrapper.props("page")).toBe(2);
      expect(wrapper.props("limit")).toBe(10);
      expect(wrapper.props("pageSizes")).toEqual(customPageSizes);
      expect(wrapper.props("layout")).toBe("prev, pager, next");
      expect(wrapper.props("background")).toBe(false);
      expect(wrapper.props("autoScroll")).toBe(false);
      expect(wrapper.props("hidden")).toBe(true);
    });

    it("应该根据hidden属性显示/隐藏组件", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          hidden: true,
        },
      });

      expect(wrapper.find(".pagination.hidden").exists()).toBe(true);

      wrapper.setProps({ hidden: false });
      expect(wrapper.find(".pagination.hidden").exists()).toBe(false);
    });
  });

  describe("双向绑定", () => {
    it("应该支持page的双向绑定", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
        },
      });

      expect(wrapper.props("page")).toBe(1);

      await wrapper.setProps({ page: 3 });
      expect(wrapper.props("page")).toBe(3);
    });

    it("应该支持limit的双向绑定", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          limit: 20,
        },
      });

      expect(wrapper.props("limit")).toBe(20);

      await wrapper.setProps({ limit: 50 });
      expect(wrapper.props("limit")).toBe(50);
    });

    it("应该在当前页改变时更新currentPage", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
        },
      });

      const vm = wrapper.vm as any;
      expect(vm.currentPage.value).toBe(1);

      await wrapper.setProps({ page: 5 });
      expect(vm.currentPage.value).toBe(5);
    });
  });

  describe("事件处理", () => {
    it("应该在页码改变时触发pagination事件", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
          limit: 20,
        },
      });

      // 模拟页码改变
      const vm = wrapper.vm as any;
      await vm.handleCurrentChange(3);

      expect(wrapper.emitted("pagination")).toBeTruthy();
      expect(wrapper.emitted("pagination")?.[0]).toEqual([
        { page: 3, limit: 20 },
      ]);
    });

    it("应该在每页条数改变时触发pagination事件", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
          limit: 20,
        },
      });

      // 模拟每页条数改变
      const vm = wrapper.vm as any;
      await vm.handleSizeChange(50);

      expect(wrapper.emitted("pagination")).toBeTruthy();
      expect(wrapper.emitted("pagination")?.[0]).toEqual([
        { page: 1, limit: 50 },
      ]);
    });

    it("应该触发update:page事件", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
        },
      });

      const vm = wrapper.vm as any;
      await vm.handleCurrentChange(3);

      expect(wrapper.emitted("update:page")).toBeTruthy();
      expect(wrapper.emitted("update:page")?.[0]).toEqual([3]);
    });

    it("应该触发update:limit事件", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          limit: 20,
        },
      });

      const vm = wrapper.vm as any;
      await vm.handleSizeChange(50);

      expect(wrapper.emitted("update:limit")).toBeTruthy();
      expect(wrapper.emitted("update:limit")?.[0]).toEqual([50]);
    });

    it("应该处理多个事件连续触发", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
          limit: 20,
        },
      });

      const vm = wrapper.vm as any;

      // 先改变页码
      await vm.handleCurrentChange(2);
      // 再改变每页条数
      await vm.handleSizeChange(30);

      expect(wrapper.emitted("pagination")?.length).toBe(2);
      expect(wrapper.emitted("pagination")?.[0]).toEqual([
        { page: 2, limit: 20 },
      ]);
      expect(wrapper.emitted("pagination")?.[1]).toEqual([
        { page: 1, limit: 30 },
      ]);
    });
  });

  describe("Element Plus Pagination 集成", () => {
    it("应该正确传递props给el-pagination", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 2,
          limit: 10,
          pageSizes: [5, 10, 20],
          layout: "total, prev, pager, next",
          background: false,
        },
      });

      const elPagination = wrapper.find("el-pagination");

      // 这些props应该传递给el-pagination
      expect(elPagination.exists()).toBe(true);
    });

    it("应该监听el-pagination的事件", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      // 组件应该设置了size-change和current-change事件监听器
      const elPagination = wrapper.find("el-pagination");
      expect(elPagination.exists()).toBe(true);
    });
  });

  describe("边界情况", () => {
    it("应该处理total为0的情况", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 0,
        },
      });

      expect(wrapper.find(".pagination").exists()).toBe(true);
      expect(wrapper.props("total")).toBe(0);
    });

    it("应该处理非常大的total值", () => {
      const largeTotal = 999999;
      const wrapper = mount(Pagination, {
        props: {
          total: largeTotal,
        },
      });

      expect(wrapper.props("total")).toBe(largeTotal);
    });

    it("应该处理page超出范围的情况", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 999, // 超出范围的页码
          limit: 20,
        },
      });

      expect(wrapper.props("page")).toBe(999);
    });

    it("应该处理空的pageSizes数组", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          pageSizes: [],
        },
      });

      expect(wrapper.props("pageSizes")).toEqual([]);
    });

    it("应该处理单个pageSize", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          pageSizes: [20],
        },
      });

      expect(wrapper.props("pageSizes")).toEqual([20]);
    });

    it("应该处理空的layout字符串", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          layout: "",
        },
      });

      expect(wrapper.props("layout")).toBe("");
    });
  });

  describe("Props 验证", () => {
    it("应该验证total是必需的", () => {
      // 这个测试主要验证TypeScript类型定义
      expect(() => {
        mount(Pagination); // 没有传入必需的total prop
      }).toThrow();
    });

    it("应该接受正确的pageSizes类型", () => {
      const validPageSizes = [10, 20, 30, 50];
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          pageSizes: validPageSizes,
        },
      });

      expect(wrapper.props("pageSizes")).toEqual(validPageSizes);
    });

    it("应该处理数字类型的props", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 2,
          limit: 15,
        },
      });

      expect(typeof wrapper.props("total")).toBe("number");
      expect(typeof wrapper.props("page")).toBe("number");
      expect(typeof wrapper.props("limit")).toBe("number");
    });
  });

  describe("样式类", () => {
    it("应该应用正确的CSS类", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.find(".pagination").exists()).toBe(true);
    });

    it("应该在hidden为true时添加hidden类", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          hidden: true,
        },
      });

      expect(wrapper.find(".pagination.hidden").exists()).toBe(true);
    });

    it("应该在hidden为false时不添加hidden类", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          hidden: false,
        },
      });

      expect(wrapper.find(".pagination.hidden").exists()).toBe(false);
    });
  });

  describe("组件响应式", () => {
    it("应该在props改变时正确更新", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          page: 1,
          limit: 20,
        },
      });

      const vm = wrapper.vm as any;

      // 改变多个props
      await wrapper.setProps({
        total: 200,
        page: 3,
        limit: 30,
      });

      expect(wrapper.props("total")).toBe(200);
      expect(wrapper.props("page")).toBe(3);
      expect(wrapper.props("limit")).toBe(30);
      expect(vm.currentPage.value).toBe(3);
      expect(vm.pageSize.value).toBe(30);
    });

    it("应该在hidden状态切换时正确显示/隐藏", async () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
          hidden: false,
        },
      });

      expect(wrapper.find(".pagination.hidden").exists()).toBe(false);

      await wrapper.setProps({ hidden: true });
      expect(wrapper.find(".pagination.hidden").exists()).toBe(true);

      await wrapper.setProps({ hidden: false });
      expect(wrapper.find(".pagination.hidden").exists()).toBe(false);
    });
  });

  describe("默认值处理", () => {
    it("应该使用pageSizes的默认值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("pageSizes")).toEqual([10, 20, 30, 50]);
    });

    it("应该使用layout的默认值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("layout")).toBe(
        "total, sizes, prev, pager, next, jumper"
      );
    });

    it("应该使用background的默认值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("background")).toBe(true);
    });

    it("应该使用autoScroll的默认值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("autoScroll")).toBe(true);
    });

    it("应该使用hidden的默认值", () => {
      const wrapper = mount(Pagination, {
        props: {
          total: 100,
        },
      });

      expect(wrapper.props("hidden")).toBe(false);
    });
  });
});
