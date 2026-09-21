import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import BranchSwitcher from "../components/BranchSwitcher.vue";
import { elStubs } from "./stubs";

const mountSwitcher = (total: number, current: number) =>
  mount(BranchSwitcher, {
    props: { total, current },
    global: { stubs: elStubs },
  });

describe("BranchSwitcher", () => {
  it("total<2 时不渲染", () => {
    expect(mountSwitcher(1, 1).find(".branch-switcher").exists()).toBe(false);
    expect(mountSwitcher(0, 0).find(".branch-switcher").exists()).toBe(false);
  });

  it("渲染 current/total 并上抛 change", async () => {
    const wrapper = mountSwitcher(3, 2);
    expect(wrapper.find(".branch-switcher__index").text()).toBe("2/3");

    await wrapper.find(".branch-switcher__prev").trigger("click");
    await wrapper.find(".branch-switcher__next").trigger("click");

    const changes = wrapper.emitted("change") ?? [];
    expect(changes).toEqual([[1], [3]]);
  });

  it("边界禁用：首页不可前移、末页不可后移", () => {
    const first = mountSwitcher(3, 1);
    expect(
      first.find(".branch-switcher__prev").attributes("disabled")
    ).toBeDefined();

    const last = mountSwitcher(3, 3);
    expect(
      last.find(".branch-switcher__next").attributes("disabled")
    ).toBeDefined();
    expect(
      last.find(".branch-switcher__prev").attributes("disabled")
    ).toBeUndefined();
  });
});
