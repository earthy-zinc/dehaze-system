import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import FeedbackBar from "../components/FeedbackBar.vue";
import type { ChatFeedbackVM } from "../types";
import { elStubs, findButton } from "./stubs";

const mountBar = (feedback: ChatFeedbackVM | null = null) =>
  mount(FeedbackBar, {
    props: { messageId: 9, feedback },
    global: { stubs: elStubs },
  });

describe("FeedbackBar", () => {
  it("点赞一键提交 rating=1", async () => {
    const wrapper = mountBar();
    await findButton(wrapper, "有帮助").trigger("click");
    expect(wrapper.emitted("submit")).toEqual([[{ rating: 1 }]]);
  });

  it("点踩展开标签并提交 rating=-1（含标签与说明）", async () => {
    const wrapper = mountBar();
    await findButton(wrapper, "待改进").trigger("click");

    // 展开后出现问题标签
    const tagButton = wrapper
      .findAll(".el-radio-button-stub")
      .find((b) => b.text().includes("内容错误"));
    await tagButton!.trigger("click");

    const input = wrapper.find(".el-input-stub");
    await input.setValue("请更详细");

    await findButton(wrapper, "提交").trigger("click");

    expect(wrapper.emitted("submit")?.[0]).toEqual([
      { rating: -1, tags: ["incorrect"], comment: "请更详细" },
    ]);
  });

  it("已有反馈展示撤销入口，无反馈展示收起入口", () => {
    expect(findButton(mountBar({ rating: 1 }), "撤销反馈")).toBeDefined();
    expect(findButton(mountBar(null), "收起")).toBeDefined();
  });

  it("取消上抛 cancel", async () => {
    const wrapper = mountBar(null);
    await findButton(wrapper, "收起").trigger("click");
    expect(wrapper.emitted("cancel")).toHaveLength(1);
  });

  it("连续重复点击产生多次 submit（无内部去重）", async () => {
    const wrapper = mountBar();
    const like = findButton(wrapper, "有帮助");
    await like.trigger("click");
    await like.trigger("click");
    expect(wrapper.emitted("submit")).toHaveLength(2);
  });
});
