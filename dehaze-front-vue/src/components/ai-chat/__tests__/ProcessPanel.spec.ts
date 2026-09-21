import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ProcessPanel from "../components/ProcessPanel.vue";
import type { ChatTraceStepVM, ChatTraceSummaryVM } from "../types";
import { elStubs } from "./stubs";

const steps: ChatTraceStepVM[] = [
  { id: "s1", kind: "input", label: "用户输入", status: "ok" },
  {
    id: "s2",
    kind: "tool_exec",
    label: "enhance_image",
    status: "failed",
    error: "超时",
    latencyMs: 800,
  },
  {
    id: "s3",
    kind: "llm_call",
    label: "模型回答",
    status: "ok",
    latencyMs: 1500,
  },
];

const summary: ChatTraceSummaryVM = {
  stepCount: 3,
  durationMs: 4200,
  credits: 1,
  failedCount: 1,
};

const mountPanel = (over: Record<string, unknown> = {}) =>
  mount(ProcessPanel, {
    props: { steps, summary, ...over },
    global: { stubs: elStubs },
  });

describe("ProcessPanel", () => {
  it("默认折叠：不渲染过程正文", () => {
    const wrapper = mountPanel();
    expect(wrapper.find(".process-panel__body").exists()).toBe(false);
    expect(wrapper.find(".step-timeline").exists()).toBe(false);
  });

  it("点击头部展开后渲染步骤时间线与摘要", async () => {
    const wrapper = mountPanel();
    await wrapper.find(".process-panel__header").trigger("click");

    expect(wrapper.find(".process-panel__body").exists()).toBe(true);
    expect(wrapper.find(".step-timeline").exists()).toBe(true);
    expect(wrapper.text()).toContain("LLM 调用");
    expect(wrapper.text()).toContain("共 3 步");
  });

  it("失败步骤显著标注（is-failed + 错误原因）", async () => {
    const wrapper = mountPanel({ defaultOpen: true });
    const failed = wrapper.find(".step-timeline__item.is-failed");
    expect(failed.exists()).toBe(true);
    expect(failed.text()).toContain("超时");
    expect(wrapper.text()).toContain("1 步失败");
  });

  it("上下文构成标签透传渲染", async () => {
    const wrapper = mountPanel({
      defaultOpen: true,
      chips: [
        { kind: "memory", label: "记忆", ratio: 30 },
        { kind: "system", label: "系统提示", ratio: 20 },
      ],
    });
    expect(wrapper.findAll(".context-chip")).toHaveLength(2);
  });

  it("无步骤时展示空态", async () => {
    const wrapper = mountPanel({
      defaultOpen: true,
      steps: [],
      summary: { stepCount: 0, failedCount: 0 },
    });
    expect(wrapper.find(".el-empty-stub").exists()).toBe(true);
  });
});
