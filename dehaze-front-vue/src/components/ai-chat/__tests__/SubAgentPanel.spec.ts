// 子智能体面板单测：有子智能体粒度用量时渲染，无数据（缺省/空）不渲染空壳。
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SubAgentPanel from "../components/SubAgentPanel.vue";
import type { ChatSubAgentUsageVM, ChatThoughtStepVM } from "../types";

const step = (over: Partial<ChatThoughtStepVM> = {}): ChatThoughtStepVM => ({
  position: 1,
  status: 1,
  tool: "task",
  ...over,
});

const agent = (
  over: Partial<ChatSubAgentUsageVM> = {}
): ChatSubAgentUsageVM => ({
  agentCode: "researcher",
  inputTokens: 3,
  outputTokens: 4,
  cachedInputTokens: 1,
  credits: 2,
  ...over,
});

describe("SubAgentPanel", () => {
  it("有子智能体用量时渲染各子智能体 Token/积分", () => {
    const wrapper = mount(SubAgentPanel, {
      props: {
        steps: [step()],
        subAgents: [agent(), agent({ agentCode: "coder", credits: 9 })],
      },
    });
    expect(wrapper.find(".sub-agent-panel").exists()).toBe(true);
    expect(wrapper.text()).toContain("researcher");
    expect(wrapper.text()).toContain("coder");
    expect(wrapper.text()).toContain("输入 3 / 输出 4");
    expect(wrapper.text()).toContain("积分 2");
  });

  it("无子智能体用量（缺省或空数组）不渲染空壳", () => {
    const missing = mount(SubAgentPanel, { props: { steps: [step()] } });
    expect(missing.find(".sub-agent-panel").exists()).toBe(false);

    const empty = mount(SubAgentPanel, {
      props: { steps: [step()], subAgents: [] },
    });
    expect(empty.find(".sub-agent-panel").exists()).toBe(false);
  });
});
