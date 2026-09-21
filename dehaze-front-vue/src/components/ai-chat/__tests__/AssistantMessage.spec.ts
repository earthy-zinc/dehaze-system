import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import AssistantMessage from "../components/AssistantMessage.vue";
import type { ChatAssistantMessageVM } from "../types";
import { elStubs, findButton } from "./stubs";

// MarkdownRenderer 依赖 katex/mermaid，测试用轻量桩替代
vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

const assistant = (
  over: Partial<ChatAssistantMessageVM> = {}
): ChatAssistantMessageVM => ({
  role: "assistant",
  id: 1,
  status: "completed",
  text: "",
  thinking: null,
  steps: [],
  toolCalls: [],
  artifacts: [],
  memories: [],
  feedback: null,
  suggestions: [],
  ...over,
});

const mountAssistant = (over: Record<string, unknown> = {}) =>
  mount(AssistantMessage, {
    props: {
      message: assistant(),
      scope: "self",
      thinking: null,
      steps: [],
      artifacts: [],
      memories: [],
      feedback: null,
      suggestions: [],
      showSuggestions: false,
      ...over,
    },
    global: { stubs: elStubs },
  });

describe("AssistantMessage", () => {
  it("流式且无正文时显示「正在思考…」", () => {
    const wrapper = mountAssistant({
      message: assistant({ status: "streaming", text: "" }),
    });
    expect(wrapper.text()).toContain("正在思考…");
    expect(wrapper.find(".md-stub").exists()).toBe(false);
  });

  it("失败时展示 el-alert（透出 error 文案）", () => {
    const wrapper = mountAssistant({
      message: assistant({
        status: "failed",
        text: "部分输出",
        error: "上游超时",
      }),
    });
    const alert = wrapper.find(".el-alert-stub");
    expect(alert.exists()).toBe(true);
    expect(alert.text()).toContain("上游超时");
    // 失败态提供重试入口
    expect(findButton(wrapper, "重试")).toBeDefined();
  });

  it("终态消息挂载即请求产物，流式态不请求", () => {
    const done = mountAssistant({
      message: assistant({ id: 5, status: "completed" }),
    });
    expect(done.emitted("load-artifacts")).toEqual([[5]]);

    const zeroId = mountAssistant({
      message: assistant({ id: 0, status: "completed" }),
    });
    expect(zeroId.emitted("load-artifacts")).toBeUndefined();

    const streaming = mountAssistant({
      message: assistant({ id: 6, status: "streaming", text: "x" }),
    });
    expect(streaming.emitted("load-artifacts")).toBeUndefined();
  });

  it("流式转终态时补拉产物", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 7, status: "streaming", text: "x" }),
    });
    expect(wrapper.emitted("load-artifacts")).toBeUndefined();
    await wrapper.setProps({
      message: assistant({ id: 7, status: "completed", text: "x" }),
    });
    expect(wrapper.emitted("load-artifacts")).toEqual([[7]]);
  });

  it("操作栏复制上抛 copy", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 2, status: "completed", text: "内容" }),
    });
    await findButton(wrapper, "复制")!.trigger("click");
    expect(wrapper.emitted("copy")).toHaveLength(1);
  });

  it("推荐追问上抛 apply-suggestion", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 3, status: "completed", text: "答" }),
      suggestions: ["追问A"],
      showSuggestions: true,
    });
    const tag = wrapper.find(".el-tag-stub");
    await tag.trigger("click");
    expect(wrapper.emitted("apply-suggestion")).toEqual([["追问A"]]);
  });

  it("产物卡片点击上抛 open-artifact", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 4, status: "completed", text: "答" }),
      artifacts: [{ id: 11, type: "metric_report", invalid: false }],
    });
    await wrapper.find(".artifact-card").trigger("click");
    expect(wrapper.emitted("open-artifact")).toEqual([
      [{ id: 11, type: "metric_report", invalid: false }],
    ]);
  });

  it("子智能体用量：usage.subAgents 存在时渲染子智能体面板", () => {
    const wrapper = mountAssistant({
      message: assistant({
        id: 9,
        status: "completed",
        text: "答",
        usage: {
          inputTokens: 5,
          outputTokens: 3,
          cachedInputTokens: 0,
          credits: 2,
          subAgents: [
            {
              agentCode: "planner",
              inputTokens: 1,
              outputTokens: 1,
              cachedInputTokens: 0,
              credits: 1,
            },
          ],
        },
      }),
      steps: [{ position: 1, status: 1, tool: "task" }],
    });
    expect(wrapper.find(".sub-agent-panel").exists()).toBe(true);
    expect(wrapper.text()).toContain("planner");
  });

  it("子智能体用量：无 subAgents 时不渲染子智能体面板（不产生空壳）", () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 10, status: "completed", text: "答" }),
      steps: [{ position: 1, status: 1, tool: "task" }],
    });
    expect(wrapper.find(".sub-agent-panel").exists()).toBe(false);
  });
});
