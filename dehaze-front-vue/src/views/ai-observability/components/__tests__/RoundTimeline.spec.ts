import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AiObservabilityTimelineRound } from "dehaze-sdk-js";

import RoundTimeline from "../RoundTimeline.vue";

// Element Plus 组件统一轻量桩：透传全部具名插槽与默认插槽，便于断言内容渲染
const passthrough = {
  template:
    '<div class="stub"><slot /><slot name="dot" /><slot name="title" /></div>',
};
const elStubs = {
  "el-timeline": passthrough,
  "el-timeline-item": passthrough,
  "el-tag": { template: '<span class="tag"><slot /></span>' },
  "el-collapse": passthrough,
  "el-collapse-item": {
    template: '<div class="collapse-item"><slot name="title" /><slot /></div>',
  },
  "el-empty": {
    template: '<div class="empty">{{ description }}</div>',
    props: ["description"],
  },
};

// LlmCallNode 内嵌 JsonViewer，raw 为 null 时应透出降级提示
vi.mock("@/components/JsonViewer.vue", () => ({
  default: {
    name: "JsonViewer",
    template: '<div class="json-viewer-stub">{{ emptyText }}</div>',
    props: ["data", "filename", "emptyText"],
  },
}));

const baseRound: AiObservabilityTimelineRound = {
  userMessage: {
    id: 1,
    role: "user",
    content: "帮我总结这段文档",
    createTime: "2026-09-13 10:00:00",
  },
  assistantMessage: {
    id: 2,
    role: "assistant",
    content: "好的，总结如下",
    createTime: "2026-09-13 10:00:05",
    inputTokens: 100,
    outputTokens: 20,
  },
  traces: [
    {
      traceId: "trace-1",
      traceType: "conversation",
      status: 1,
      durationMs: 4200,
      events: [
        {
          kind: "input",
          ts: "2026-09-13 10:00:00",
          message: { id: 1, role: "user", content: "帮我总结这段文档" },
        },
        {
          kind: "llm_call",
          ts: "2026-09-13 10:00:01",
          seq: 1,
          model: "qwen3-0.6b",
          status: 1,
          durationMs: 666,
          firstTokenMs: 431,
          promptTokens: 478,
          completionTokens: 24,
          rawRequest: null,
          rawResponse: null,
          summary: { inputSnapshot: null, outputSnapshot: null },
        },
        {
          kind: "tool_exec",
          ts: "2026-09-13 10:00:02",
          position: 1,
          tool: "enhance_image",
          toolInput: { imageId: 9 },
          observation: "处理成功",
          latencyMs: 800,
          isSubagent: 0,
        },
        {
          kind: "billing",
          ts: "2026-09-13 10:00:04",
          billType: "conversation",
          credits: 1,
          tokens: { input: 478, output: 24, cached: 0 },
        },
      ],
    },
    {
      traceId: "trace-bypass",
      traceType: "summary",
      status: 1,
      durationMs: 100,
      events: [
        {
          kind: "llm_call",
          ts: "2026-09-13 10:00:06",
          seq: 1,
          model: "qwen3-0.6b",
          status: 1,
          durationMs: 100,
          rawRequest: { model: "qwen3-0.6b" },
          rawResponse: { id: "chatcmpl-x" },
        },
      ],
    },
  ],
};

const mountTimeline = (round: AiObservabilityTimelineRound) =>
  mount(RoundTimeline, {
    props: { round },
    global: { stubs: elStubs },
  });

describe("RoundTimeline", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("按 kind 渲染各事件节点内容", () => {
    const wrapper = mountTimeline(baseRound);

    const text = wrapper.text();
    // 事件类型标签
    expect(text).toContain("用户输入");
    expect(text).toContain("LLM 调用");
    expect(text).toContain("工具执行");
    expect(text).toContain("计费");
    // 用户消息全文
    expect(text).toContain("帮我总结这段文档");
    // 工具入参/返回/耗时
    expect(text).toContain("enhance_image");
    expect(text).toContain("处理成功");
    // 计费明细
    expect(text).toContain("conversation");
    // 助手输出收尾
    expect(text).toContain("助手输出");
    expect(text).toContain("好的，总结如下");
  });

  it("llm_call 节点渲染摘要卡，raw 为 null 时 JsonViewer 显示空态", () => {
    const wrapper = mountTimeline(baseRound);
    const text = wrapper.text();

    expect(text).toContain("#1");
    expect(text).toContain("qwen3-0.6b");
    // 主对话 llm_call 节点挂出请求/响应两个 JsonViewer（raw 为 null，空态文案由 JsonViewer 默认值提供）
    const stubs = wrapper.findAll(".json-viewer-stub");
    expect(stubs.length).toBeGreaterThanOrEqual(2);
    // 工具入参 JsonViewer 透出 emptyText prop
    expect(text).toContain("无入参数据");
  });

  it("旁路 trace 挂尾部并带类型徽标", () => {
    const wrapper = mountTimeline(baseRound);

    expect(wrapper.text()).toContain("旁路调用");
    expect(wrapper.text()).toContain("摘要压缩");
    expect(wrapper.text()).toContain("trace-bypass");
  });

  it("主事件为空时显示空态", () => {
    const wrapper = mountTimeline({
      userMessage: { id: 1, role: "user", content: "hi" },
      traces: [
        {
          traceId: "t",
          traceType: "conversation",
          status: 1,
          durationMs: 0,
          events: [],
        },
      ],
    });

    expect(wrapper.find(".empty").exists()).toBe(true);
    expect(wrapper.text()).toContain("该轮次无事件数据");
  });

  it("events 按 ts 交织排序：input 先于 llm_call 先于 tool_exec", () => {
    const wrapper = mountTimeline(baseRound);

    const html = wrapper.html();
    expect(html.indexOf("帮我总结这段文档")).toBeLessThan(
      html.indexOf("enhance_image")
    );
    expect(html.indexOf("enhance_image")).toBeLessThan(
      html.indexOf("助手输出")
    );
  });

  it("ts 为 null 的事件沉底展示", () => {
    const round: AiObservabilityTimelineRound = {
      userMessage: { id: 1, role: "user", content: "hi" },
      traces: [
        {
          traceId: "t",
          traceType: "conversation",
          status: 1,
          durationMs: 0,
          events: [
            { kind: "billing", ts: undefined, billType: "conversation" },
            {
              kind: "input",
              ts: "2026-09-13 10:00:00",
              message: { id: 1, role: "user", content: "有时刻的用户输入" },
            },
            {
              kind: "input",
              ts: null,
              message: { id: 2, role: "user", content: "无时刻的用户输入" },
            },
          ],
        },
      ],
    };
    const wrapper = mountTimeline(round);

    const html = wrapper.html();
    expect(html.indexOf("有时刻的用户输入")).toBeLessThan(
      html.indexOf("无时刻的用户输入")
    );
  });
});
