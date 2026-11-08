import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import type { Artifact, Block, Message, ToolCall, Usage } from "clarity-sdk-js";
import AssistantMessage from "../components/AssistantMessage.vue";
import { elStubs, findButton } from "./stubs";

// MarkdownRenderer 依赖 katex/mermaid，测试用轻量桩替代
vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content", "streaming"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

const EMPTY_USAGE: Usage = {
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  credits: 0,
};

function textBlock(id: string, text: string): Block {
  return { id, kind: "text", seq: 1, text };
}

const assistant = (over: Partial<Message> = {}): Message => ({
  id: 1,
  conversationId: 1,
  parentMessageId: null,
  role: "assistant",
  status: "completed",
  model: null,
  error: null,
  blocks: [],
  plan: null,
  pendingInterrupt: null,
  usage: { ...EMPTY_USAGE },
  artifactIds: [],
  memoryIds: [],
  suggestions: [],
  edited: false,
  createTime: "2026-01-01T00:00:00Z",
  updateTime: null,
  ...over,
});

const mountAssistant = (over: Record<string, unknown> = {}) =>
  mount(AssistantMessage, {
    props: {
      message: assistant(),
      scope: "self",
      showSuggestions: false,
      ...over,
    },
    global: { stubs: elStubs },
  });

describe("AssistantMessage", () => {
  it("running 且无正文时显示「正在思考…」", () => {
    const wrapper = mountAssistant({
      message: assistant({ status: "running" }),
    });
    expect(wrapper.text()).toContain("正在思考…");
    expect(wrapper.find(".md-stub").exists()).toBe(false);
  });

  it("正文由 text 块派生并按 seq 拼接", () => {
    const wrapper = mountAssistant({
      message: assistant({
        blocks: [
          { id: "b2", kind: "text", seq: 2, text: "后半" },
          { id: "b1", kind: "text", seq: 1, text: "前半" },
        ],
      }),
    });
    expect(wrapper.find(".md-stub").text()).toBe("前半\n\n后半");
  });

  it("失败时展示 el-alert（透出 error 文案）并提供重试", () => {
    const wrapper = mountAssistant({
      message: assistant({
        status: "failed",
        blocks: [textBlock("b1", "部分输出")],
        error: "上游超时",
      }),
    });
    const alert = wrapper.find(".el-alert-stub");
    expect(alert.exists()).toBe(true);
    expect(alert.text()).toContain("上游超时");
    expect(findButton(wrapper, "重试")).toBeDefined();
  });

  it("终态消息挂载即请求产物，running 态不请求", () => {
    const done = mountAssistant({
      message: assistant({ id: 5, status: "completed" }),
    });
    expect(done.emitted("load-artifacts")).toEqual([[5]]);

    const zeroId = mountAssistant({
      message: assistant({ id: 0, status: "completed" }),
    });
    expect(zeroId.emitted("load-artifacts")).toBeUndefined();

    const running = mountAssistant({
      message: assistant({
        id: 6,
        status: "running",
        blocks: [textBlock("b", "x")],
      }),
    });
    expect(running.emitted("load-artifacts")).toBeUndefined();
  });

  it("running 转终态时补拉产物", async () => {
    const wrapper = mountAssistant({
      message: assistant({
        id: 7,
        status: "running",
        blocks: [textBlock("b", "x")],
      }),
    });
    expect(wrapper.emitted("load-artifacts")).toBeUndefined();
    await wrapper.setProps({
      message: assistant({
        id: 7,
        status: "completed",
        blocks: [textBlock("b", "x")],
      }),
    });
    expect(wrapper.emitted("load-artifacts")).toEqual([[7]]);
  });

  it("操作栏复制上抛 message-action(copy)", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 2, blocks: [textBlock("b1", "内容")] }),
    });
    await findButton(wrapper, "复制")!.trigger("click");
    expect(wrapper.emitted("action")).toEqual([
      [{ type: "copy", messageId: 2 }],
    ]);
  });

  it("操作栏默认只露复制，其余操作收进「···」菜单", () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 4, blocks: [textBlock("b1", "内容")] }),
    });

    const bar = wrapper.find(".message-action-bar");
    expect(bar.findAll(":scope > button").map((b) => b.text())).toEqual([
      "复制",
    ]);
    expect(findButton(wrapper, "···")).toBeTruthy();
    const menuText = bar.find(".el-dropdown-menu-stub").text();
    expect(menuText).toContain("重新生成");
    expect(menuText).toContain("删除");
  });

  it("非链尾回复不提供删除入口", () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 4, blocks: [textBlock("b1", "内容")] }),
      canDelete: false,
    });
    const menuText = wrapper.find(".el-dropdown-menu-stub").text();
    expect(menuText).not.toContain("删除");
    expect(menuText).toContain("重新生成");
  });

  it("中断等待中（awaiting_input）不提供重新生成/删除入口", () => {
    const wrapper = mountAssistant({
      message: assistant({
        id: 6,
        status: "awaiting_input",
        blocks: [textBlock("b1", "部分输出")],
        pendingInterrupt: {
          id: "i1",
          type: "confirm",
          confirmKind: "tool_permission",
          createdAt: "2026-01-01T00:00:00Z",
          expiresAt: "2099-01-01T00:00:00Z",
          actionable: true,
        },
      }),
    });

    // 复制不受影响；中断挂起时后端拒绝重新生成/删除，整组菜单不出现
    expect(findButton(wrapper, "复制")).toBeTruthy();
    expect(findButton(wrapper, "···")).toBeFalsy();
    expect(wrapper.text()).not.toContain("重新生成");
    expect(wrapper.text()).not.toContain("删除");
  });

  it("「···」菜单项上抛对应 message-action", async () => {
    const wrapper = mountAssistant({
      message: assistant({ id: 5, blocks: [textBlock("b1", "内容")] }),
    });

    await findButton(wrapper, "重新生成")!.trigger("click");
    expect(wrapper.emitted("action")).toEqual([
      [{ type: "regenerate", messageId: 5 }],
    ]);
  });

  it("推荐追问上抛 apply-suggestion", async () => {
    const wrapper = mountAssistant({
      message: assistant({
        id: 3,
        blocks: [textBlock("b1", "答")],
        suggestions: ["追问A"],
      }),
      showSuggestions: true,
    });
    const tag = wrapper.find(".el-tag-stub");
    await tag.trigger("click");
    expect(wrapper.emitted("action")).toEqual([
      [{ type: "apply-suggestion", question: "追问A" }],
    ]);
  });

  it("产物由宿主按 id 注入，点击上抛 open-artifact", async () => {
    const artifact: Artifact = {
      id: 11,
      conversationId: 1,
      messageId: 4,
      type: "metric_report",
      isInvalid: false,
    };
    const wrapper = mountAssistant({
      message: assistant({ id: 4, blocks: [textBlock("b1", "答")] }),
      artifacts: [artifact],
    });
    await wrapper.find(".artifact-card").trigger("click");
    expect(wrapper.emitted("open-artifact")).toEqual([[artifact]]);
  });

  it("工具步骤由 blocks 派生：tool_use 块渲染推理过程，状态字符串驱动徽标", () => {
    const tool: ToolCall = {
      callId: "c1",
      name: "task",
      status: "success",
      subAgent: false,
    };
    const wrapper = mountAssistant({
      message: assistant({
        id: 9,
        blocks: [{ id: "t1", kind: "tool_use", seq: 1, tool }],
      }),
    });
    expect(wrapper.find(".thought-chain").exists()).toBe(true);
    expect(wrapper.text()).toContain("task");
    expect(wrapper.text()).toContain("成功");
  });

  it("子智能体用量：usage.subAgents 存在时渲染子智能体面板", () => {
    const tool: ToolCall = {
      callId: "c1",
      name: "task",
      status: "success",
      subAgent: true,
    };
    const wrapper = mountAssistant({
      message: assistant({
        id: 9,
        blocks: [{ id: "t1", kind: "tool_use", seq: 1, tool }],
        usage: {
          ...EMPTY_USAGE,
          inputTokens: 5,
          outputTokens: 3,
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
    });
    expect(wrapper.find(".sub-agent-panel").exists()).toBe(true);
    expect(wrapper.text()).toContain("planner");
  });

  it("子智能体用量：无 subAgents 时不渲染子智能体面板（不产生空壳）", () => {
    const tool: ToolCall = {
      callId: "c1",
      name: "task",
      status: "success",
      subAgent: true,
    };
    const wrapper = mountAssistant({
      message: assistant({
        id: 10,
        blocks: [{ id: "t1", kind: "tool_use", seq: 1, tool }],
      }),
    });
    expect(wrapper.find(".sub-agent-panel").exists()).toBe(false);
  });
});
