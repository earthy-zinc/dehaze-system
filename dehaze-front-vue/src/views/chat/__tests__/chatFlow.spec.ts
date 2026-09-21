// 用户端对话页端到端回归：真实 chatStore + 宿主绑定层（useChatVmBinding）+ ai-chat 无状态组件 + 页面层逻辑。
// 以 SDK（AiConversationAPI）为桩边界回放 wire 契约事件，覆盖：
// 发送 → 流式增量 → 停止 → 重新生成 → 编辑重发 → 删除 → 引用 → 朗读 → 复制 → 中断恢复 → 产物打开 → 历史分页加载。
// 页面级：挂载真实 views/chat/index.vue（重组件用桩隔离），以 DOM 交互驱动绑定层 emit→store 回接。
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { ElMessage, ElMessageBox } from "element-plus";
import { defineComponent, h, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import {
  AiConversationAPI,
  type AiMessageVO,
  type MessageStreamHandlers,
} from "dehaze-sdk-js";
import { elStubs, findButton } from "@/components/ai-chat/__tests__/stubs";
import { useChatStore } from "@/store/modules/chat";
import { useVoiceStore } from "@/store/modules/voice";
import ChatPage from "@/views/chat/index.vue";

// MarkdownRenderer 依赖 katex/mermaid，用轻量桩替代（与 ai-chat 组件测试口径一致）
vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

let handlers: MessageStreamHandlers;

/** 构造流式开启函数：捕获 handler（sendMessage/regenerate/editMessage/resumeMessage 共用） */
function streamFn() {
  return vi.fn((...args: unknown[]) => {
    handlers = args[args.length - 1] as MessageStreamHandlers;
    return new AbortController();
  });
}

vi.mock("dehaze-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("dehaze-sdk-js")>();
  return {
    ...actual,
    AiModelAPI: {
      ...actual.AiModelAPI,
      listEnabledModels: vi.fn(),
    },
    AiBillingAPI: {
      ...actual.AiBillingAPI,
      getBalance: vi.fn(),
    },
    AiConversationAPI: {
      ...actual.AiConversationAPI,
      getConversations: vi.fn(),
      createConversation: vi.fn(),
      getMessages: vi.fn(),
      getBranches: vi.fn(),
      switchBranch: vi.fn(),
      markConversationRead: vi.fn(),
      sendMessage: streamFn(),
      regenerate: streamFn(),
      editMessage: streamFn(),
      resumeMessage: streamFn(),
      stopMessage: vi.fn(),
      reconnectStream: vi.fn(),
      getMessageDetail: vi.fn(),
      getMessageArtifacts: vi.fn(),
      getArtifactDetail: vi.fn(),
      getFeedback: vi.fn(),
      submitFeedback: vi.fn(),
      deleteFeedback: vi.fn(),
      deleteMessage: vi.fn(),
    },
  };
});

/** 输入区桩：暴露 send/stop 触发点（真实 MessageInput 依赖语音/上传，与本回归无关） */
const MessageInputStub = defineComponent({
  name: "MessageInput",
  emits: ["send", "stop"],
  setup:
    (_props, { emit }) =>
    () =>
      h("div", { class: "message-input-stub" }, [
        h(
          "button",
          { class: "mi-send", onClick: () => emit("send", "你好") },
          "send"
        ),
        h("button", { class: "mi-stop", onClick: () => emit("stop") }, "stop"),
      ]),
});

/** 产物详情对话框桩：渲染 default 插槽以便断言页面层详情内容 */
const ElDialogStub = defineComponent({
  name: "ElDialog",
  props: ["modelValue", "title"],
  setup:
    (props, { slots }) =>
    () =>
      props.modelValue
        ? h("div", { class: "el-dialog-stub" }, slots.default?.())
        : null,
});

const STUBS = {
  ChatSidebar: true,
  ChatHeader: true,
  QuotaIndicator: true,
  EmptyState: true,
  ConversationSettings: true,
  MemoryPanel: true,
  SaveAsTaskDialog: true,
  MessageInput: MessageInputStub,
  "el-dialog": ElDialogStub,
  ...elStubs,
};

async function mountPage() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: { render: () => null } },
      { path: "/chat/:conversationId", component: { render: () => null } },
    ],
  });
  const wrapper = mount(ChatPage, {
    global: { plugins: [router], stubs: STUBS },
  });
  // 等待页面 onMounted（initScope + 会话列表 + 模型/配额）完成
  await flushPromises();
  await flushPromises();
  return wrapper;
}

/** 让 Store 进入"已有会话 + 指定消息"的稳定态（页面 onMounted 之后调用） */
async function openConversation(wrapper: VueWrapper, list: AiMessageVO[]) {
  const chat = useChatStore();
  vi.mocked(AiConversationAPI.getMessages).mockResolvedValueOnce({
    list,
    total: list.length,
    hasMore: false,
  });
  await chat.fetchMessages(1);
  await flushPromises();
  await nextTick();
  return wrapper;
}

/** 逐帧刷入流式缓冲 + 等 DOM 更新 */
async function flushFrames() {
  await new Promise((resolve) => setTimeout(resolve, 5));
  await nextTick();
}

function setScroll(
  el: HTMLElement,
  opts: { scrollTop: number; scrollHeight: number; clientHeight: number }
) {
  Object.defineProperty(el, "scrollHeight", {
    configurable: true,
    value: opts.scrollHeight,
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    value: opts.clientHeight,
  });
  el.scrollTop = opts.scrollTop;
}

/** 生成降序 id 序列（from > to，后端消息分页为倒序口径） */
function descIds(from: number, to: number): number[] {
  return Array.from({ length: from - to + 1 }, (_, i) => from - i);
}

const START = {
  messageId: 4,
  conversationId: 1,
  model: "qwen3-0.6b",
  streamSessionId: "stream-1",
};

const assistantDone = (over: Partial<AiMessageVO> = {}): AiMessageVO => ({
  id: 4,
  conversationId: 1,
  role: "assistant",
  status: 2,
  content: "答",
  createTime: "2026-01-01T00:00:00Z",
  ...over,
});

const userMessage = (over: Partial<AiMessageVO> = {}): AiMessageVO => ({
  id: 3,
  conversationId: 1,
  role: "user",
  status: 2,
  content: "问",
  createTime: "2026-01-01T00:00:00Z",
  ...over,
});

describe("用户端对话页 end-to-end（store + 绑定层 + 无状态组件）", () => {
  beforeEach(() => {
    handlers = undefined!;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    // mockReset 会清空工厂内的初值，此处重设各接口默认返回（各用例按需 mockResolvedValueOnce 覆盖）
    vi.mocked(AiConversationAPI.getConversations).mockResolvedValue({
      list: [],
      total: 0,
    } as never);
    vi.mocked(AiConversationAPI.createConversation).mockResolvedValue({
      id: 1,
      title: "新对话",
    } as never);
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [],
      total: 0,
      hasMore: false,
    } as never);
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([] as never);
    vi.mocked(AiConversationAPI.switchBranch).mockResolvedValue({
      id: 1,
    } as never);
    vi.mocked(AiConversationAPI.markConversationRead).mockResolvedValue(
      {} as never
    );
    vi.mocked(AiConversationAPI.stopMessage).mockResolvedValue({
      id: 4,
      status: 4,
    } as never);
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue(
      [] as never
    );
    vi.mocked(AiConversationAPI.getArtifactDetail).mockResolvedValue({
      id: 11,
      type: "metric_report",
    } as never);
    vi.mocked(AiConversationAPI.getFeedback).mockResolvedValue(null as never);
    vi.mocked(AiConversationAPI.submitFeedback).mockResolvedValue({
      id: 1,
      messageId: 4,
      userId: 1,
      rating: 1,
      createTime: "2026-01-01T00:00:00Z",
    } as never);
    vi.mocked(AiConversationAPI.deleteMessage).mockResolvedValue(
      undefined as never
    );
  });

  it("发送（空态先建会话）→ 流式增量渲染 → 停止落取消态", async () => {
    const wrapper = await mountPage();

    // 空态发送：先建会话再发送
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.createConversation).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);

    const chat = useChatStore();
    expect(chat.currentConversationId).toBe(1);
    expect(chat.messages.map((m) => m.role)).toEqual(["user", "assistant"]);

    // 流式：message.start 绑定服务端 id → 正文增量
    handlers.onStart?.(START);
    handlers.onContentBlockDelta?.({
      index: 0,
      delta: { type: "text_delta", text: "你是" },
    });
    handlers.onContentBlockDelta?.({
      index: 0,
      delta: { type: "text_delta", text: "好" },
    });
    await flushFrames();
    expect(wrapper.text()).toContain("你是好");

    // 停止 → 取消态 + 调用停止接口（服务端已落库 id）
    await wrapper.find(".mi-stop").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.stopMessage).toHaveBeenCalledWith(4);
    expect(chat.messages.find((m) => m.id === 4)?.status).toBe(4);
    expect(chat.streamingMessageId).toBeNull();
  });

  it("重新生成：DOM 操作栏点击 → store.regenerate → SDK.regenerate", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);

    await findButton(wrapper, "重新生成")!.trigger("click");
    await flushPromises();
    expect(AiConversationAPI.regenerate).toHaveBeenCalledWith(
      4,
      expect.anything()
    );
  });

  it("编辑重发：二次确认通过后调用 editMessage 并续流", async () => {
    vi.spyOn(ElMessageBox, "prompt").mockResolvedValue({
      value: "改后的问题",
      action: "confirm",
    } as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [userMessage()]);

    await findButton(wrapper, "编辑")!.trigger("click");
    await flushPromises();
    expect(AiConversationAPI.editMessage).toHaveBeenCalledWith(
      3,
      { content: "改后的问题" },
      expect.anything()
    );
  });

  it("删除：二次确认通过后调用 deleteMessage 并从列表移除", async () => {
    vi.spyOn(ElMessageBox, "confirm").mockResolvedValue("confirm" as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);

    await findButton(wrapper, "删除")!.trigger("click");
    await flushPromises();
    expect(AiConversationAPI.deleteMessage).toHaveBeenCalledWith(4);
    expect(useChatStore().messages.some((m) => m.id === 4)).toBe(false);
  });

  it("引用：写入 store.quotedMessage", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);

    await findButton(wrapper, "引用")!.trigger("click");
    await flushPromises();
    expect(useChatStore().quotedMessage?.id).toBe(4);
  });

  it("朗读：speak 回接 voiceStore.playSpeech（纯文本化后）", async () => {
    const voice = useVoiceStore();
    const playSpeech = vi
      .spyOn(voice, "playSpeech")
      .mockResolvedValue(undefined);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone({ content: "**答**案" })]);

    await findButton(wrapper, "朗读")!.trigger("click");
    await flushPromises();
    expect(playSpeech).toHaveBeenCalledWith("答案");
  });

  it("复制：剪贴板写入消息内容 + 成功提示（页面层）", async () => {
    const success = vi
      .spyOn(ElMessage, "success")
      .mockImplementation(() => ({}) as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [userMessage({ content: "要复制的内容" })]);

    await findButton(wrapper, "复制")!.trigger("click");
    await flushPromises();
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("要复制的内容");
    expect(success).toHaveBeenCalledWith("已复制");
  });

  it("中断恢复：confirm(tool_permission) 卡片点击 → resume 载荷回接 store→SDK", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onInterrupt?.({
      type: "confirm",
      data: { confirmKind: "tool_permission", tool: "write_file" },
    });
    await nextTick();

    expect(wrapper.find(".interrupt-card").exists()).toBe(true);
    await findButton(wrapper, "允许")!.trigger("click");
    await flushPromises();
    expect(AiConversationAPI.resumeMessage).toHaveBeenCalledWith(
      4,
      { confirm: true },
      expect.anything()
    );
  });

  it("计划确认：新增任务 → 传给 SDK 的 plan_edit.add 为单对象（非数组）", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onInterrupt?.({
      type: "plan_approve",
      data: {
        plan: {
          tasks: [{ id: "t1", description: "步骤一", dependsOn: [] }],
          revisions: [],
        },
      },
    });
    await nextTick();

    expect(wrapper.find(".interrupt-card").exists()).toBe(true);
    await findButton(wrapper, "+ 添加任务")!.trigger("click");
    const inputs = wrapper.findAll(".el-input-stub");
    await inputs[inputs.length - 1].setValue("新任务");
    await findButton(wrapper, "批准执行")!.trigger("click");
    await flushPromises();

    expect(AiConversationAPI.resumeMessage).toHaveBeenCalledWith(
      4,
      { planEdit: { add: { description: "新任务", dependsOn: [] } } },
      expect.anything()
    );
  });

  it("产物：load-artifacts 回填 VM + 点击卡片打开页面层详情对话框", async () => {
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValueOnce([
      {
        id: 11,
        conversationId: 1,
        messageId: 4,
        type: "metric_report",
        isInvalid: 0,
      },
    ] as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    await flushPromises();
    await nextTick();

    // 终态助手消息挂载即 emit load-artifacts
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(4);
    expect(wrapper.find(".artifact-card").exists()).toBe(true);

    await wrapper.find(".artifact-card").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.getArtifactDetail).toHaveBeenCalledWith(11);
    expect(wrapper.find(".el-dialog-stub").text()).toContain("metric_report");
  });

  it("历史分页：滚动到顶触发 reach-top → loadMoreMessages 以最早 id 为 before 前置更早消息", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();

    // 首页：最新 50 条（后端 id 倒序口径，hasMore）；更早页：剩余 50 条（到底）
    vi.mocked(AiConversationAPI.getMessages)
      .mockResolvedValueOnce({
        list: descIds(100, 51).map((id) =>
          assistantDone({ id, content: `m${id}` })
        ),
        total: 100,
        hasMore: true,
      })
      .mockResolvedValueOnce({
        list: descIds(50, 1).map((id) =>
          assistantDone({ id, content: `m${id}` })
        ),
        total: 100,
        hasMore: false,
      });
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    expect(chat.messages).toHaveLength(50);
    expect(chat.messagesHasMore).toBe(true);

    const body = wrapper.find(".chat-message-list__body");
    setScroll(body.element as HTMLElement, {
      scrollTop: 0,
      scrollHeight: 5000,
      clientHeight: 600,
    });
    await body.trigger("scroll");
    await flushPromises();

    // 首页最早一条 id=51 → before=51
    expect(AiConversationAPI.getMessages).toHaveBeenLastCalledWith(1, {
      before: 51,
      limit: 50,
    });
    expect(chat.messages).toHaveLength(100);
    expect(chat.messages[0].id).toBe(1);
    expect(chat.messages[99].id).toBe(100);
    expect(chat.messagesHasMore).toBe(false);
  });

  it("推荐问题：apply-suggestion 点击 → 作为新消息发送", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    chat.suggestions = ["换个角度呢"];
    await nextTick();

    const suggestion = wrapper.find(".suggestion-list__item");
    expect(suggestion.exists()).toBe(true);
    await suggestion.trigger("click");
    await flushPromises();
    // 推荐问题作为新消息发送 → 触发一次新的 sendMessage
    expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);
    expect(chat.suggestions).toEqual([]);
  });

  it("用户消息：wire edited/originalContent 接入后渲染「已编辑」与原文入口", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      userMessage({ edited: 1, originalContent: "改前内容" }),
    ]);
    await nextTick();

    expect(wrapper.find(".user-message__edited").text()).toContain("已编辑");
    expect(wrapper.find(".user-message__original").exists()).toBe(true);
  });

  it("用户消息：未编辑时不渲染「已编辑」标识", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [userMessage()]);
    await nextTick();

    expect(wrapper.find(".user-message__edited").text()).not.toContain(
      "已编辑"
    );
    expect(wrapper.find(".user-message__original").exists()).toBe(false);
  });

  it("工具消息：wire status 接入后展示执行状态徽标", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      {
        id: 9,
        conversationId: 1,
        role: "system",
        status: 2,
        content: '{"ok":true}',
        createTime: "2026-01-01T00:00:00Z",
      },
    ]);
    await nextTick();

    const tool = wrapper.find(".tool-message");
    expect(tool.exists()).toBe(true);
    expect(tool.text()).toContain("完成");
  });

  it("计划：plan 事件在助手消息内渲染计划面板（任务/状态/修订）", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onPlan?.({
      tasks: [
        { id: "t1", description: "步骤一", dependsOn: [], status: "done" },
        {
          id: "t2",
          description: "步骤二",
          dependsOn: ["t1"],
          status: "executing",
        },
      ],
      status: "executing",
      revisions: [],
      phase: "executing",
    });
    await nextTick();

    const panel = wrapper.find(".plan-panel");
    expect(panel.exists()).toBe(true);
    expect(panel.text()).toContain("步骤一");
    expect(panel.text()).toContain("步骤二");
    expect(panel.text()).toContain("执行中");
    expect(panel.text()).toContain("依赖：t1");
  });

  it("过程透明：thought + usage 派生过程面板，展开显示时间线/摘要/计费", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onThought?.({
      position: 1,
      thought: "先检索相关算法",
      status: 1,
      latencyMs: 120,
    });
    handlers.onThought?.({
      position: 2,
      tool: "enhance_image",
      toolInput: {},
      observation: "已增强",
      status: 2,
      error: "超时",
      latencyMs: 800,
    });
    handlers.onEnd?.({
      usage: {
        inputTokens: 10,
        outputTokens: 5,
        cachedInputTokens: 2,
        credits: 3,
      },
    } as never);
    await nextTick();

    const panel = wrapper.find(".process-panel");
    expect(panel.exists()).toBe(true);
    // 默认折叠：不渲染过程正文
    expect(panel.find(".process-panel__body").exists()).toBe(false);

    await panel.find(".process-panel__header").trigger("click");
    await nextTick();
    expect(wrapper.find(".step-timeline").exists()).toBe(true);
    expect(wrapper.text()).toContain("enhance_image");
    expect(wrapper.text()).toContain("共 3 步");
    expect(wrapper.text()).toContain("1 步失败");
    expect(wrapper.text()).toContain("消耗 3 积分");
  });

  it("过程透明：wire usedMemoryIds 派生上下文构成记忆标签（不虚构）", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      assistantDone({ id: 4, usedMemoryIds: [7, 8] }),
    ]);
    await nextTick();

    const panel = wrapper.find(".process-panel");
    expect(panel.exists()).toBe(true);
    // 无步骤时上下文构成仍展示
    await panel.find(".process-panel__header").trigger("click");
    await nextTick();
    const chip = wrapper.find(".context-chip");
    expect(chip.exists()).toBe(true);
    expect(chip.text()).toContain("记忆");
  });

  it("分支：切换走真实端点（getBranches→switchBranch→刷新消息列表）并切换展示分支", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();

    const siblings = [
      assistantDone({ id: 5, content: "分支二", parentMessageId: 3 }),
      assistantDone({ id: 4, content: "分支一", parentMessageId: 3 }),
      userMessage(),
    ];
    // 首屏与切换后刷新均返回同一批消息（持久 mock，供 fetchMessages 复用）
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: siblings,
      total: siblings.length,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    expect(wrapper.find(".branch-switcher").exists()).toBe(true);
    expect(wrapper.find(".branch-switcher__index").text()).toBe("2/2");
    // 默认仅展示最新分支（其余兄弟折叠保留）
    expect(wrapper.findAll(".assistant-message")).toHaveLength(1);
    expect(wrapper.text()).toContain("分支二");
    expect(wrapper.text()).not.toContain("分支一");

    // 后端按时间倒序返回分叉点(3)的兄弟分支
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      assistantDone({ id: 5, content: "分支二", parentMessageId: 3 }),
      assistantDone({ id: 4, content: "分支一", parentMessageId: 3 }),
    ] as never);
    const fetchCalls = vi.mocked(AiConversationAPI.getMessages).mock.calls
      .length;

    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    await nextTick();

    // 真切换：拉真实兄弟分支 → 调 switchBranch 落库 → 刷新消息列表（复用 fetchMessages）
    expect(AiConversationAPI.getBranches).toHaveBeenCalledWith(1, 3);
    expect(AiConversationAPI.switchBranch).toHaveBeenCalledWith(1, 4);
    expect(vi.mocked(AiConversationAPI.getMessages).mock.calls.length).toBe(
      fetchCalls + 1
    );

    expect(wrapper.find(".branch-switcher__index").text()).toBe("1/2");
    expect(wrapper.text()).toContain("分支一");
    expect(wrapper.text()).not.toContain("分支二");
  });

  it("分支：切换失败给用户可见提示并保持原展示分支（不静默、不预置选中态）", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    const siblings = [
      assistantDone({ id: 5, content: "分支二", parentMessageId: 3 }),
      assistantDone({ id: 4, content: "分支一", parentMessageId: 3 }),
      userMessage(),
    ];
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: siblings,
      total: siblings.length,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    const error = vi
      .spyOn(ElMessage, "error")
      .mockImplementation(() => ({}) as never);
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      assistantDone({ id: 5, content: "分支二", parentMessageId: 3 }),
      assistantDone({ id: 4, content: "分支一", parentMessageId: 3 }),
    ] as never);
    vi.mocked(AiConversationAPI.switchBranch).mockRejectedValue(
      new Error("boom")
    );

    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    await nextTick();

    expect(error).toHaveBeenCalledWith("切换分支失败，请稍后重试");
    // 失败保持原状态：仍展示最新分支（分支二）
    expect(wrapper.find(".branch-switcher__index").text()).toBe("2/2");
    expect(wrapper.text()).toContain("分支二");
    expect(wrapper.text()).not.toContain("分支一");
  });
});
