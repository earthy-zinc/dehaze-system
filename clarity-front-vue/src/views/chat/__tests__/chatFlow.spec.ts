// 用户端对话页端到端回归：真实 chatStore + 会话视图（useChatSessionView）+ ai-chat 无状态组件 + 页面层逻辑。
// 以 SDK（AiConversationAPI）为桩边界回放 wire 契约事件（message.start / block.* / plan.update / interrupt / usage / suggestions / message.end），覆盖：
// 发送 → 流式增量 → 停止 → 重新生成 → 编辑重发 → 删除 → 引用 → 朗读 → 复制 → 中断恢复 → 产物打开 → 历史分页加载。
// 页面级：挂载真实 views/chat/index.vue（重组件用桩隔离），以 DOM 交互驱动绑定层单一 message-action → store 回接。
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { ElMessage, ElMessageBox } from "element-plus";
import { defineComponent, h, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import {
  AiConversationAPI,
  type Block,
  type Interrupt,
  type Message,
  type MessageStartEvent,
  type MessageStreamHandlers,
  type Usage,
} from "clarity-sdk-js";
import { elStubs, findButton } from "@/components/ai-chat/__tests__/stubs";
import { useChatStore } from "@/store/modules/chat";
import { useVoiceStore } from "@/store/modules/voice";
import ChatPage from "@/views/chat/index.vue";

// MarkdownRenderer 依赖 katex/mermaid，用轻量桩替代（与 ai-chat 组件测试口径一致）
vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content", "streaming"],
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

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
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
      getArtifacts: vi.fn(),
      getMessageArtifacts: vi.fn(),
      getArtifactDetail: vi.fn(),
      getConversationFeedback: vi.fn(),
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

// ===== 契约 Message / Block 夹具（blocks 是唯一内容源） =====

const usageOf = (over: Partial<Usage> = {}): Usage => ({
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  credits: 0,
  ...over,
});

const textBlock = (id: string, seq: number, text: string): Block => ({
  id,
  kind: "text",
  seq,
  text,
});

const baseMessage = (id: number, conversationId = 1): Message => ({
  id,
  conversationId,
  parentMessageId: null,
  role: "assistant",
  status: "completed",
  model: null,
  error: null,
  blocks: [],
  plan: null,
  pendingInterrupt: null,
  usage: usageOf(),
  artifactIds: [],
  memoryIds: [],
  suggestions: [],
  edited: false,
  createTime: "2026-01-01T00:00:00Z",
  updateTime: null,
});

const assistantDone = (over: Partial<Message> = {}): Message => ({
  ...baseMessage(4),
  blocks: [textBlock("4-text", 1, "答")],
  ...over,
});

const userMessage = (over: Partial<Message> = {}): Message => ({
  ...baseMessage(3),
  role: "user",
  blocks: [textBlock("3-text", 1, "问")],
  ...over,
});

const interrupt = (over: Partial<Interrupt> = {}): Interrupt => ({
  id: "int-1",
  type: "confirm",
  createdAt: "2026-01-01T00:00:00Z",
  expiresAt: "2099-01-01T00:00:00Z",
  actionable: true,
  ...over,
});

/** message.start：服务端已落库的用户消息 + 助手消息（流初始为空块） */
function startEvent(): MessageStartEvent {
  const user = userMessage({ id: 3, blocks: [textBlock("3-text", 1, "你好")] });
  return {
    streamSessionId: "stream-1",
    userMessage: user,
    assistantMessage: {
      ...assistantDone({ id: 4, status: "running", blocks: [] }),
      parentMessageId: user.id,
    },
  };
}

const START = startEvent();

/** 新一轮发送（id 与既有链不重叠）：用于流式期间不牵连既有分支的用例 */
const NEXT_START: MessageStartEvent = {
  streamSessionId: "stream-2",
  userMessage: userMessage({
    id: 6,
    blocks: [textBlock("6-text", 1, "下一问")],
  }),
  assistantMessage: {
    ...assistantDone({ id: 7, status: "running", blocks: [] }),
    parentMessageId: 6,
  },
};

/** 契约 Artifact 夹具 */
const artifactOf = (id: number, messageId: number) => ({
  id,
  conversationId: 1,
  messageId,
  type: "metric_report",
  isInvalid: false,
});

/** 让 Store 进入"已有会话 + 指定消息"的稳定态（页面 onMounted 之后调用） */
async function openConversation(wrapper: VueWrapper, list: Message[]) {
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
    vi.mocked(AiConversationAPI.stopMessage).mockResolvedValue(
      assistantDone({ status: "canceled" }) as never
    );
    vi.mocked(AiConversationAPI.getArtifacts).mockResolvedValue({
      list: [],
      total: 0,
    } as never);
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue(
      [] as never
    );
    vi.mocked(AiConversationAPI.getArtifactDetail).mockResolvedValue({
      id: 11,
      type: "metric_report",
    } as never);
    vi.mocked(AiConversationAPI.getConversationFeedback).mockResolvedValue(
      [] as never
    );
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
    expect(chat.messageList.map((m) => m.role)).toEqual(["user", "assistant"]);

    // 流式：message.start 绑定服务端 id → block.start / block.delta 正文增量
    handlers.onStart?.(START);
    handlers.onBlockStart?.({
      messageId: 4,
      block: { id: "4-text", kind: "text", seq: 1, text: "" },
    });
    handlers.onBlockDelta?.({
      messageId: 4,
      blockId: "4-text",
      delta: "你是",
    });
    handlers.onBlockDelta?.({
      messageId: 4,
      blockId: "4-text",
      delta: "好",
    });
    await flushFrames();
    expect(wrapper.text()).toContain("你是好");

    // 停止 → 取消态 + 调用停止接口（服务端已落库 id）
    await wrapper.find(".mi-stop").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.stopMessage).toHaveBeenCalledWith(4);
    expect(chat.messages[4]?.status).toBe("canceled");
    expect(chat.streamMessageId).toBeNull();
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
    expect(useChatStore().messages[4]).toBeUndefined();
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
    await openConversation(wrapper, [
      assistantDone({ blocks: [textBlock("4-text", 1, "**答**案")] }),
    ]);

    await findButton(wrapper, "朗读")!.trigger("click");
    await flushPromises();
    expect(playSpeech).toHaveBeenCalledWith("答案");
  });

  it("复制：剪贴板写入消息内容 + 成功提示（页面层）", async () => {
    const success = vi
      .spyOn(ElMessage, "success")
      .mockImplementation(() => ({}) as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      userMessage({ blocks: [textBlock("3-text", 1, "要复制的内容")] }),
    ]);

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
      messageId: 4,
      interrupt: interrupt({
        confirmKind: "tool_permission",
        detail: "调用 write_file",
      }),
    });
    // 中断后本轮通道关闭（契约：message.end awaiting_input）
    handlers.onEnd?.({ messageId: 4, status: "awaiting_input" });
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
      messageId: 4,
      interrupt: interrupt({
        type: "plan_approve",
        confirmKind: null,
        plan: [{ id: "t1", description: "步骤一", dependsOn: [] }],
      }),
    });
    handlers.onEnd?.({ messageId: 4, status: "awaiting_input" });
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

  it("中断过期：actionable=false 时按钮禁用、提示不可继续且不发起 resume", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onInterrupt?.({
      messageId: 4,
      interrupt: interrupt({
        type: "quota",
        actionable: false,
        expiresAt: "2026-01-01T00:00:01Z",
        upgradeTip: "积分不足",
      }),
    });
    handlers.onEnd?.({ messageId: 4, status: "awaiting_input" });
    await nextTick();

    expect(wrapper.find(".interrupt-card").text()).toContain("已失效");
    const retry = findButton(wrapper, "重试")!;
    expect(retry.attributes("disabled")).toBeDefined();
    await retry.trigger("click");
    await flushPromises();
    expect(AiConversationAPI.resumeMessage).not.toHaveBeenCalled();
  });

  it("产物：进入会话按会话批量拉取（1 次）后回填，点击卡片打开页面层详情对话框", async () => {
    vi.mocked(AiConversationAPI.getArtifacts).mockResolvedValueOnce({
      list: [artifactOf(11, 4)],
      total: 1,
    } as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    await flushPromises();
    await nextTick();

    // 会话级批量：一次 getArtifacts，不再逐条 getMessageArtifacts
    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledWith(1, {
      pageNum: 1,
      pageSize: 100,
    });
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();
    expect(wrapper.find(".artifact-card").exists()).toBe(true);

    await wrapper.find(".artifact-card").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.getArtifactDetail).toHaveBeenCalledWith(11);
    expect(wrapper.find(".el-dialog-stub").text()).toContain("metric_report");
  });

  it("产物：会话产物超过一页时按 pageNum 翻页取满 total", async () => {
    vi.mocked(AiConversationAPI.getArtifacts)
      .mockResolvedValueOnce({
        list: Array.from({ length: 100 }, (_, i) => artifactOf(i + 1, 4)),
        total: 120,
      } as never)
      .mockResolvedValueOnce({
        list: Array.from({ length: 20 }, (_, i) => artifactOf(101 + i, 4)),
        total: 120,
      } as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    await flushPromises();
    await nextTick();

    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(2);
    expect(AiConversationAPI.getArtifacts).toHaveBeenNthCalledWith(2, 1, {
      pageNum: 2,
      pageSize: 100,
    });
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();
  });

  it("反馈：进入会话一次批量回填，不再逐条查询", async () => {
    vi.mocked(AiConversationAPI.getConversationFeedback).mockResolvedValueOnce([
      {
        id: 1,
        messageId: 4,
        userId: 1,
        rating: 1,
        createTime: "2026-01-01T00:00:00Z",
      },
    ] as never);
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    await flushPromises();
    await nextTick();

    expect(AiConversationAPI.getConversationFeedback).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getConversationFeedback).toHaveBeenCalledWith(1);
    expect(AiConversationAPI.getFeedback).not.toHaveBeenCalled();
    expect(useChatStore().feedbackByMessage[4]?.rating).toBe(1);
  });

  it("附加数据：切换会话后上一个会话的迟到回包被丢弃（不串会话）", async () => {
    const chat = useChatStore();
    let resolveArtifacts!: (value: unknown) => void;
    let resolveFeedback!: (value: unknown) => void;
    const staleArtifacts = new Promise((resolve) => {
      resolveArtifacts = resolve;
    });
    const staleFeedback = new Promise((resolve) => {
      resolveFeedback = resolve;
    });
    vi.mocked(AiConversationAPI.getArtifacts).mockImplementation(
      (conversationId: number) =>
        (conversationId === 1
          ? staleArtifacts
          : Promise.resolve({ list: [], total: 0 })) as never
    );
    vi.mocked(AiConversationAPI.getConversationFeedback).mockImplementation(
      (conversationId: number) =>
        (conversationId === 1 ? staleFeedback : Promise.resolve([])) as never
    );
    vi.mocked(AiConversationAPI.getMessages)
      .mockResolvedValueOnce({
        list: [assistantDone({ id: 4 })],
        total: 1,
        hasMore: false,
      } as never)
      .mockResolvedValueOnce({
        list: [assistantDone({ id: 8 })],
        total: 1,
        hasMore: false,
      } as never);

    const wrapper = await mountPage();
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();
    await chat.fetchMessages(2);
    await flushPromises();
    await nextTick();

    // 会话 1 的批量请求此时才回包：当前会话已是 2，必须整体丢弃
    resolveArtifacts({
      list: [artifactOf(11, 4)],
      total: 1,
    });
    resolveFeedback([
      {
        id: 1,
        messageId: 4,
        userId: 1,
        rating: 1,
        createTime: "2026-01-01T00:00:00Z",
      },
    ]);
    await flushPromises();
    await nextTick();

    expect(chat.currentConversationId).toBe(2);
    expect(chat.feedbackByMessage[4]).toBeUndefined();
    expect(wrapper.find(".artifact-card").exists()).toBe(false);
  });

  it("产物：批量之后出现的流式新消息终态仍走单条增量", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone()]);
    await flushPromises();
    await nextTick();
    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();

    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValueOnce([
      artifactOf(12, 7),
    ] as never);
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(NEXT_START);
    handlers.onEnd?.({ messageId: 7, status: "completed" });
    await flushPromises();
    await nextTick();

    // 新消息不在批量结果里：终态后单条补拉一次
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(7);
    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(1);
  });

  it("历史分页：滚动到顶触发 reach-top → loadMoreMessages 以最早 id 为 before 前置更早消息", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();

    // 首页：最新 50 条（后端 id 倒序口径，hasMore）；更早页：剩余 50 条（到底）
    vi.mocked(AiConversationAPI.getMessages)
      .mockResolvedValueOnce({
        list: descIds(100, 51).map((id) =>
          assistantDone({
            id,
            blocks: [textBlock(`${id}-text`, 1, `m${id}`)],
          })
        ),
        total: 100,
        hasMore: true,
      })
      .mockResolvedValueOnce({
        list: descIds(50, 1).map((id) =>
          assistantDone({
            id,
            blocks: [textBlock(`${id}-text`, 1, `m${id}`)],
          })
        ),
        total: 100,
        hasMore: false,
      });
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    expect(chat.messageList).toHaveLength(50);
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
    expect(chat.messageList).toHaveLength(100);
    expect(chat.messageList[0].id).toBe(1);
    expect(chat.messageList[99].id).toBe(100);
    expect(chat.messagesHasMore).toBe(false);
  });

  it("推荐问题：apply-suggestion 点击 → 作为新消息发送", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      assistantDone({ suggestions: ["换个角度呢"] }),
    ]);
    await nextTick();

    const suggestion = wrapper.find(".suggestion-list__item");
    expect(suggestion.exists()).toBe(true);
    await suggestion.trigger("click");
    await flushPromises();
    // 推荐问题作为新消息发送 → 触发一次新的 sendMessage
    expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);
  });

  it("用户消息：edited=true 时渲染「已编辑」标识", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [
      userMessage({
        edited: true,
        blocks: [textBlock("3-text", 1, "改前内容")],
      }),
    ]);
    await nextTick();

    expect(wrapper.find(".user-message__edited").text()).toContain("已编辑");
  });

  it("用户消息：未编辑时不渲染「已编辑」标识", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [userMessage()]);
    await nextTick();

    expect(wrapper.find(".user-message__edited").text()).not.toContain(
      "已编辑"
    );
  });

  it("计划：plan.update 在助手消息内渲染计划面板（任务/状态/修订）", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onPlanUpdate?.({
      messageId: 4,
      plan: {
        tasks: [
          {
            id: "t1",
            description: "步骤一",
            dependsOn: [],
            status: "completed",
          },
          {
            id: "t2",
            description: "步骤二",
            dependsOn: ["t1"],
            status: "running",
          },
        ],
        status: "running",
        revisions: [],
        phase: "executing",
        awaitingApproval: false,
      },
    });
    await nextTick();

    const panel = wrapper.find(".plan-panel");
    expect(panel.exists()).toBe(true);
    expect(panel.text()).toContain("步骤一");
    expect(panel.text()).toContain("步骤二");
    expect(panel.text()).toContain("执行中");
    expect(panel.text()).toContain("依赖：t1");
  });

  it("过程透明：工具步骤由 blocks 派生，逐条用量在 footer 展示", async () => {
    const wrapper = await mountPage();
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(START);
    handlers.onBlockStart?.({
      messageId: 4,
      block: {
        id: "4-tool",
        kind: "tool_use",
        seq: 1,
        tool: {
          callId: "c-1",
          name: "enhance_image",
          input: {},
          status: "running",
          subAgent: false,
        },
      },
    });
    handlers.onBlockStop?.({
      messageId: 4,
      block: {
        id: "4-tool",
        kind: "tool_use",
        seq: 1,
        tool: {
          callId: "c-1",
          name: "enhance_image",
          input: {},
          output: "已增强",
          status: "failed",
          error: "超时",
          latencyMs: 800,
          subAgent: false,
        },
      },
    });
    handlers.onEnd?.({
      messageId: 4,
      status: "completed",
      usage: usageOf({
        inputTokens: 10,
        outputTokens: 5,
        cachedInputTokens: 2,
        credits: 3,
      }),
    });
    await nextTick();

    expect(wrapper.find(".thought-chain").exists()).toBe(true);
    expect(wrapper.text()).toContain("查看过程（1 步）");
    expect(wrapper.text()).toContain("enhance_image");
    expect(wrapper.text()).toContain("1 步失败");
    expect(wrapper.text()).toContain(
      "输入 10 / 输出 5 tokens（缓存 2） · 积分 3"
    );
  });

  it("记忆引用：memoryIds 在 footer 报条数，不伪造上下文构成", async () => {
    const wrapper = await mountPage();
    await openConversation(wrapper, [assistantDone({ memoryIds: [7, 8] })]);
    await nextTick();

    expect(wrapper.text()).toContain("引用记忆 2 条");
    expect(wrapper.find(".process-panel").exists()).toBe(false);
  });

  it("分支：激活链由后端给出，切换走真实端点（getBranches→switchBranch→重拉链）", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();

    const user = userMessage();
    const branchTwo = assistantDone({
      id: 5,
      parentMessageId: 3,
      siblingIndex: 2,
      siblingCount: 2,
      blocks: [textBlock("5-text", 1, "分支二")],
    });
    const branchOne = assistantDone({
      id: 4,
      parentMessageId: 3,
      siblingIndex: 1,
      siblingCount: 2,
      blocks: [textBlock("4-text", 1, "分支一")],
    });
    // 首屏：后端 self 视角只返回激活链（分支二）；切换后重拉返回分支一
    vi.mocked(AiConversationAPI.getMessages)
      .mockResolvedValueOnce({
        list: [branchTwo, user],
        total: 2,
        hasMore: false,
      } as never)
      .mockResolvedValueOnce({
        list: [branchOne, user],
        total: 2,
        hasMore: false,
      } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    expect(wrapper.find(".branch-switcher").exists()).toBe(true);
    expect(wrapper.find(".branch-switcher__index").text()).toBe("2/2");
    // 激活链只含当前分支：前端不再折叠兄弟
    expect(wrapper.findAll(".assistant-message")).toHaveLength(1);
    expect(wrapper.text()).toContain("分支二");
    expect(wrapper.text()).not.toContain("分支一");

    // 目标兄弟不在激活链上：切换前向服务端取分叉点的全部兄弟（时间倒序）
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      branchTwo,
      branchOne,
    ] as never);
    const fetchCalls = vi.mocked(AiConversationAPI.getMessages).mock.calls
      .length;

    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    await nextTick();

    expect(AiConversationAPI.getBranches).toHaveBeenCalledWith(1, 3);
    expect(AiConversationAPI.switchBranch).toHaveBeenCalledWith(1, 4);
    expect(vi.mocked(AiConversationAPI.getMessages).mock.calls.length).toBe(
      fetchCalls + 1
    );

    // 展示以服务端重拉的链为准
    expect(wrapper.find(".branch-switcher__index").text()).toBe("1/2");
    expect(wrapper.text()).toContain("分支一");
    expect(wrapper.text()).not.toContain("分支二");
  });

  it("分支：切换失败给用户可见提示并保持服务端原链（不本地预置选中态）", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    const user = userMessage();
    const branchTwo = assistantDone({
      id: 5,
      parentMessageId: 3,
      siblingIndex: 2,
      siblingCount: 2,
      blocks: [textBlock("5-text", 1, "分支二")],
    });
    const branchOne = assistantDone({
      id: 4,
      parentMessageId: 3,
      siblingIndex: 1,
      siblingCount: 2,
      blocks: [textBlock("4-text", 1, "分支一")],
    });
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [branchTwo, user],
      total: 2,
      hasMore: false,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    const error = vi
      .spyOn(ElMessage, "error")
      .mockImplementation(() => ({}) as never);
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      branchTwo,
      branchOne,
    ] as never);
    vi.mocked(AiConversationAPI.switchBranch).mockRejectedValue(
      new Error("boom")
    );

    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    await nextTick();

    expect(error).toHaveBeenCalledWith("切换分支失败，请稍后重试");
    // 失败保持原链：仍展示分支二，不写任何本地选择态
    expect(wrapper.find(".branch-switcher__index").text()).toBe("2/2");
    expect(wrapper.text()).toContain("分支二");
    expect(wrapper.text()).not.toContain("分支一");
  });

  it("分支：流式进行中禁用切换器且不发起请求，message.end 后恢复", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    const user = userMessage();
    const branchTwo = assistantDone({
      id: 5,
      parentMessageId: 3,
      siblingIndex: 2,
      siblingCount: 2,
      blocks: [textBlock("5-text", 1, "分支二")],
    });
    const branchOne = assistantDone({
      id: 4,
      parentMessageId: 3,
      siblingIndex: 1,
      siblingCount: 2,
      blocks: [textBlock("4-text", 1, "分支一")],
    });
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [branchTwo, user],
      total: 2,
      hasMore: false,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    // 默认展示服务端激活链（2/2）：此时「上一条」可点
    expect(
      wrapper.find(".branch-switcher__prev").attributes("disabled")
    ).toBeUndefined();

    // 发起新一轮流（新消息 id 与既有链不重叠）：running 期间两个按钮都禁用
    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(NEXT_START);
    await nextTick();

    expect(chat.streamStatus).toBe("running");
    expect(
      wrapper.find(".branch-switcher__prev").attributes("disabled")
    ).toBeDefined();
    expect(
      wrapper.find(".branch-switcher__next").attributes("disabled")
    ).toBeDefined();
    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.getBranches).not.toHaveBeenCalled();
    expect(AiConversationAPI.switchBranch).not.toHaveBeenCalled();

    // message.end 回到 idle：切换恢复
    handlers.onEnd?.({ messageId: 7, status: "completed" });
    await nextTick();
    expect(chat.streamStatus).toBe("idle");
    expect(
      wrapper.find(".branch-switcher__prev").attributes("disabled")
    ).toBeUndefined();

    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      branchTwo,
      branchOne,
    ] as never);
    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();

    expect(AiConversationAPI.getBranches).toHaveBeenCalledWith(1, 3);
    expect(AiConversationAPI.switchBranch).toHaveBeenCalledWith(1, 4);
  });

  it("分支：中断等待中同样禁用切换器且不发起请求（后端仍持有流锁）", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    const user = userMessage();
    const branchTwo = assistantDone({
      id: 5,
      parentMessageId: 3,
      siblingIndex: 2,
      siblingCount: 2,
      blocks: [textBlock("5-text", 1, "分支二")],
    });
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [branchTwo, user],
      total: 2,
      hasMore: false,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    await wrapper.find(".mi-send").trigger("click");
    await flushPromises();
    handlers.onStart?.(NEXT_START);
    handlers.onInterrupt?.({
      messageId: 7,
      interrupt: interrupt({ confirmKind: "tool_permission" }),
    });
    handlers.onEnd?.({ messageId: 7, status: "awaiting_input" });
    await nextTick();

    expect(chat.streamStatus).toBe("interrupted");
    expect(
      wrapper.find(".branch-switcher__prev").attributes("disabled")
    ).toBeDefined();
    expect(
      wrapper.find(".branch-switcher__next").attributes("disabled")
    ).toBeDefined();
    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.getBranches).not.toHaveBeenCalled();
    expect(AiConversationAPI.switchBranch).not.toHaveBeenCalled();
  });

  it("分支：刷新后 idle 但存在持久化中断，切换器仍禁用且不发起请求", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    const user = userMessage();
    const branchTwo = assistantDone({
      id: 5,
      parentMessageId: 3,
      siblingIndex: 2,
      siblingCount: 2,
      status: "awaiting_input",
      pendingInterrupt: interrupt({ confirmKind: "tool_permission" }),
      blocks: [textBlock("5-text", 1, "分支二")],
    });
    // 刷新进入会话：消息从历史接口还原，没有活跃流（全局态 idle）
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [branchTwo, user],
      total: 2,
      hasMore: false,
    } as never);
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    expect(chat.streamStatus).toBe("idle");
    expect(
      wrapper.find(".branch-switcher__prev").attributes("disabled")
    ).toBeDefined();
    expect(
      wrapper.find(".branch-switcher__next").attributes("disabled")
    ).toBeDefined();
    await wrapper.find(".branch-switcher__prev").trigger("click");
    await flushPromises();
    expect(AiConversationAPI.getBranches).not.toHaveBeenCalled();
    expect(AiConversationAPI.switchBranch).not.toHaveBeenCalled();
  });

  it("历史加载：首屏失败展示行内错误条（不出现空态），重试成功后渲染消息", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    vi.mocked(AiConversationAPI.getMessages).mockRejectedValueOnce(
      new Error("网络异常")
    );

    await expect(chat.fetchMessages(1)).rejects.toThrow("网络异常");
    await nextTick();

    expect(wrapper.find(".chat-message-list__error").text()).toContain(
      "网络异常"
    );
    // 失败不能落成"没有消息"的空态：用户需要看到错误与重试出口
    expect(wrapper.find("empty-state-stub").exists()).toBe(false);

    vi.mocked(AiConversationAPI.getMessages).mockResolvedValueOnce({
      list: [
        assistantDone({ blocks: [textBlock("4-text", 1, "恢复后的回复")] }),
      ],
      total: 1,
      hasMore: false,
    });
    await wrapper.find(".chat-message-list__retry").trigger("click");
    await flushPromises();
    await nextTick();

    expect(wrapper.find(".chat-message-list__error").exists()).toBe(false);
    expect(wrapper.text()).toContain("恢复后的回复");
  });

  it("历史加载：上翻失败保留 hasMore 并提示，重试沿用相同 before 游标", async () => {
    const chat = useChatStore();
    const wrapper = await mountPage();
    vi.mocked(AiConversationAPI.getMessages)
      .mockResolvedValueOnce({
        list: descIds(100, 51).map((id) =>
          assistantDone({
            id,
            blocks: [textBlock(`${id}-text`, 1, `m${id}`)],
          })
        ),
        total: 100,
        hasMore: true,
      })
      .mockRejectedValueOnce(new Error("翻页失败"))
      .mockResolvedValueOnce({
        list: descIds(50, 1).map((id) =>
          assistantDone({
            id,
            blocks: [textBlock(`${id}-text`, 1, `m${id}`)],
          })
        ),
        total: 100,
        hasMore: false,
      });
    await chat.fetchMessages(1);
    await flushPromises();
    await nextTick();

    const body = wrapper.find(".chat-message-list__body");
    setScroll(body.element as HTMLElement, {
      scrollTop: 0,
      scrollHeight: 5000,
      clientHeight: 600,
    });
    await body.trigger("scroll");
    await flushPromises();
    await nextTick();

    // 失败不能被当成"到底"
    expect(chat.messagesHasMore).toBe(true);
    expect(wrapper.find(".chat-message-list__error").text()).toContain(
      "翻页失败"
    );

    await wrapper.find(".chat-message-list__retry").trigger("click");
    await flushPromises();
    await nextTick();

    expect(AiConversationAPI.getMessages).toHaveBeenLastCalledWith(1, {
      before: 51,
      limit: 50,
    });
    expect(chat.messageList).toHaveLength(100);
    expect(chat.messagesHasMore).toBe(false);
    expect(wrapper.find(".chat-message-list__error").exists()).toBe(false);
  });
});
