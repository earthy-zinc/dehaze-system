// 会话视图单测：激活链原样透传（不再前端过滤）、分支序号取自消息自身 sibling 字段、
// 链上 pending 阻塞输入、会话级批量产物/反馈、单条产物增量去重、分支切换解析目标。
import { createPinia, setActivePinia } from "pinia";
import { effectScope, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Artifact, Block, Feedback, Message } from "clarity-sdk-js";
import { AiConversationAPI } from "clarity-sdk-js";
import { useChatSessionView } from "@/composables/useChatSessionView";
import { useChatStore } from "@/store/modules/chat";

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return {
    ...actual,
    AiConversationAPI: {
      ...actual.AiConversationAPI,
      getArtifacts: vi.fn(),
      getMessageArtifacts: vi.fn(),
      getConversationFeedback: vi.fn(),
      getBranches: vi.fn(),
      switchBranch: vi.fn(),
    },
  };
});

function textBlock(id: string, seq: number, text: string): Block {
  return { id, kind: "text", seq, text };
}

function message(id: number, over: Partial<Message> = {}): Message {
  return {
    id,
    conversationId: 1,
    parentMessageId: null,
    role: id % 2 === 0 ? "assistant" : "user",
    status: "completed",
    model: null,
    error: null,
    blocks: [textBlock(`b-${id}`, 1, `内容 ${id}`)],
    plan: null,
    pendingInterrupt: null,
    usage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      credits: 0,
    },
    artifactIds: [],
    memoryIds: [],
    suggestions: [],
    edited: false,
    createTime: "2026-01-01T00:00:00",
    updateTime: null,
    ...over,
  };
}

function artifact(id: number, messageId: number): Artifact {
  return {
    id,
    conversationId: 1,
    messageId,
    type: "metric_report",
    isInvalid: false,
  };
}

function feedback(messageId: number): Feedback {
  return { id: messageId, messageId, userId: 1, rating: 1 };
}

/** 写入实体表与顺序（顺序即后端返回的激活链） */
function seed(
  chat: ReturnType<typeof useChatStore>,
  list: Message[],
  order: number[] = list.map((item) => item.id)
) {
  const record: Record<number, Message> = {};
  for (const item of list) record[item.id] = item;
  chat.messages = record;
  chat.messageOrder = order;
  chat.currentConversationId = 1;
}

function bind() {
  const scope = effectScope();
  const view = scope.run(() => useChatSessionView())!;
  return { view, scope };
}

/** 等待批量请求链（多级 await）全部落地 */
async function settle() {
  for (let i = 0; i < 6; i++) await Promise.resolve();
  await nextTick();
}

describe("useChatSessionView", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.mocked(AiConversationAPI.getArtifacts).mockReset();
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockReset();
    vi.mocked(AiConversationAPI.getConversationFeedback).mockReset();
    vi.mocked(AiConversationAPI.getBranches).mockReset();
    vi.mocked(AiConversationAPI.switchBranch).mockReset();
    vi.mocked(AiConversationAPI.getArtifacts).mockResolvedValue({
      list: [],
      total: 0,
    });
    vi.mocked(AiConversationAPI.getConversationFeedback).mockResolvedValue([]);
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue([]);
  });

  it("messages 直接是激活链：后端返回什么就渲染什么，前端不再过滤兄弟分支", () => {
    const chat = useChatStore();
    seed(chat, [
      message(1, { role: "user" }),
      message(2, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 1,
        siblingCount: 2,
      }),
      message(3, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 2,
        siblingCount: 2,
      }),
    ]);

    const { view, scope } = bind();

    // 不复制、不再做"每个分叉选一条"的可见性过滤
    expect(view.messages.value).toBe(chat.messageList);
    expect(view.messages.value.map((item) => item.id)).toEqual([1, 2, 3]);
    scope.stop();
  });

  it("分支序号只取自消息自身的 siblingIndex/siblingCount，并复用序号对象", () => {
    const chat = useChatStore();
    seed(chat, [
      message(1, { role: "user" }),
      message(2, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 1,
        siblingCount: 2,
      }),
      message(4, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 2,
        siblingCount: 2,
      }),
      // 无分叉：siblingCount 为 1 或无字段都不产生切换器
      message(6, {
        role: "assistant",
        parentMessageId: 5,
        siblingIndex: 1,
        siblingCount: 1,
      }),
      // root 级助手消息后端不填 sibling 字段
      message(8, { role: "assistant", parentMessageId: null }),
    ]);

    const { view, scope } = bind();

    const first = view.branches.value[2];
    expect(first).toEqual({ current: 1, total: 2 });
    expect(view.branches.value[4]).toEqual({ current: 2, total: 2 });
    expect(view.branches.value[6]).toBeUndefined();
    expect(view.branches.value[8]).toBeUndefined();
    expect(view.branches.value[1]).toBeUndefined();

    // 流式帧替换实体（新对象、同 id）：序号对象仍复用，子组件 props 引用稳定
    chat.messages = {
      ...chat.messages,
      2: { ...chat.messages[2], blocks: [textBlock("b-2", 1, "流式增量")] },
    };
    expect(view.branches.value[2]).toBe(first);
    scope.stop();
  });

  it("pending 只扫描激活链：不在链上的兄弟残留中断不阻塞输入", () => {
    const chat = useChatStore();
    const pendingInterrupt = {
      id: "int-1",
      type: "confirm" as const,
      confirmKind: "tool_permission" as const,
      createdAt: "2026-01-01T00:00:00",
      expiresAt: "2099-01-01T00:00:00",
      actionable: true,
    };
    seed(
      chat,
      [
        message(1, { role: "user" }),
        // 兄弟分支遗留中断：实体表里存在，但不在激活链（messageOrder）上
        message(2, {
          role: "assistant",
          parentMessageId: 1,
          status: "awaiting_input",
          pendingInterrupt,
        }),
        message(3, { role: "assistant", parentMessageId: 1 }),
      ],
      [1, 3]
    );

    const { view, scope } = bind();
    expect(view.messages.value.map((item) => item.id)).toEqual([1, 3]);
    expect(view.isStreaming.value).toBe(false);
    expect(view.interruptedMessageId.value).toBeNull();

    // 中断挂在链上消息时才阻塞输入
    chat.messages = {
      ...chat.messages,
      3: {
        ...chat.messages[3],
        status: "awaiting_input",
        pendingInterrupt: { ...pendingInterrupt, id: "int-2" },
      },
    };
    expect(view.isStreaming.value).toBe(true);
    expect(view.interruptedMessageId.value).toBe(3);

    // 过期中断不阻塞
    chat.messages = {
      ...chat.messages,
      3: {
        ...chat.messages[3],
        pendingInterrupt: {
          ...chat.messages[3].pendingInterrupt!,
          actionable: false,
        },
      },
    };
    expect(view.isStreaming.value).toBe(false);
    scope.stop();
  });

  it("刷新后 idle + 链上持久化 actionable 中断：分支切换仍禁用", () => {
    const chat = useChatStore();
    seed(chat, [
      message(1, { role: "user" }),
      message(2, {
        role: "assistant",
        parentMessageId: 1,
        status: "awaiting_input",
        pendingInterrupt: {
          id: "int-1",
          type: "confirm",
          confirmKind: "tool_permission",
          createdAt: "2026-01-01T00:00:00",
          expiresAt: "2099-01-01T00:00:00",
          actionable: true,
        },
      }),
    ]);

    const { view, scope } = bind();
    // 刷新后没有活跃流，但后端仍持有该轮流锁
    expect(chat.streamStatus).toBe("idle");
    expect(view.branchDisabled.value).toBe(true);

    // 已过期（不可执行）的中断不再锁切换
    chat.messages = {
      ...chat.messages,
      2: {
        ...chat.messages[2],
        pendingInterrupt: {
          ...chat.messages[2].pendingInterrupt!,
          actionable: false,
        },
      },
    };
    expect(view.branchDisabled.value).toBe(false);
    scope.stop();
  });

  it("进入会话按会话批量拉取产物（一次请求）并按 messageId 归组", async () => {
    const chat = useChatStore();
    vi.mocked(AiConversationAPI.getArtifacts).mockResolvedValue({
      list: [artifact(9, 2), artifact(10, 2), artifact(11, 4)],
      total: 3,
    });
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
      message(3, { role: "user", parentMessageId: 2 }),
      message(4, { role: "assistant", parentMessageId: 3 }),
    ]);

    const { view, scope } = bind();
    await settle();

    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledWith(1, {
      pageNum: 1,
      pageSize: 100,
    });
    expect(view.artifacts.value[2].map((item) => item.id)).toEqual([9, 10]);
    expect(view.artifacts.value[4].map((item) => item.id)).toEqual([11]);
    scope.stop();
  });

  it("产物总数超过一页时继续翻页直到取满 total", async () => {
    const chat = useChatStore();
    const firstPage = Array.from({ length: 100 }, (_, i) => artifact(i + 1, 2));
    const secondPage = Array.from({ length: 20 }, (_, i) =>
      artifact(101 + i, 2)
    );
    vi.mocked(AiConversationAPI.getArtifacts)
      .mockResolvedValueOnce({ list: firstPage, total: 120 })
      .mockResolvedValueOnce({ list: secondPage, total: 120 });
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);

    const { view, scope } = bind();
    await settle();

    expect(AiConversationAPI.getArtifacts).toHaveBeenCalledTimes(2);
    expect(AiConversationAPI.getArtifacts).toHaveBeenNthCalledWith(2, 1, {
      pageNum: 2,
      pageSize: 100,
    });
    expect(view.artifacts.value[2]).toHaveLength(120);
    scope.stop();
  });

  it("反馈按会话一次批量回填，不覆盖其它消息的既有反馈", async () => {
    const chat = useChatStore();
    chat.feedbackByMessage = { 99: feedback(99) };
    vi.mocked(AiConversationAPI.getConversationFeedback).mockResolvedValue([
      feedback(2),
      feedback(4),
    ]);
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
      message(3, { role: "user", parentMessageId: 2 }),
      message(4, { role: "assistant", parentMessageId: 3 }),
    ]);

    const { view, scope } = bind();
    await settle();

    expect(AiConversationAPI.getConversationFeedback).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getConversationFeedback).toHaveBeenCalledWith(1);
    expect(chat.feedbackByMessage[2]?.rating).toBe(1);
    expect(chat.feedbackByMessage[4]?.rating).toBe(1);
    // 不在本批结果里的已有反馈保持原样
    expect(chat.feedbackByMessage[99]?.messageId).toBe(99);
    expect(view.feedback.value[2]?.messageId).toBe(2);

    // 同一会话内的列表刷新（如切分支后重拉）不重复批量请求
    chat.messagesLoading = true;
    await nextTick();
    chat.messagesLoading = false;
    await nextTick();
    expect(AiConversationAPI.getConversationFeedback).toHaveBeenCalledTimes(1);
    scope.stop();
  });

  it("批量反馈以本会话为准：未返回的既有反馈视为已撤销，其它会话不受影响", async () => {
    const chat = useChatStore();
    chat.feedbackByMessage = {
      2: feedback(2),
      4: feedback(4),
      99: feedback(99),
    };
    vi.mocked(AiConversationAPI.getConversationFeedback).mockResolvedValue([
      feedback(4),
    ]);
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
      message(3, { role: "user", parentMessageId: 2 }),
      message(4, { role: "assistant", parentMessageId: 3 }),
    ]);

    const { scope } = bind();
    await settle();

    // 批量返回的照常回填；本会话消息 2 已有反馈但批量未返回 → 已撤销
    expect(chat.feedbackByMessage[4]?.rating).toBe(1);
    expect(chat.feedbackByMessage[2]).toBeUndefined();
    // 范围外的条目保持原样
    expect(chat.feedbackByMessage[99]?.messageId).toBe(99);
    scope.stop();
  });

  it("切换会话时丢弃上一个会话的批量产物回包（竞态）", async () => {
    const chat = useChatStore();
    let resolveFirst!: (value: { list: Artifact[]; total: number }) => void;
    const firstPage = new Promise<{ list: Artifact[]; total: number }>(
      (resolve) => {
        resolveFirst = resolve;
      }
    );
    vi.mocked(AiConversationAPI.getArtifacts).mockImplementation(
      (conversationId: number) =>
        conversationId === 1
          ? (firstPage as never)
          : Promise.resolve({ list: [], total: 0 } as never)
    );
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);

    const { view, scope } = bind();
    await nextTick();

    // 回包前会话已切走：旧会话的产物不能写进当前视图
    chat.currentConversationId = 2;
    await nextTick();
    resolveFirst({ list: [artifact(9, 2)], total: 1 });
    await settle();

    expect(view.artifacts.value).toEqual({});
    scope.stop();
  });

  it("流式新消息终态仍走单条产物增量路径，同一消息只拉一次", async () => {
    const chat = useChatStore();
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue([
      artifact(9, 8),
    ]);

    const { view, scope } = bind();
    await settle();

    // 批量结果覆盖链上消息后，只有批量之后出现的新消息才逐条补拉
    await view.onLoadArtifacts(2);
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();

    await view.onLoadArtifacts(8);
    await view.onLoadArtifacts(8);
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(8);
    expect(view.artifacts.value[8]).toHaveLength(1);
    scope.stop();
  });

  it("批量加载期间挂载的新消息在批量落地后仍会补拉产物", async () => {
    const chat = useChatStore();
    let resolveBatch!: (value: { list: Artifact[]; total: number }) => void;
    vi.mocked(AiConversationAPI.getArtifacts).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveBatch = resolve;
        }) as never
    );
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue([
      artifact(12, 8),
    ]);
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);

    const { view, scope } = bind();
    await nextTick();

    // 批量还没回包：新消息挂载时先记账，不并发逐条请求
    await view.onLoadArtifacts(8);
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();

    resolveBatch({ list: [], total: 0 });
    await settle();

    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledTimes(1);
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(8);
    expect(view.artifacts.value[8]).toHaveLength(1);
    scope.stop();
  });

  it("批量在途时新完成的消息不在覆盖快照内：批量落地后仍单条补齐产物", async () => {
    const chat = useChatStore();
    let resolveBatch!: (value: { list: Artifact[]; total: number }) => void;
    vi.mocked(AiConversationAPI.getArtifacts).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveBatch = resolve;
        }) as never
    );
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue([
      artifact(12, 8),
    ]);
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);

    const { view, scope } = bind();
    await nextTick();

    // 批量请求在途：流式消息完成并拿到服务端 id（批量响应里没有它的产物）
    chat.messages = {
      ...chat.messages,
      8: message(8, { role: "assistant", parentMessageId: 3 }),
    };
    chat.messageOrder = [...chat.messageOrder, 8];

    resolveBatch({ list: [], total: 0 });
    await settle();

    // id 8 在批量成功时已在列表里，但不属于批量发起时的快照，不能被覆盖判断跳过
    await view.onLoadArtifacts(8);
    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(8);
    expect(view.artifacts.value[8]).toHaveLength(1);
    scope.stop();
  });

  it("批量产物失败时在途消息回退单条路径，不丢产物", async () => {
    const chat = useChatStore();
    let rejectBatch!: (reason: unknown) => void;
    vi.mocked(AiConversationAPI.getArtifacts).mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectBatch = reject;
        }) as never
    );
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValue([
      artifact(12, 8),
    ]);
    seed(chat, [
      message(1, { role: "user" }),
      message(2, { role: "assistant", parentMessageId: 1 }),
    ]);

    const { view, scope } = bind();
    await nextTick();

    // 批量在途：终态消息挂载时先记账，不并发逐条请求
    await view.onLoadArtifacts(8);
    expect(AiConversationAPI.getMessageArtifacts).not.toHaveBeenCalled();

    rejectBatch(new Error("boom"));
    await settle();

    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(8);
    expect(view.artifacts.value[8]).toHaveLength(1);
    scope.stop();
  });

  it("分支切换向服务端解析兄弟目标（不写本地选择态）", async () => {
    const chat = useChatStore();
    seed(chat, [
      message(1, { role: "user" }),
      message(2, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 2,
        siblingCount: 2,
      }),
    ]);
    // 后端按时间倒序返回兄弟，第二位才是展示序号 1
    vi.mocked(AiConversationAPI.getBranches).mockResolvedValue([
      message(2, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 2,
        siblingCount: 2,
      }),
      message(4, {
        role: "assistant",
        parentMessageId: 1,
        siblingIndex: 1,
        siblingCount: 2,
      }),
    ]);
    const switchBranch = vi.fn().mockResolvedValue(undefined);
    chat.switchBranch = switchBranch;

    const { view, scope } = bind();
    await view.onSwitchBranch(chat.messages[2], 1);

    expect(AiConversationAPI.getBranches).toHaveBeenCalledWith(1, 1);
    expect(switchBranch).toHaveBeenCalledWith(4);
    // 本地不写选择态：展示仍然只由 store 的激活链决定
    expect(view.messages.value.map((item) => item.id)).toEqual([1, 2]);
    scope.stop();
  });

  it("动作原样转交 store", async () => {
    const chat = useChatStore();
    seed(chat, [message(2, { role: "assistant" })]);
    const dispatch = vi.fn().mockResolvedValue(undefined);
    chat.dispatchMessageAction = dispatch;

    const { view, scope } = bind();
    await view.onMessageAction({ type: "delete", messageId: 2 });

    expect(dispatch).toHaveBeenCalledWith({ type: "delete", messageId: 2 });
    scope.stop();
  });
});
