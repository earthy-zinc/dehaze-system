// chat store 测试：以 AiConversationAPI 为桩边界直接回放新契约 SSE 事件，
// 覆盖实体绑定（message.start rekey）/块增量/计划/中断/用量/终态/error/async 轮询/
// 断线重连/超时/停止/游标分页/僵尸 running/initScope/单一 action 入口。
// 时间相关行为（重连间隔/轮询/超时）统一用假时钟驱动。
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AiConversationAPI,
  type Block,
  type Interrupt,
  type Message,
  type MessageList,
  type MessageStreamHandlers,
} from "clarity-sdk-js";
import { resolveConfirmKind, useChatStore } from "@/store/modules/chat";
import { MESSAGES_PAGE_SIZE } from "@/store/modules/chat/shared";
import { useVoiceStore } from "@/store/modules/voice";

let handlers: MessageStreamHandlers;
let controller: AbortController;
/** resume 续流的回调与载荷捕获（断言传给 SDK 的载荷形状） */
let resumeHandlers: MessageStreamHandlers;
let resumePayload: unknown;

vi.mock("clarity-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("clarity-sdk-js")>();
  return {
    ...actual,
    AiConversationAPI: {
      ...actual.AiConversationAPI,
      sendMessage: vi.fn(
        (_conversationId: number, _data: unknown, h: MessageStreamHandlers) => {
          handlers = h;
          controller = new AbortController();
          return controller;
        }
      ),
      resumeMessage: vi.fn(
        (_messageId: number, data: unknown, h: MessageStreamHandlers) => {
          resumePayload = data;
          resumeHandlers = h;
          controller = new AbortController();
          return controller;
        }
      ),
      regenerate: vi.fn((_messageId: number, h: MessageStreamHandlers) => {
        handlers = h;
        controller = new AbortController();
        return controller;
      }),
      stopMessage: vi.fn(),
      getMessageDetail: vi.fn(),
      reconnectStream: vi.fn(),
      getMessages: vi.fn(),
      getConversations: vi.fn(),
      getBranches: vi.fn(),
      switchBranch: vi.fn(),
      markConversationRead: vi.fn().mockResolvedValue({}),
      deleteMessage: vi.fn(),
      submitFeedback: vi.fn(),
      getFeedback: vi.fn(),
      deleteFeedback: vi.fn(),
    },
  };
});

const EMPTY_USAGE = {
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  credits: 0,
};

/** 契约 Message 夹具 */
function message(id: number, over: Partial<Message> = {}): Message {
  return {
    id,
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
    createTime: "2026-01-01T00:00:00",
    updateTime: null,
    ...over,
  };
}

function textBlock(id: string, seq: number, text: string): Block {
  return { id, kind: "text", seq, text, durationMs: null, tool: null };
}

function thinkingBlock(id: string, seq: number, text: string): Block {
  return { id, kind: "thinking", seq, text, durationMs: null, tool: null };
}

function startOf(userId = 101, assistantId = 102) {
  return {
    streamSessionId: "stream-1",
    userMessage: message(userId, {
      role: "user",
      status: "completed",
      blocks: [textBlock("u1", 1, "你好")],
    }),
    assistantMessage: message(assistantId, { status: "running" }),
  };
}

function confirmInterrupt(over: Partial<Interrupt> = {}): Interrupt {
  return {
    id: "i1",
    type: "confirm",
    createdAt: "2026-01-01T00:00:00",
    expiresAt: "2026-01-01T00:10:00",
    actionable: true,
    ...over,
  };
}

/** 构造一页游标消息（后端按 id 倒序口径：入参按 id 降序给出） */
function messagePage(ids: number[], hasMore = false): MessageList {
  return {
    list: ids.map((id) => message(id)),
    total: ids.length,
    hasMore,
  };
}

/** 生成降序 id 序列（from > to） */
function descIds(from: number, to: number): number[] {
  return Array.from({ length: from - to + 1 }, (_, i) => from - i);
}

/** 生成升序 id 序列（from < to） */
function ascIds(from: number, to: number): number[] {
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

/** 手控 Promise：制造"请求未返回期间切换会话"的竞态 */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function openStream() {
  const chat = useChatStore();
  chat.currentConversationId = 1;
  chat.sendMessage("你好");
  return chat;
}

/** 完成 message.start：本地占位被服务端实体替换 */
function startStream(chat: ReturnType<typeof useChatStore>) {
  handlers.onStart?.(startOf());
  return chat;
}

function messageOf(chat: ReturnType<typeof useChatStore>, id = 102) {
  return chat.messages[id];
}

const END_USAGE = {
  inputTokens: 10,
  outputTokens: 5,
  cachedInputTokens: 2,
  credits: 3,
};

describe("chat store：消息实体与流式状态机", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    handlers = undefined!;
    controller = undefined!;
    resumeHandlers = undefined!;
    resumePayload = undefined;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("首屏请求去重", () => {
    it("joinInflight 时并发触发的会话列表加载合并为一次请求", async () => {
      vi.mocked(AiConversationAPI.getConversations).mockResolvedValue({
        list: [],
        total: 0,
      } as never);

      const chat = useChatStore();
      chat.initScope("self");
      await Promise.all([
        chat.fetchConversations(true),
        chat.fetchConversations(true),
      ]);

      expect(AiConversationAPI.getConversations).toHaveBeenCalledTimes(1);
    });

    it("缺省（变更/搜索/筛选后）不复用在途请求，始终取最新数据", async () => {
      vi.mocked(AiConversationAPI.getConversations).mockResolvedValue({
        list: [],
        total: 0,
      } as never);

      const chat = useChatStore();
      chat.initScope("self");
      await Promise.all([chat.fetchConversations(), chat.fetchConversations()]);

      expect(AiConversationAPI.getConversations).toHaveBeenCalledTimes(2);
    });
  });

  describe("message.start 实体绑定", () => {
    it("用服务端 user/assistant 实体替换本地占位，顺序与流状态同步", () => {
      const chat = openStream();
      expect(chat.messageOrder.every((id) => id < 0)).toBe(true);
      expect(chat.streamStatus).toBe("running");
      expect(chat.streamStatus).not.toBe("idle");

      startStream(chat);

      expect(chat.messageOrder).toEqual([101, 102]);
      expect(chat.messages[101].role).toBe("user");
      expect(chat.messages[102].status).toBe("running");
      expect(chat.streamMessageId).toBe(102);
      expect(chat.streamMessageId).toBe(102);
      expect(chat.streamStatus).not.toBe("idle");
    });

    it("本地占位 ID 不因毫秒跳变碰撞", () => {
      let now = 1_000_000;
      vi.spyOn(Date, "now").mockImplementation(() => now++);

      const chat = openStream();
      const ids = chat.messageOrder;
      expect(ids).toHaveLength(2);
      expect(ids[0]).not.toBe(ids[1]);
    });

    it("流式进行中重复发送被拒绝（同时仅允许一个流）", () => {
      const chat = openStream();
      chat.sendMessage("第二条");
      expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);
      expect(chat.messageOrder).toHaveLength(2);
    });

    it("历史与流式共用同一实体形状：start 后正文来自 blocks", () => {
      const chat = startStream(openStream());
      expect(messageOf(chat).blocks).toEqual([]);
      expect(messageOf(chat, 101).blocks).toEqual([textBlock("u1", 1, "你好")]);
    });
  });

  describe("块增量归约", () => {
    it("block.start 追加块，block.delta 按 blockId 原位追加", () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({ messageId: 102, blockId: "b1", delta: "你好" });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "b1",
        delta: "，世界",
      });

      expect(messageOf(chat).blocks).toEqual([
        textBlock("b1", 1, "你好，世界"),
      ]);
    });

    it("正文与思考各自成块，互不混写", () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: thinkingBlock("t1", 1, ""),
      });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "t1",
        delta: "先思考",
      });
      handlers.onBlockStop?.({
        messageId: 102,
        block: thinkingBlock("t1", 1, "先思考"),
      });
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "b1",
        delta: "再回答",
      });

      const blocks = messageOf(chat).blocks;
      expect(blocks.map((block) => block.kind)).toEqual(["thinking", "text"]);
      expect(blocks[0].text).toBe("先思考");
      expect(blocks[1].text).toBe("再回答");
    });

    it("block.stop 用完整块替换（写入耗时与工具结果）", () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: {
          id: "tool1",
          kind: "tool_use",
          seq: 1,
          text: null,
          durationMs: null,
          tool: {
            callId: "c1",
            name: "get_weather",
            input: null,
            output: null,
            summary: null,
            status: "running",
            error: null,
            latencyMs: null,
            agentCode: null,
            subAgent: false,
          },
        },
      });
      // 工具参数增量不入实体
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "tool1",
        delta: '{"city":',
      });
      expect(messageOf(chat).blocks[0].text).toBeNull();

      handlers.onBlockStop?.({
        messageId: 102,
        block: {
          id: "tool1",
          kind: "tool_use",
          seq: 1,
          text: null,
          durationMs: 12,
          tool: {
            callId: "c1",
            name: "get_weather",
            input: { city: "北京" },
            output: "晴",
            summary: null,
            status: "success",
            error: null,
            latencyMs: 12,
            agentCode: null,
            subAgent: false,
          },
        },
      });

      const block = messageOf(chat).blocks[0];
      expect(block.durationMs).toBe(12);
      expect(block.tool?.input).toEqual({ city: "北京" });
      expect(block.tool?.status).toBe("success");
    });
  });

  describe("计划 / 用量 / 推荐 / 中断", () => {
    it("plan.update 写到消息实体", () => {
      const chat = startStream(openStream());
      handlers.onPlanUpdate?.({
        messageId: 102,
        plan: {
          phase: "plan",
          status: "running",
          tasks: [{ id: "t1", description: "步骤1", dependsOn: [] }],
          revisions: [],
          awaitingApproval: false,
        },
      });

      expect(messageOf(chat).plan?.tasks).toHaveLength(1);
      expect(messageOf(chat).plan?.phase).toBe("plan");
    });

    it("usage / suggestions 写到消息实体（不再有全局副本）", () => {
      const chat = startStream(openStream());
      handlers.onUsage?.({ messageId: 102, usage: END_USAGE });
      handlers.onSuggestions?.({
        messageId: 102,
        suggestions: ["然后呢", "换个话题"],
      });

      expect(messageOf(chat).usage).toEqual(END_USAGE);
      expect(messageOf(chat).suggestions).toEqual(["然后呢", "换个话题"]);
    });

    it("interrupt 写 pendingInterrupt，进入等待确认态", () => {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ confirmKind: "tool_permission" }),
      });

      expect(messageOf(chat).pendingInterrupt?.type).toBe("confirm");
      expect(chat.streamStatus).toBe("interrupted");
      expect(chat.streamMessageId).toBe(102);

      handlers.onEnd?.({
        messageId: 102,
        status: "awaiting_input",
        usage: END_USAGE,
      });
      expect(messageOf(chat).status).toBe("awaiting_input");
      expect(chat.streamStatus).toBe("interrupted");
      expect(chat.streamStatus).not.toBe("idle");
    });

    it("中断后连接关闭按正常收束处理，不判失败", () => {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ confirmKind: "tool_permission" }),
      });
      handlers.onClose?.();

      expect(messageOf(chat).status).toBe("awaiting_input");
      expect(messageOf(chat).error).toBeNull();
      expect(chat.streamMessageId).toBe(102);
    });
  });

  describe("message.end 终态", () => {
    it("completed：写终态与用量，清空中断", () => {
      const chat = startStream(openStream());
      handlers.onEnd?.({
        messageId: 102,
        status: "completed",
        usage: END_USAGE,
      });

      const msg = messageOf(chat);
      expect(msg.status).toBe("completed");
      expect(msg.usage).toEqual(END_USAGE);
      expect(msg.pendingInterrupt).toBeNull();
      expect(chat.streamStatus).toBe("idle");
      expect(chat.streamStatus).toBe("idle");
    });

    it("canceled / failed 直接落实体状态与错误文案", () => {
      const chat = startStream(openStream());
      handlers.onEnd?.({
        messageId: 102,
        status: "failed",
        error: "模型 qwen3-0.6b 未配置用户售价",
      });

      expect(messageOf(chat).status).toBe("failed");
      expect(messageOf(chat).error).toBe("模型 qwen3-0.6b 未配置用户售价");
      expect(chat.streamStatus).toBe("idle");
    });

    it("error 事件记录原因，终态由随后的 message.end 表达", () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({ messageId: 102, blockId: "b1", delta: "半截" });
      handlers.onError?.({
        code: "A0601",
        message: "模型未配置用户售价",
        messageId: 102,
      });

      expect(messageOf(chat).error).toBe("模型未配置用户售价");
      expect(messageOf(chat).status).toBe("running");

      handlers.onEnd?.({
        messageId: 102,
        status: "failed",
        error: "模型未配置用户售价",
      });
      handlers.onClose?.();

      expect(messageOf(chat).status).toBe("failed");
      expect(messageOf(chat).blocks[0].text).toBe("半截");
      expect(chat.streamStatus).toBe("idle");
    });
  });

  describe("async_wait 中断", () => {
    it("message.end 后轮询消息终态并整体替换实体", async () => {
      const chat = startStream(openStream());
      vi.mocked(AiConversationAPI.getMessageDetail).mockResolvedValue(
        message(102, {
          status: "completed",
          blocks: [textBlock("b9", 1, "异步完成")],
        })
      );
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ type: "async_wait" }),
      });
      handlers.onEnd?.({ messageId: 102, status: "awaiting_input" });

      await vi.advanceTimersByTimeAsync(5_000);

      expect(AiConversationAPI.getMessageDetail).toHaveBeenCalledWith(102);
      const msg = messageOf(chat);
      expect(msg.status).toBe("completed");
      expect(msg.blocks).toEqual([textBlock("b9", 1, "异步完成")]);
      expect(chat.streamStatus).toBe("idle");
    });
  });

  describe("断线重连", () => {
    it("携带 lastEventId 在 3 秒后自动重连", async () => {
      const chat = startStream(openStream());
      handlers.onEventId?.("3");

      handlers.onNetworkError?.(new Error("断开"));
      await vi.advanceTimersByTimeAsync(3_000);

      expect(AiConversationAPI.reconnectStream).toHaveBeenCalledWith(
        1,
        "stream-1",
        "3",
        expect.anything()
      );
      expect(messageOf(chat).status).toBe("running");
    });

    it("重连 3 次耗尽后判失败", async () => {
      const chat = startStream(openStream());

      for (let i = 0; i < 3; i++) {
        handlers.onNetworkError?.(new Error("断开"));
        await vi.advanceTimersByTimeAsync(3_000);
      }
      handlers.onNetworkError?.(new Error("断开"));

      const msg = messageOf(chat);
      expect(msg.status).toBe("failed");
      expect(msg.error).toContain("网络连接中断");
    });

    it("HTTP 业务错误按真实原因提示且不重连", async () => {
      const chat = startStream(openStream());
      const httpError = Object.assign(new Error("参数校验失败"), {
        name: "SSEHttpError",
        status: 400,
        code: "A0400",
        msg: "单条消息超出模型上下文预算（14136 字符，模型 qwen3-0.6b）",
      });

      handlers.onNetworkError?.(httpError);
      await vi.advanceTimersByTimeAsync(5_000);

      const msg = messageOf(chat);
      expect(msg.status).toBe("failed");
      expect(msg.error).toBe(
        "单条消息超出模型上下文预算（14136 字符，模型 qwen3-0.6b）"
      );
      expect(AiConversationAPI.reconnectStream).not.toHaveBeenCalled();
    });

    it("流已达终态后 onClose 不再重连", async () => {
      const chat = startStream(openStream());
      handlers.onEnd?.({ messageId: 102, status: "completed" });
      handlers.onClose?.();
      await vi.advanceTimersByTimeAsync(5_000);

      expect(AiConversationAPI.reconnectStream).not.toHaveBeenCalled();
      expect(messageOf(chat).status).toBe("completed");
    });
  });

  describe("流式超时", () => {
    it("120 秒无新内容判超时失败并中断连接", async () => {
      const chat = startStream(openStream());

      await vi.advanceTimersByTimeAsync(150_000);

      const msg = messageOf(chat);
      expect(msg.status).toBe("failed");
      expect(msg.error).toContain("流式输出超时");
      expect(controller.signal.aborted).toBe(true);
    });

    it("收到增量刷新计时；ping 不刷新", async () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });

      await vi.advanceTimersByTimeAsync(100_000);
      handlers.onBlockDelta?.({ messageId: 102, blockId: "b1", delta: "续命" });
      await vi.advanceTimersByTimeAsync(100_000);
      expect(messageOf(chat).status).toBe("running");

      handlers.onPing?.();
      await vi.advanceTimersByTimeAsync(30_000);
      expect(messageOf(chat).status).toBe("failed");
    });
  });

  describe("停止生成", () => {
    it("message.start 前停止：本地取消且不调停止接口", async () => {
      const chat = openStream();
      const localId = chat.streamMessageId!;

      await chat.stopStreaming();

      expect(AiConversationAPI.stopMessage).not.toHaveBeenCalled();
      expect(chat.messages[localId].status).toBe("canceled");
      expect(controller.signal.aborted).toBe(true);
      expect(chat.streamStatus).toBe("idle");
    });

    it("message.start 后停止：调用停止接口并用终态实体替换", async () => {
      const chat = startStream(openStream());
      vi.mocked(AiConversationAPI.stopMessage).mockResolvedValue(
        message(102, {
          status: "canceled",
          blocks: [textBlock("b1", 1, "半截")],
        })
      );

      await chat.stopStreaming();

      expect(AiConversationAPI.stopMessage).toHaveBeenCalledWith(102);
      expect(messageOf(chat).status).toBe("canceled");
      expect(messageOf(chat).blocks[0].text).toBe("半截");
    });
  });

  describe("历史消息游标分页", () => {
    it("首屏缺省 before 取最新页 + loadMoreMessages 以最早 id 为 before 向上增量加载", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockResolvedValueOnce(messagePage(descIds(50, 1), false));

      await chat.fetchMessages(1);
      expect(AiConversationAPI.getMessages).toHaveBeenNthCalledWith(1, 1, {
        limit: MESSAGES_PAGE_SIZE,
      });
      expect(chat.messageOrder).toEqual(ascIds(51, 100));
      expect(chat.messagesHasMore).toBe(true);

      await chat.loadMoreMessages();
      expect(AiConversationAPI.getMessages).toHaveBeenNthCalledWith(2, 1, {
        before: 51,
        limit: MESSAGES_PAGE_SIZE,
      });
      expect(chat.messageOrder).toEqual([...ascIds(1, 50), ...ascIds(51, 100)]);
      expect(chat.messagesHasMore).toBe(false);
    });

    it("到底：响应 hasMore=false 置 messagesHasMore=false 且后续 no-op", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(60, 11), true))
        .mockResolvedValueOnce(messagePage(descIds(10, 1), false));

      await chat.fetchMessages(1);
      expect(chat.messagesHasMore).toBe(true);

      await chat.loadMoreMessages();
      expect(chat.messagesHasMore).toBe(false);

      await chat.loadMoreMessages();
      expect(AiConversationAPI.getMessages).toHaveBeenCalledTimes(2);
    });

    it("并发：重复调用 loadMoreMessages 只发起一次请求", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockResolvedValueOnce(messagePage(descIds(50, 1), false));

      await chat.fetchMessages(1);
      await Promise.all([chat.loadMoreMessages(), chat.loadMoreMessages()]);
      expect(AiConversationAPI.getMessages).toHaveBeenCalledTimes(2);
    });

    it("边界重叠：加载更早页与已有页 id 重叠时按 id 去重、不重不漏", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockResolvedValueOnce(messagePage(descIds(55, 1), false));

      await chat.fetchMessages(1);
      await chat.loadMoreMessages();

      const ids = chat.messageOrder;
      expect(ids).toEqual(ascIds(1, 100));
      expect(new Set(ids).size).toBe(ids.length);
    });

    it("竞态：loadMoreMessages 进行中切换会话，过期响应被丢弃", async () => {
      const chat = useChatStore();
      const stalePage = deferred<MessageList>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockReturnValueOnce(stalePage.promise)
        .mockResolvedValueOnce(messagePage(descIds(200, 151), false));

      await chat.fetchMessages(1);
      expect(chat.messageOrder).toEqual(ascIds(51, 100));

      const pending = chat.loadMoreMessages();
      await chat.fetchMessages(2);
      const ids2 = chat.messageOrder;
      expect(chat.currentConversationId).toBe(2);
      expect(ids2).toEqual(ascIds(151, 200));

      stalePage.resolve(messagePage(descIds(50, 1), false));
      await pending;

      expect(chat.messageOrder).toEqual(ids2);
    });

    it("竞态：fetchMessages(A) 未返回即切到 B，A 的过期响应不覆盖 B", async () => {
      const chat = useChatStore();
      const stalePage = deferred<MessageList>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockReturnValueOnce(stalePage.promise)
        .mockResolvedValueOnce(messagePage(descIds(200, 151), false));

      const pending = chat.fetchMessages(1);
      await chat.fetchMessages(2);
      const ids2 = chat.messageOrder;

      stalePage.resolve(messagePage(descIds(100, 51), true));
      await pending;

      expect(chat.messageOrder).toEqual(ids2);
    });

    it("本地负数占位不能作为 before 游标", async () => {
      const chat = useChatStore();
      chat.currentConversationId = 1;
      chat.sendMessage("你好");
      vi.mocked(AiConversationAPI.getMessages).mockResolvedValue(
        messagePage(descIds(10, 1), false)
      );

      await chat.fetchMessages(1);
      await chat.loadMoreMessages();

      // 首屏没有正数消息时 loadMore 直接短路，不发请求
      expect(AiConversationAPI.getMessages).toHaveBeenCalledTimes(1);
    });
  });

  describe("历史加载失败态与重试", () => {
    it("首屏失败：记录 initial 错误且列表为空，重试成功后渲染并清空错误", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages).mockRejectedValueOnce(
        new Error("网络异常")
      );

      await expect(chat.fetchMessages(1)).rejects.toThrow("网络异常");
      expect(chat.messagesError).toEqual({
        kind: "initial",
        message: "网络异常",
      });
      expect(chat.messages).toEqual({});
      expect(chat.messageOrder).toEqual([]);
      expect(chat.messagesHasMore).toBe(false);

      vi.mocked(AiConversationAPI.getMessages).mockResolvedValueOnce(
        messagePage(descIds(10, 1), false)
      );
      await chat.retryMessages();

      expect(chat.messagesError).toBeNull();
      expect(chat.messageOrder).toEqual(ascIds(1, 10));
    });

    it("A→B 且 B 失败：列表不残留 A 的消息与分页状态", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages).mockResolvedValueOnce(
        messagePage(descIds(10, 1), true)
      );
      await chat.fetchMessages(1);
      expect(chat.messageOrder).toEqual(ascIds(1, 10));

      vi.mocked(AiConversationAPI.getMessages).mockRejectedValueOnce(
        new Error("网络异常")
      );
      await expect(chat.fetchMessages(2)).rejects.toThrow("网络异常");

      expect(chat.currentConversationId).toBe(2);
      expect(chat.messages).toEqual({});
      expect(chat.messageOrder).toEqual([]);
      expect(chat.messagesHasMore).toBe(false);
      expect(chat.messagesError?.kind).toBe("initial");
    });

    it("上翻失败：hasMore 保留原值、记录 more 错误且不产生未处理拒绝；重试沿用同一游标", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockRejectedValueOnce(new Error("网络异常"))
        .mockResolvedValueOnce(messagePage(descIds(50, 1), false));

      await chat.fetchMessages(1);
      // loadMore 失败不向上抛：滚动触发没有等待结果的调用方
      await expect(chat.loadMoreMessages()).resolves.toBeUndefined();

      expect(chat.messagesError).toEqual({
        kind: "more",
        message: "网络异常",
      });
      expect(chat.messagesHasMore).toBe(true);
      expect(chat.messageOrder).toEqual(ascIds(51, 100));

      await chat.retryMessages();

      expect(AiConversationAPI.getMessages).toHaveBeenNthCalledWith(3, 1, {
        before: 51,
        limit: MESSAGES_PAGE_SIZE,
      });
      expect(chat.messagesError).toBeNull();
      expect(chat.messagesHasMore).toBe(false);
      expect(chat.messageOrder).toEqual([...ascIds(1, 50), ...ascIds(51, 100)]);
    });

    it("同会话刷新失败：保留已有消息，错误仍可重试", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(10, 1), false))
        .mockRejectedValueOnce(new Error("网络异常"));
      await chat.fetchMessages(1);

      await expect(chat.fetchMessages(1)).rejects.toThrow("网络异常");

      expect(chat.messageOrder).toEqual(ascIds(1, 10));
      expect(chat.messagesError?.kind).toBe("initial");
    });

    it("上翻在途切会话：加载标志复位，过期结果被丢弃", async () => {
      const chat = useChatStore();
      const stalePage = deferred<MessageList>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockReturnValueOnce(stalePage.promise)
        .mockResolvedValueOnce(messagePage(descIds(200, 151), false));

      await chat.fetchMessages(1);
      const pending = chat.loadMoreMessages();
      expect(chat.messagesLoadingMore).toBe(true);

      await chat.fetchMessages(2);
      expect(chat.messagesLoadingMore).toBe(false);

      stalePage.resolve(messagePage(descIds(50, 1), false));
      await pending;

      expect(chat.messagesLoadingMore).toBe(false);
      expect(chat.messageOrder).toEqual(ascIds(151, 200));
    });

    it("切分支：指针成功后刷新失败只记历史加载错误，不误报切换失败", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages).mockResolvedValueOnce(
        messagePage(descIds(10, 1), false)
      );
      await chat.fetchMessages(1);
      vi.mocked(AiConversationAPI.switchBranch).mockResolvedValue({
        id: 1,
      } as never);
      vi.mocked(AiConversationAPI.getMessages).mockRejectedValueOnce(
        new Error("网络异常")
      );

      // 指针已切换：刷新失败不得抛成"切换分支失败"
      await expect(chat.switchBranch(5)).resolves.toBeUndefined();
      expect(AiConversationAPI.switchBranch).toHaveBeenCalledWith(1, 5);
      expect(chat.messagesError?.kind).toBe("initial");

      // 指针接口本身失败仍向上抛，由视图层提示
      vi.mocked(AiConversationAPI.switchBranch).mockRejectedValueOnce(
        new Error("指针写入失败")
      );
      await expect(chat.switchBranch(6)).rejects.toThrow("指针写入失败");
    });
  });

  describe("刷新后 running 僵尸", () => {
    it("无活跃流的 running 消息按 failed 收口", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
        list: [message(2, { status: "running" }), message(1, { role: "user" })],
        total: 2,
        hasMore: false,
      });

      await chat.fetchMessages(1);

      expect(chat.messages[2].status).toBe("failed");
      expect(chat.messages[2].error).toBe("上次回复未完成");
      expect(chat.streamStatus).toBe("idle");
    });

    it("同会话活跃流的 running 实体保留，不被历史覆盖", async () => {
      const chat = startStream(openStream());
      const live = messageOf(chat);
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "b1",
        delta: "流式中",
      });

      vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
        list: [
          message(102, { status: "running" }),
          message(101, { role: "user" }),
        ],
        total: 2,
        hasMore: false,
      });
      await chat.fetchMessages(1);

      const after = messageOf(chat);
      expect(after).toBe(live);
      expect(after.status).toBe("running");
      expect(after.blocks[0].text).toBe("流式中");
      expect(chat.streamStatus).not.toBe("idle");
    });
  });

  describe("跨会话 initScope", () => {
    it("切换 scope 不清消息、不打断进行中的流", () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "b1",
        delta: "进行中",
      });

      chat.initScope("admin");
      expect(chat.messageOrder).toEqual([101, 102]);
      expect(chat.currentConversationId).toBe(1);
      expect(chat.streamStatus).not.toBe("idle");
      expect(chat.messages[102].blocks[0].text).toBe("进行中");

      chat.initScope("self");
      expect(chat.messageOrder).toEqual([101, 102]);
      expect(chat.scope).toBe("self");
    });

    it("重复 initScope 同一 scope 不重置列表", () => {
      const chat = useChatStore();
      chat.initScope("self");
      chat.conversations = [
        {
          id: 1,
          title: "t",
          status: 1,
          messageCount: 0,
          pinned: 0,
          createTime: "2026-01-01T00:00:00",
        },
      ];
      chat.initScope("self");
      expect(chat.conversations).toHaveLength(1);
    });

    it("切到另一个会话：无活跃 session 的 interrupted 态复位，不指向旧消息", async () => {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ confirmKind: "tool_permission" }),
      });
      handlers.onEnd?.({ messageId: 102, status: "awaiting_input" });
      expect(chat.streamStatus).toBe("interrupted");
      expect(chat.streamMessageId).toBe(102);

      vi.mocked(AiConversationAPI.getMessages).mockResolvedValue(
        messagePage(descIds(2, 1), false)
      );
      await chat.fetchMessages(2);

      expect(chat.streamStatus).toBe("idle");
      expect(chat.streamMessageId).toBeNull();
    });
  });

  describe("confirmKind 读取（直接用契约字段）", () => {
    it.each<[string, Partial<Interrupt>]>([
      ["algorithm_recommend", { confirmKind: "algorithm_recommend" }],
      ["tool_permission", { confirmKind: "tool_permission" }],
      ["dangerous_op", { confirmKind: "dangerous_op" }],
      ["write_conflict", { confirmKind: "write_conflict" }],
    ])("confirm 子类型 %s 原样返回", (expected, over) => {
      expect(resolveConfirmKind(confirmInterrupt(over))).toBe(expected);
    });

    it("不再从 dangerous_op + action 推导 write_conflict", () => {
      expect(
        resolveConfirmKind(
          confirmInterrupt({
            confirmKind: "dangerous_op",
            action: "write_conflict",
          })
        )
      ).toBe("dangerous_op");
    });

    it("非 confirm 中断无子类型", () => {
      expect(
        resolveConfirmKind(confirmInterrupt({ type: "quota" }))
      ).toBeUndefined();
    });

    it("对抗：未知 / 缺失子类型返回 undefined（不静默兜底，也不抛错）", () => {
      expect(
        resolveConfirmKind(
          confirmInterrupt({ confirmKind: "bogus" as Interrupt["confirmKind"] })
        )
      ).toBeUndefined();
      expect(resolveConfirmKind(confirmInterrupt({}))).toBeUndefined();
    });
  });

  describe("resumeInterrupt 子类型路由", () => {
    function interruptOf(interrupt: Interrupt) {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({ messageId: 102, interrupt });
      handlers.onEnd?.({ messageId: 102, status: "awaiting_input" });
      return chat;
    }

    it("algorithm_recommend：{confirm, params:{modelId}}（取调用方选择）", () => {
      const chat = interruptOf(
        confirmInterrupt({
          confirmKind: "algorithm_recommend",
          recommendation: {
            modelId: "aecrnet",
            modelName: "AECR-Net",
            reason: "最佳",
          },
        })
      );
      chat.resumeInterrupt(102, { confirm: true, params: { modelId: "dcp" } });

      expect(resumePayload).toEqual({
        confirm: true,
        params: { modelId: "dcp" },
      });
    });

    it("algorithm_recommend：未指定则回退推荐模型标识", () => {
      const chat = interruptOf(
        confirmInterrupt({
          confirmKind: "algorithm_recommend",
          recommendation: {
            modelId: "aecrnet",
            modelName: "AECR-Net",
            reason: "最佳",
          },
        })
      );
      chat.resumeInterrupt(102, { confirm: true });

      expect(resumePayload).toEqual({
        confirm: true,
        params: { modelId: "aecrnet" },
      });
    });

    it("plan_approve：透传 planEdit", () => {
      const chat = interruptOf(confirmInterrupt({ type: "plan_approve" }));
      chat.resumeInterrupt(102, {
        planEdit: { remove: ["t1"], reorder: ["t2", "t3"] },
      });

      expect(resumePayload).toEqual({
        planEdit: { remove: ["t1"], reorder: ["t2", "t3"] },
      });
    });

    it("tool_permission：{confirm}", () => {
      const chat = interruptOf(
        confirmInterrupt({ confirmKind: "tool_permission" })
      );
      chat.resumeInterrupt(102, { confirm: true });
      expect(resumePayload).toEqual({ confirm: true });
    });

    it("对抗：未知 confirmKind 不发起 resume，不清中断点", () => {
      const chat = interruptOf(
        confirmInterrupt({ confirmKind: "bogus" as Interrupt["confirmKind"] })
      );
      chat.resumeInterrupt(102, { confirm: true });
      expect(messageOf(chat).pendingInterrupt).not.toBeNull();
      expect(messageOf(chat).status).toBe("awaiting_input");
    });

    it("续流首个事件到达才清中断点（未达前保留）", () => {
      const chat = interruptOf(
        confirmInterrupt({ confirmKind: "tool_permission" })
      );
      expect(messageOf(chat).pendingInterrupt).not.toBeNull();

      chat.resumeInterrupt(102, { confirm: true });
      expect(messageOf(chat).pendingInterrupt).not.toBeNull();

      resumeHandlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b2", 2, ""),
      });
      expect(messageOf(chat).pendingInterrupt).toBeNull();
      expect(chat.streamStatus).toBe("running");
    });

    it("恢复失败（会话忙）：保留中断点且不重复发起", () => {
      const chat = interruptOf(
        confirmInterrupt({ confirmKind: "tool_permission" })
      );
      chat.resumeInterrupt(102, { confirm: true });
      expect(AiConversationAPI.resumeMessage).toHaveBeenCalledTimes(1);

      chat.resumeInterrupt(102, { confirm: true });
      expect(AiConversationAPI.resumeMessage).toHaveBeenCalledTimes(1);
      expect(messageOf(chat).pendingInterrupt).not.toBeNull();
    });
  });

  describe("单一 action 入口", () => {
    it("quote：记录引用的消息实体", async () => {
      const chat = startStream(openStream());
      await chat.dispatchMessageAction({ type: "quote", messageId: 101 });
      expect(chat.quotedMessage?.id).toBe(101);
    });

    it("delete：删除实体与顺序", async () => {
      const chat = startStream(openStream());
      vi.mocked(AiConversationAPI.deleteMessage).mockResolvedValue(undefined);
      await chat.dispatchMessageAction({ type: "delete", messageId: 102 });

      expect(AiConversationAPI.deleteMessage).toHaveBeenCalledWith(102);
      expect(chat.messages[102]).toBeUndefined();
      expect(chat.messageOrder).toEqual([101]);
    });

    it("delete：删除中断挂起的消息后复位全局流式态", async () => {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ confirmKind: "tool_permission" }),
      });
      handlers.onEnd?.({ messageId: 102, status: "awaiting_input" });
      expect(chat.streamStatus).toBe("interrupted");

      vi.mocked(AiConversationAPI.deleteMessage).mockResolvedValue(undefined);
      await chat.dispatchMessageAction({ type: "delete", messageId: 102 });

      expect(chat.streamStatus).toBe("idle");
      expect(chat.streamMessageId).toBeNull();
    });

    it("feedback：写入反馈并推进投影版本", async () => {
      const chat = startStream(openStream());
      vi.mocked(AiConversationAPI.submitFeedback).mockResolvedValue({
        id: 1,
        messageId: 102,
        userId: 1,
        rating: 1,
      });
      await chat.dispatchMessageAction({
        type: "feedback",
        messageId: 102,
        data: { rating: 1 },
      });

      expect(chat.feedbackByMessage[102]?.rating).toBe(1);
    });

    it("apply-suggestion：以推荐问题发送新消息", async () => {
      const chat = useChatStore();
      chat.currentConversationId = 1;
      await chat.dispatchMessageAction({
        type: "apply-suggestion",
        question: "再详细一点",
      });

      expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);
      const localUser = chat.messageOrder[chat.messageOrder.length - 2];
      expect(chat.messages[localUser].blocks[0].text).toBe("再详细一点");
    });

    it("regenerate：以原消息的父分支重新起一轮生成", async () => {
      const chat = startStream(openStream());
      handlers.onEnd?.({ messageId: 102, status: "completed" });

      await chat.dispatchMessageAction({ type: "regenerate", messageId: 102 });

      expect(AiConversationAPI.regenerate).toHaveBeenCalledWith(
        102,
        expect.anything()
      );
      expect(chat.streamStatus).toBe("running");
    });

    it("retry：与 regenerate 同路径（失败消息重试）", async () => {
      const chat = startStream(openStream());
      handlers.onEnd?.({ messageId: 102, status: "failed", error: "失败" });

      await chat.dispatchMessageAction({ type: "retry", messageId: 102 });

      expect(AiConversationAPI.regenerate).toHaveBeenCalledWith(
        102,
        expect.anything()
      );
      expect(chat.streamStatus).toBe("running");
    });

    it("speak：把消息正文交给语音播报", async () => {
      const chat = startStream(openStream());
      handlers.onBlockStart?.({
        messageId: 102,
        block: textBlock("b1", 1, ""),
      });
      handlers.onBlockDelta?.({
        messageId: 102,
        blockId: "b1",
        delta: "你好，世界",
      });
      handlers.onEnd?.({ messageId: 102, status: "completed" });
      const playSpeech = vi
        .spyOn(useVoiceStore(), "playSpeech")
        .mockResolvedValue();

      await chat.dispatchMessageAction({ type: "speak", messageId: 102 });

      expect(playSpeech).toHaveBeenCalledWith("你好，世界");
    });

    it("resume-interrupt：携带确认表单继续生成", async () => {
      const chat = startStream(openStream());
      handlers.onInterrupt?.({
        messageId: 102,
        interrupt: confirmInterrupt({ confirmKind: "tool_permission" }),
      });
      handlers.onEnd?.({ messageId: 102, status: "awaiting_input" });

      await chat.dispatchMessageAction({
        type: "resume-interrupt",
        messageId: 102,
        data: { confirm: true },
      });

      expect(resumePayload).toEqual({ confirm: true });
      expect(chat.streamStatus).toBe("running");
    });
  });
});
