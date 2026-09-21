// chat store SSE 流式状态机测试：以 AiConversationAPI 为桩边界直接回放后端 wire 契约事件，
// 覆盖消息绑定/流式增量/思考块/工具块/终态/error/中断挂起/async 轮询/断线重连/超时/停止全链路。
// 时间相关行为（重连间隔/轮询/超时/rAF）统一用假时钟驱动。
import { flushPromises } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AiConversationAPI,
  type AiMessageVO,
  type InterruptEvent,
  type MessageStreamHandlers,
} from "dehaze-sdk-js";
import { resolveConfirmKind, useChatStore } from "@/store/modules/chat";
import { MESSAGES_PAGE_SIZE } from "@/store/modules/chat/shared";

let handlers: MessageStreamHandlers;
let controller: AbortController;
/** resume 续流的回调与载荷捕获（断言传给 SDK 的载荷形状） */
let resumeHandlers: MessageStreamHandlers;
let resumePayload: unknown;

vi.mock("dehaze-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("dehaze-sdk-js")>();
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
      stopMessage: vi.fn(),
      getMessageDetail: vi.fn(),
      reconnectStream: vi.fn(),
      getMessages: vi.fn(),
      markConversationRead: vi.fn().mockResolvedValue({}),
    },
  };
});

/** 构造 confirm 中断（用例需注入契约外对抗值，故按 Record 构造后断言） */
function confirmInterrupt(data: Record<string, unknown>): InterruptEvent {
  return { type: "confirm", data } as unknown as InterruptEvent;
}

/** 构造一页游标消息（后端按 id 倒序口径：入参按 id 降序给出） */
function messagePage(
  ids: number[],
  hasMore = false
): {
  list: AiMessageVO[];
  total: number;
  hasMore: boolean;
} {
  return {
    list: ids.map((id) => ({
      id,
      conversationId: 1,
      role: "assistant",
      status: 2,
      content: `m${id}`,
      createTime: "2026-01-01T00:00:00Z",
    })),
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

/** 手控 Promise：制造"请求未返回期间切换会话"的竞态，用于验证过期响应被丢弃 */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const START = {
  messageId: 4,
  conversationId: 1,
  model: "qwen3-0.6b",
  streamSessionId: "stream-1",
};

function openStream() {
  const chat = useChatStore();
  chat.currentConversationId = 1;
  chat.sendMessage("你好");
  return chat;
}

function messageOf(chat: ReturnType<typeof useChatStore>, id = 4) {
  return chat.messages.find((m) => m.id === id)!;
}

function textDelta(text: string) {
  return handlers.onContentBlockDelta!({
    index: 0,
    delta: { type: "text_delta", text },
  });
}

const END_USAGE = {
  inputTokens: 10,
  outputTokens: 5,
  cachedInputTokens: 2,
  credits: 3,
};

/** 刷新流式缓冲：假时钟下 rAF 按 16ms 帧调度，推进一整帧确保回调执行 */
async function flushFrames() {
  await vi.advanceTimersByTimeAsync(20);
}

describe("chat store：SSE 流式状态机", () => {
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

  describe("消息绑定", () => {
    it("onStart 将服务端 messageId 绑定到本地占位消息", async () => {
      const chat = openStream();
      handlers.onStart?.(START);

      expect(messageOf(chat).id).toBe(4);
      expect(chat.streamingMessageId).toBe(4);
      expect(chat.isStreaming).toBe(true);
    });

    it("本地占位 ID 不因毫秒跳变碰撞（用户消息与 assistant 占位各拿唯一 ID）", async () => {
      // 每次取时钟 +1，强制复现 -Date.now()-1 与 -Date.now() 跨毫秒相等的历史时序
      let now = 1_000_000;
      vi.spyOn(Date, "now").mockImplementation(() => now++);

      const chat = openStream();
      const user = chat.messages.find((m) => m.role === "user")!;
      const placeholder = chat.messages.find(
        (m) => m.role === "assistant" && m.status === 1
      )!;
      expect(placeholder).toBeTruthy();
      expect(user.id).not.toBe(placeholder.id);

      handlers.onStart?.(START);
      await flushFrames();
      expect(messageOf(chat).role).toBe("assistant");
      expect(user.content).toBe("你好");
    });

    it("流式进行中重复发送被拒绝（同会话仅允许一个流）", () => {
      const chat = openStream();
      chat.sendMessage("第二条");
      expect(AiConversationAPI.sendMessage).toHaveBeenCalledTimes(1);
    });
  });

  describe("流式增量", () => {
    it("text 增量经 rAF 批量写入 assistant 消息 content", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      textDelta("你好");
      textDelta("，世界");
      await flushFrames();

      expect(messageOf(chat).content).toBe("你好，世界");
    });

    it("多段思考按段累积，正文与思考互不混写", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 1, type: "thinking" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "第一段" },
      });
      handlers.onContentBlockStop?.({ index: 1 });
      handlers.onContentBlockStart?.({ index: 0, type: "text" });
      textDelta("回复正文");
      handlers.onContentBlockStop?.({ index: 0 });
      handlers.onContentBlockStart?.({ index: 1, type: "thinking" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "第二段" },
      });
      await flushFrames();

      const state = chat.thinkingByMessage[4];
      expect(state.segments.map((s) => s.text)).toEqual(["第一段", "第二段"]);
      expect(state.segments.map((s) => s.closed)).toEqual([true, false]);
      expect(state.endAt).not.toBeNull();
      expect(messageOf(chat).content).toBe("回复正文");
    });

    it("同一思考段的增量跨 rAF 帧持续累积为一段（历史碎片回归）", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 1, type: "thinking" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "我们需要" },
      });
      await flushFrames();
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "回答用户" },
      });
      await flushFrames();
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "的问题" },
      });
      await flushFrames();

      const state = chat.thinkingByMessage[4];
      expect(state.segments).toHaveLength(1);
      expect(state.segments[0].text).toBe("我们需要回答用户的问题");
    });

    it("message.end 收尾闭合未关闭思考段并定格计时", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 1, type: "thinking" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "thinking_delta", thinking: "未收尾的思考" },
      });
      await flushFrames();
      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });

      const state = chat.thinkingByMessage[4];
      expect(state.segments).toHaveLength(1);
      expect(state.segments[0]).toEqual({
        text: "未收尾的思考",
        closed: true,
      });
      expect(state.endAt).not.toBeNull();
    });

    it("无 content_block.start 的残留思考增量也能累积", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockDelta?.({
        index: 0,
        delta: { type: "thinking_delta", thinking: "残留思考" },
      });
      await flushFrames();

      expect(chat.thinkingByMessage[4].segments).toEqual([
        { text: "残留思考", closed: false },
      ]);
    });
  });

  describe("工具调用块", () => {
    it("input_json_delta 组装并解析为对象，onEnd 挂载到消息", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 1, type: "tool_use" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: {
          type: "input_json_delta",
          name: "get_weather",
          partialJson: '{"city":',
        },
      });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "input_json_delta", partialJson: '"北京"}' },
      });
      handlers.onContentBlockStop?.({ index: 1 });
      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });

      expect(chat.toolCallsByMessage[4]).toEqual([
        { name: "get_weather", arguments: { city: "北京" } },
      ]);
      expect(messageOf(chat).toolCalls).toEqual([
        { name: "get_weather", arguments: { city: "北京" } },
      ]);
    });

    it("非法 JSON 参数保留原始字符串", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 1, type: "tool_use" });
      handlers.onContentBlockDelta?.({
        index: 1,
        delta: { type: "input_json_delta", partialJson: "not-json" },
      });
      handlers.onContentBlockStop?.({ index: 1 });

      expect(chat.toolCallsByMessage[4]).toEqual([
        { name: undefined, arguments: "not-json" },
      ]);
    });
  });

  describe("事件透传", () => {
    it("thought 按 position 排序，同位更新覆盖", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onThought?.({ position: 2, thought: "第二步", status: 1 });
      handlers.onThought?.({ position: 1, thought: "第一步", status: 1 });
      handlers.onThought?.({ position: 1, thought: "第一步(修订)", status: 1 });

      expect(chat.thoughtsByMessage[4].map((t) => t.thought)).toEqual([
        "第一步(修订)",
        "第二步",
      ]);
    });

    it("suggestions 展开为问题数组", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onSuggestions?.({
        questions: [{ question: "然后呢" }, { question: "换个话题" }],
      });

      expect(chat.suggestions).toEqual(["然后呢", "换个话题"]);
    });
  });

  describe("message.end 终态", () => {
    it("end_turn：消息置完成态并写入用量", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });

      const msg = messageOf(chat);
      expect(msg.status).toBe(2);
      expect(msg.inputTokens).toBe(10);
      expect(msg.outputTokens).toBe(5);
      expect(msg.cachedInputTokens).toBe(2);
      expect(msg.credits).toBe(3);
      expect(chat.isStreaming).toBe(false);
    });

    it("canceled 置取消态、error 置失败态", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onEnd?.({ stopReason: "canceled", usage: END_USAGE });
      expect(messageOf(chat).status).toBe(4);

      const chat2 = openStream();
      // 第二个 store 会话：新流 id 不同，单独驱动
      handlers.onStart?.({ ...START, messageId: 5 });
      handlers.onEnd?.({ stopReason: "error", usage: END_USAGE });
      expect(messageOf(chat2, 5).status).toBe(3);
    });
  });

  describe("error 事件（无 message.end 收尾）", () => {
    it("消息置失败态并渲染已收到的内容（线上 A0601 场景回放）", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onContentBlockStart?.({ index: 0, type: "text" });
      for (const text of ["我", "在这里", "，", "随时", "为您", "服务", "。"]) {
        textDelta(text);
      }
      handlers.onContentBlockStop?.({ index: 0 });
      handlers.onError?.({
        code: "A0601",
        message: "模型 qwen3-0.6b 未配置用户售价",
      });
      handlers.onClose?.();
      await flushFrames();

      const msg = messageOf(chat);
      expect(msg.content).toBe("我在这里，随时为您服务。");
      expect(msg.status).toBe(3);
      expect((msg as AiMessageVO & { error?: string }).error).toBe(
        "模型 qwen3-0.6b 未配置用户售价"
      );
      expect(chat.streamingMessageId).toBeNull();
      expect(chat.isStreaming).toBe(false);
    });
  });

  describe("中断挂起", () => {
    it("confirm 中断：挂起消息保持生成中，等待 resume", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onInterrupt?.({
        type: "confirm",
        data: {
          recommendation: {
            recommendationId: 1,
            algorithmId: 2,
            algorithmName: "去雾算法",
            reason: "需要确认",
          },
        },
      });

      expect(chat.interrupts).toHaveLength(1);
      expect(chat.interruptedMessageId).toBe(4);
      expect(chat.streamingMessageId).toBeNull();
      expect(chat.isStreaming).toBe(true);

      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });
      expect(messageOf(chat).status).toBe(1);
    });

    it("async_wait 中断：message.end 后轮询消息终态并清理中断", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      vi.mocked(AiConversationAPI.getMessageDetail).mockResolvedValue({
        id: 4,
        status: 2,
        content: "异步完成",
      } as AiMessageVO);
      handlers.onInterrupt?.({
        type: "async_wait",
        data: { taskId: "t1" },
      });
      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });

      await vi.advanceTimersByTimeAsync(5_000);

      expect(AiConversationAPI.getMessageDetail).toHaveBeenCalledWith(4);
      const msg = messageOf(chat);
      expect(msg.status).toBe(2);
      expect(msg.content).toBe("异步完成");
      expect(chat.interrupts).toHaveLength(0);
      expect(chat.interruptedMessageId).toBeNull();
    });
  });

  describe("断线重连", () => {
    it("携带 lastEventId 在 3 秒后自动重连", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      textDelta("部分内容");
      handlers.onEventId?.("3");

      handlers.onNetworkError?.(new Error("断开"));
      await vi.advanceTimersByTimeAsync(3_000);

      expect(AiConversationAPI.reconnectStream).toHaveBeenCalledWith(
        1,
        "stream-1",
        "3",
        expect.anything()
      );
      expect(messageOf(chat).status).toBe(1);
    });

    it("重连 3 次耗尽后判失败", async () => {
      const chat = openStream();
      handlers.onStart?.(START);

      for (let i = 0; i < 3; i++) {
        handlers.onNetworkError?.(new Error("断开"));
        await vi.advanceTimersByTimeAsync(3_000);
      }
      handlers.onNetworkError?.(new Error("断开"));

      const msg = messageOf(chat) as AiMessageVO & { error?: string };
      expect(msg.status).toBe(3);
      expect(msg.error).toContain("网络连接中断");
    });

    it("流已达终态后 onClose 不再重连", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });
      handlers.onClose?.();
      await vi.advanceTimersByTimeAsync(5_000);

      expect(AiConversationAPI.reconnectStream).not.toHaveBeenCalled();
    });
  });

  describe("流式超时", () => {
    it("120 秒无新内容判超时失败并中断连接", async () => {
      const chat = openStream();
      handlers.onStart?.(START);

      await vi.advanceTimersByTimeAsync(150_000);

      const msg = messageOf(chat) as AiMessageVO & { error?: string };
      expect(msg.status).toBe(3);
      expect(msg.error).toContain("流式输出超时");
      expect(controller.signal.aborted).toBe(true);
    });

    it("收到增量刷新计时；ping 不刷新", async () => {
      const chat = openStream();
      handlers.onStart?.(START);

      await vi.advanceTimersByTimeAsync(100_000);
      textDelta("续命");
      await vi.advanceTimersByTimeAsync(100_000);
      expect(messageOf(chat).status).toBe(1);

      handlers.onPing?.();
      await vi.advanceTimersByTimeAsync(30_000);
      expect(messageOf(chat).status).toBe(3);
    });
  });

  describe("停止生成", () => {
    it("message.start 前停止：本地取消且不调停止接口", async () => {
      const chat = openStream();

      await chat.stopStreaming();

      expect(AiConversationAPI.stopMessage).not.toHaveBeenCalled();
      const placeholder = chat.messages.find(
        (m) => m.role === "assistant" && m.status === 4
      );
      expect(placeholder).toBeTruthy();
      expect(controller.signal.aborted).toBe(true);
    });

    it("message.start 后停止：调用停止接口并落取消态", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      vi.mocked(AiConversationAPI.stopMessage).mockResolvedValue({
        id: 4,
        status: 4,
      } as AiMessageVO);

      await chat.stopStreaming();

      expect(AiConversationAPI.stopMessage).toHaveBeenCalledWith(4);
      expect(messageOf(chat).status).toBe(4);
    });
  });

  describe("消息列表覆盖保护", () => {
    it("流式进行中 fetchMessages 保留流式对象引用，onEnd 仍能落到同一消息", async () => {
      const chat = openStream();
      handlers.onStart?.(START);
      const live = messageOf(chat);

      vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
        list: [
          {
            id: 4,
            conversationId: 1,
            role: "assistant",
            status: 1,
            content: "服务端快照",
            createTime: "2026-09-13 08:00:00",
          },
        ],
        total: 1,
        hasMore: false,
      });
      await chat.fetchMessages(1);

      const after = chat.messages.find((m) => m.id === 4)!;
      expect(after).toBe(live);

      handlers.onEnd?.({ stopReason: "stop", usage: END_USAGE });
      expect(messageOf(chat).status).toBe(2);
      expect(chat.isStreaming).toBe(false);
    });
  });

  describe("plan 事件消费", () => {
    it("生成：plan 事件映射为计划状态", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onPlan?.({
        tasks: [
          {
            id: "t1",
            description: "步骤1",
            dependsOn: ["t0"],
            status: "pending",
            paradigm: "react",
          },
        ],
        status: "pending",
        revisions: [],
        phase: "plan",
      });

      expect(chat.planByMessage[4]).toEqual({
        messageId: 4,
        phase: "plan",
        tasks: [
          {
            id: "t1",
            description: "步骤1",
            dependsOn: ["t0"],
            status: "pending",
            paradigm: "react",
          },
        ],
        status: "pending",
        revisions: [],
        awaitingApproval: false,
      });
    });

    it("更新：后续 plan 事件覆盖任务列表与状态", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onPlan?.({
        tasks: [{ id: "t1", description: "A", dependsOn: [] }],
        status: "pending",
        revisions: [],
      });
      handlers.onPlan?.({
        tasks: [{ id: "t1", description: "A", dependsOn: [], status: "done" }],
        status: "done",
        phase: "revised",
        revisions: [],
      });

      expect(chat.planByMessage[4].tasks).toEqual([
        { id: "t1", description: "A", dependsOn: [], status: "completed" },
      ]);
      expect(chat.planByMessage[4].status).toBe("completed");
      expect(chat.planByMessage[4].phase).toBe("revised");
    });

    it("重规划：revisions 累积去重并保留修订说明", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onPlan?.({
        tasks: [],
        status: "executing",
        revisions: [{ revisionNo: 1, reason: "B", changedTaskIds: ["B2"] }],
        phase: "plan",
      });
      // 后端每次下发全量 revisions：第二次含 revisionNo=1、2，去重后不应重复 1
      handlers.onPlan?.({
        tasks: [],
        status: "revised",
        revisions: [
          { revisionNo: 1, reason: "B", changedTaskIds: ["B2"] },
          { revisionNo: 2, reason: "C", changedTaskIds: ["C2"] },
        ],
        phase: "revised",
      });

      expect(chat.planByMessage[4].revisions).toEqual([
        { revisionNo: 1, reason: "B" },
        { revisionNo: 2, reason: "C" },
      ]);
    });

    it("对抗：revisions 乱序到达按 revisionNo 升序重排", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onPlan?.({
        tasks: [],
        revisions: [{ revisionNo: 2, reason: "后", changedTaskIds: [] }],
      });
      handlers.onPlan?.({
        tasks: [],
        revisions: [{ revisionNo: 1, reason: "先", changedTaskIds: [] }],
      });

      expect(
        chat.planByMessage[4].revisions.map((item) => item.revisionNo)
      ).toEqual([1, 2]);
      expect(
        chat.planByMessage[4].revisions.map((item) => item.reason)
      ).toEqual(["先", "后"]);
    });

    it("对抗：plan.tasks 为空 / dependsOn 缺失均容错", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onPlan?.({ tasks: [], revisions: [] });
      expect(chat.planByMessage[4].tasks).toEqual([]);

      handlers.onPlan?.({
        tasks: [{ id: "t1", description: "无依赖" }],
        revisions: [],
      });
      expect(chat.planByMessage[4].tasks).toEqual([
        { id: "t1", description: "无依赖" },
      ]);
    });

    it("plan_approve 中断置待批准，恢复成功后清零", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onInterrupt?.({
        type: "plan_approve",
        data: {
          plan: {
            tasks: [{ id: "t1", description: "步骤", dependsOn: [] }],
            status: "pending",
            revisions: [],
          },
        },
      });
      expect(chat.planByMessage[4].awaitingApproval).toBe(true);

      chat.resumeInterrupt(4, { confirm: true });
      resumeHandlers.onStart?.({ ...START, messageId: 4 });
      expect(chat.planByMessage[4].awaitingApproval).toBe(false);
    });
  });

  describe("confirmKind 归一", () => {
    it.each<[string, Record<string, unknown>]>([
      ["algorithm_recommend", { confirmKind: "algorithm_recommend" }],
      ["tool_permission", { confirmKind: "tool_permission" }],
      ["dangerous_op", { confirmKind: "dangerous_op" }],
      [
        "write_conflict",
        { confirmKind: "dangerous_op", action: "write_conflict" },
      ],
    ])("confirm 子类型 %s 归一", (expected, data) => {
      expect(resolveConfirmKind(confirmInterrupt(data))).toBe(expected);
    });

    it("非 confirm 中断无子类型", () => {
      expect(resolveConfirmKind({ type: "quota", data: {} })).toBeUndefined();
    });

    it("对抗：未知 / 缺失子类型显式报错（不静默兜底）", () => {
      expect(() =>
        resolveConfirmKind(confirmInterrupt({ confirmKind: "bogus" }))
      ).toThrow(/未知/);
      expect(() => resolveConfirmKind(confirmInterrupt({}))).toThrow(/未知/);
    });
  });

  describe("resumeInterrupt 子类型路由", () => {
    function interruptOf(data: Record<string, unknown>) {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onInterrupt?.(confirmInterrupt(data));
      return chat;
    }

    it("algorithm_recommend：{confirm, params:{algorithmId}}（取调用方选择）", () => {
      const chat = interruptOf({
        confirmKind: "algorithm_recommend",
        recommendation: {
          recommendationId: 1,
          algorithmId: 7,
          algorithmName: "去雾",
          reason: "最佳",
        },
      });
      chat.resumeInterrupt(4, { confirm: true, params: { algorithmId: 8 } });

      expect(resumePayload).toEqual({
        confirm: true,
        params: { algorithmId: 8 },
      });
    });

    it("algorithm_recommend：未指定则回退推荐算法 id", () => {
      const chat = interruptOf({
        confirmKind: "algorithm_recommend",
        recommendation: {
          recommendationId: 1,
          algorithmId: 7,
          algorithmName: "去雾",
          reason: "最佳",
        },
      });
      chat.resumeInterrupt(4, { confirm: true });

      expect(resumePayload).toEqual({
        confirm: true,
        params: { algorithmId: 7 },
      });
    });

    it("plan_approve：透传 planEdit（wire 与 VM 同形）", () => {
      const chat = openStream();
      handlers.onStart?.(START);
      handlers.onInterrupt?.({
        type: "plan_approve",
        data: {
          plan: {
            tasks: [{ id: "t1", description: "x", dependsOn: [] }],
            revisions: [],
          },
        },
      });
      chat.resumeInterrupt(4, {
        planEdit: { remove: ["t1"], reorder: ["t2", "t3"] },
      });

      expect(resumePayload).toEqual({
        planEdit: { remove: ["t1"], reorder: ["t2", "t3"] },
      });
    });

    it("tool_permission：{confirm}", () => {
      const chat = interruptOf({
        confirmKind: "tool_permission",
        tool: "write_file",
        reason: "权限不足",
      });
      chat.resumeInterrupt(4, { confirm: true });
      expect(resumePayload).toEqual({ confirm: true });
    });

    it("write_conflict（dangerous_op 写冲突）：{confirm}", () => {
      const chat = interruptOf({
        confirmKind: "dangerous_op",
        action: "write_conflict",
        tool: "write_file",
        resource: "/a.txt",
        previousWriter: "子AgentA",
      });
      chat.resumeInterrupt(4, { confirm: false });
      expect(resumePayload).toEqual({ confirm: false });
    });

    it("对抗：未知 confirmKind 路由时显式报错，不清中断点", () => {
      const chat = interruptOf({ confirmKind: "bogus" });
      expect(() => chat.resumeInterrupt(4, { confirm: true })).toThrow(/未知/);
      expect(chat.interrupts).toHaveLength(1);
    });

    it("中断点仅恢复成功（首个事件到达）后清理，未达前保留", () => {
      const chat = interruptOf({ confirmKind: "tool_permission" });
      expect(chat.interrupts).toHaveLength(1);
      expect(chat.interruptedMessageId).toBe(4);

      chat.resumeInterrupt(4, { confirm: true });
      // 尚未收到续流事件：中断点保留
      expect(chat.interrupts).toHaveLength(1);
      expect(chat.interruptedMessageId).toBe(4);

      resumeHandlers.onStart?.({ ...START, messageId: 4 });
      expect(chat.interrupts).toHaveLength(0);
      expect(chat.interruptedMessageId).toBeNull();
    });

    it("恢复失败（会话忙）：保留中断点且不重复发起", () => {
      const chat = interruptOf({ confirmKind: "tool_permission" });
      chat.resumeInterrupt(4, { confirm: true });
      expect(AiConversationAPI.resumeMessage).toHaveBeenCalledTimes(1);

      chat.resumeInterrupt(4, { confirm: true });
      expect(AiConversationAPI.resumeMessage).toHaveBeenCalledTimes(1);
      expect(chat.interrupts).toHaveLength(1);
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
      expect(chat.messages.map((m) => m.id)).toEqual(ascIds(51, 100));
      expect(chat.messagesHasMore).toBe(true);

      await chat.loadMoreMessages();
      // before = 当前列表最早一条（id=51）
      expect(AiConversationAPI.getMessages).toHaveBeenNthCalledWith(2, 1, {
        before: 51,
        limit: MESSAGES_PAGE_SIZE,
      });
      expect(chat.messages.map((m) => m.id)).toEqual([
        ...ascIds(1, 50),
        ...ascIds(51, 100),
      ]);
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
        // 更早页与首页在 51..55 区间重叠
        .mockResolvedValueOnce(messagePage(descIds(55, 1), false));

      await chat.fetchMessages(1);
      await chat.loadMoreMessages();

      const ids = chat.messages.map((m) => m.id);
      expect(ids).toEqual(ascIds(1, 100));
      expect(new Set(ids).size).toBe(ids.length);
    });

    it("并发插入：加载更早页进行中新增最新消息，合并后仍按 id 去重、不重不漏", async () => {
      const chat = useChatStore();
      const pending = deferred<ReturnType<typeof messagePage>>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockReturnValueOnce(pending.promise);

      await chat.fetchMessages(1);
      const loading = chat.loadMoreMessages();

      // 加载更早历史期间，发送/流式并发插入一条更新的消息（id=101）
      chat.messages.push({
        id: 101,
        conversationId: 1,
        role: "assistant",
        status: 2,
        content: "新消息",
        createTime: "2026-01-01T00:00:01Z",
      });

      pending.resolve(messagePage(descIds(50, 1), false));
      await loading;

      const ids = chat.messages.map((m) => m.id);
      expect(ids).toEqual([...ascIds(1, 100), 101]);
      expect(new Set(ids).size).toBe(ids.length);
    });

    it("竞态：loadMoreMessages 进行中切换会话，过期响应被丢弃不污染新会话", async () => {
      const chat = useChatStore();
      const stalePage = deferred<ReturnType<typeof messagePage>>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockReturnValueOnce(stalePage.promise)
        .mockResolvedValueOnce(messagePage(descIds(200, 151), false));

      await chat.fetchMessages(1);
      expect(chat.messages.map((m) => m.id)).toEqual(ascIds(51, 100));

      // 会话1向上加载进行中（响应挂起）→ 切到会话2
      const pending = chat.loadMoreMessages();
      await chat.fetchMessages(2);
      const ids2 = chat.messages.map((m) => m.id);
      expect(chat.currentConversationId).toBe(2);
      expect(ids2).toEqual(ascIds(151, 200));

      // 会话1的历史响应此刻才到达：应被丢弃，不得前置拼入会话2
      stalePage.resolve(messagePage(descIds(50, 1), false));
      await pending;

      expect(chat.messages.map((m) => m.id)).toEqual(ids2);
    });

    it("竞态：fetchMessages(A) 未返回即切到 B，A 的过期响应不覆盖 B", async () => {
      const chat = useChatStore();
      const stalePage = deferred<ReturnType<typeof messagePage>>();
      vi.mocked(AiConversationAPI.getMessages)
        .mockReturnValueOnce(stalePage.promise)
        .mockResolvedValueOnce(messagePage(descIds(200, 151), false));

      const pending = chat.fetchMessages(1);
      await chat.fetchMessages(2);
      const ids2 = chat.messages.map((m) => m.id);
      expect(ids2).toEqual(ascIds(151, 200));

      stalePage.resolve(messagePage(descIds(100, 51), true));
      await pending;

      expect(chat.messages.map((m) => m.id)).toEqual(ids2);
    });

    it("对抗：空页返回（到底）不改变列表且 hasMore=false", async () => {
      const chat = useChatStore();
      vi.mocked(AiConversationAPI.getMessages)
        .mockResolvedValueOnce(messagePage(descIds(100, 51), true))
        .mockResolvedValueOnce(messagePage([], false));

      await chat.fetchMessages(1);
      const before = chat.messages.map((m) => m.id);

      await chat.loadMoreMessages();
      expect(chat.messages.map((m) => m.id)).toEqual(before);
      expect(chat.messagesHasMore).toBe(false);
    });
  });
});
