// 流式传输生命周期单测：断线重连后的 controller 归属，以及终态后迟到事件的守卫。
// 以 AiConversationAPI 为桩边界驱动真实 store（sendMessage / stopStreaming / 超时监控）。
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  AiConversationAPI,
  type Block,
  type Message,
  type MessageStreamHandlers,
} from "clarity-sdk-js";
import { useChatStore } from "@/store/modules/chat";

let handlers: MessageStreamHandlers;
let controller: AbortController;

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
      stopMessage: vi.fn(),
      reconnectStream: vi.fn(),
      markConversationRead: vi.fn().mockResolvedValue({}),
    },
  };
});

const EMPTY_USAGE = {
  inputTokens: 0,
  outputTokens: 0,
  cachedInputTokens: 0,
  credits: 0,
};

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

function startOf() {
  return {
    streamSessionId: "stream-1",
    userMessage: message(101, {
      role: "user",
      status: "completed",
      blocks: [textBlock("u1", 1, "你好")],
    }),
    assistantMessage: message(102, { status: "running" }),
  };
}

function openStream() {
  const chat = useChatStore();
  chat.currentConversationId = 1;
  chat.sendMessage("你好");
  return chat;
}

function startStream(chat: ReturnType<typeof useChatStore>) {
  handlers.onStart?.(startOf());
  return chat;
}

describe("chat 流式传输：重连 controller 与终态守卫", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    handlers = undefined!;
    controller = undefined!;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("重连成功后 stopStreaming 中止新连接的 controller", async () => {
    const chat = startStream(openStream());
    const reconnectController = new AbortController();
    vi.mocked(AiConversationAPI.reconnectStream).mockReturnValue(
      reconnectController
    );
    vi.mocked(AiConversationAPI.stopMessage).mockResolvedValue(
      message(102, { status: "canceled" })
    );

    handlers.onNetworkError?.(new Error("断开"));
    await vi.advanceTimersByTimeAsync(3_000);

    expect(AiConversationAPI.reconnectStream).toHaveBeenCalledTimes(1);
    expect(reconnectController.signal.aborted).toBe(false);

    await chat.stopStreaming();

    expect(AiConversationAPI.stopMessage).toHaveBeenCalledWith(102);
    expect(reconnectController.signal.aborted).toBe(true);
    // 旧连接的 controller 不再是会话持有者：替换生效才让它保持未中止
    expect(controller.signal.aborted).toBe(false);
    expect(chat.streamStatus).toBe("idle");
  });

  it("超时终态后迟到的 block.delta 不写实体、不改变流状态", async () => {
    const chat = startStream(openStream());
    handlers.onBlockStart?.({ messageId: 102, block: textBlock("b1", 1, "") });
    handlers.onBlockDelta?.({ messageId: 102, blockId: "b1", delta: "半截" });

    await vi.advanceTimersByTimeAsync(150_000);

    expect(chat.messages[102].status).toBe("failed");
    expect(chat.streamStatus).toBe("idle");

    handlers.onBlockDelta?.({ messageId: 102, blockId: "b1", delta: "迟到" });

    expect(chat.messages[102].blocks).toEqual([textBlock("b1", 1, "半截")]);
    expect(chat.messages[102].status).toBe("failed");
    expect(chat.streamStatus).toBe("idle");
  });

  it("终态后迟到的 block.start 不追加块、不把流状态拉回 running", async () => {
    const chat = startStream(openStream());
    handlers.onEnd?.({ messageId: 102, status: "completed" });
    expect(chat.streamStatus).toBe("idle");

    handlers.onBlockStart?.({
      messageId: 102,
      block: textBlock("b9", 1, "迟到"),
    });

    expect(chat.messages[102].blocks).toEqual([]);
    expect(chat.messages[102].status).toBe("completed");
    expect(chat.streamStatus).toBe("idle");
  });

  it("终态后的 suggestions 仍写入（唯一放行的迟到事件）", () => {
    const chat = startStream(openStream());
    handlers.onEnd?.({ messageId: 102, status: "completed" });

    handlers.onSuggestions?.({ messageId: 102, suggestions: ["继续追问"] });

    expect(chat.messages[102].suggestions).toEqual(["继续追问"]);
    expect(chat.streamStatus).toBe("idle");
  });
});
