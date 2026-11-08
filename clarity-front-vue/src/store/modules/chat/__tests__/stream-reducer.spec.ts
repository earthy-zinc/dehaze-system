// 流式归约单测：不引入 SDK mock / 定时器 / pinia —— 归约函数确定性，
// 仅以普通对象回放契约事件并断言实体写入结果。
import { describe, expect, it } from "vitest";
import type { Block, Message } from "clarity-sdk-js";
import {
  reduceStreamEvent,
  type StreamReducerContext,
  type StreamEvent,
} from "../stream-reducer";

/** 契约 Message 夹具（只声明归约器读取的字段） */
function entity(id: number, over: Partial<Message> = {}): Message {
  return {
    id,
    conversationId: 1,
    parentMessageId: null,
    role: "assistant",
    status: "running",
    model: null,
    error: null,
    blocks: [],
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

function block(
  id: string,
  kind: Block["kind"],
  seq: number,
  text: string | null = null
): Block {
  return { id, kind, seq, text, durationMs: null, tool: null };
}

function ctxOf(
  messages: Message[],
  state: Partial<StreamReducerContext["state"]> = {}
): StreamReducerContext {
  const table: Record<number, Message> = {};
  for (const message of messages) table[message.id] = message;
  return {
    messages: table,
    order: messages.map((message) => message.id),
    state: {
      messageId: messages[messages.length - 1]?.id ?? 0,
      localUserMessageId: null,
      started: false,
      interrupted: false,
      clearInterruptOnEvent: false,
      ...state,
    },
  };
}

function send(ctx: StreamReducerContext, event: StreamEvent) {
  return reduceStreamEvent(ctx, event);
}

describe("流式归约：message.start", () => {
  it("用服务端 user/assistant 实体替换本地占位，并原位保持顺序", () => {
    const localUser = entity(-1, { role: "user", status: "completed" });
    const localAssistant = entity(-2);
    const ctx = ctxOf(
      [entity(7, { role: "user" }), localUser, localAssistant],
      {
        messageId: -2,
        localUserMessageId: -1,
        started: false,
        interrupted: false,
        clearInterruptOnEvent: false,
      }
    );

    const effects = send(ctx, {
      kind: "start",
      data: {
        streamSessionId: "s-1",
        userMessage: entity(8, { role: "user", status: "completed" }),
        assistantMessage: entity(9),
      },
    });

    expect(effects).toEqual([{ type: "status", status: "running" }]);
    expect(ctx.order).toEqual([7, 8, 9]);
    expect(ctx.messages[-1]).toBeUndefined();
    expect(ctx.messages[-2]).toBeUndefined();
    expect(ctx.messages[8].role).toBe("user");
    expect(ctx.messages[9].status).toBe("running");
    expect(ctx.state.messageId).toBe(9);
    expect(ctx.state.localUserMessageId).toBeNull();
    expect(ctx.state.started).toBe(true);
  });

  it("重生成沿用已存在的 user 消息：不重复插入，并移除同 parent 的旧回复", () => {
    const original = entity(5);
    const placeholder = entity(-1);
    const ctx = ctxOf([entity(4, { role: "user" }), original, placeholder], {
      messageId: -1,
      localUserMessageId: null,
      started: false,
      interrupted: false,
      clearInterruptOnEvent: false,
    });

    send(ctx, {
      kind: "start",
      data: {
        streamSessionId: "s-2",
        userMessage: entity(4, { role: "user", status: "completed" }),
        assistantMessage: entity(6),
      },
    });

    // 服务端激活链只保留新回复：本地旧兄弟连同其顺序一并移除
    expect(ctx.order).toEqual([4, 6]);
    expect(ctx.messages[5]).toBeUndefined();
    expect(ctx.messages[4].role).toBe("user");
    expect(ctx.state.messageId).toBe(6);
  });

  it("乐观兄弟清理只删同 parent 的 assistant，不动 parent 及更早的消息", () => {
    const rootUser = entity(1, { role: "user" });
    const earlierReply = entity(2, { parentMessageId: 1 });
    const earlierUser = entity(3, { role: "user", parentMessageId: 2 });
    const parentUser = entity(4, { role: "user", parentMessageId: 3 });
    const oldReply = entity(5, { parentMessageId: 4 });
    const placeholder = entity(-1, { parentMessageId: 4 });
    const ctx = ctxOf(
      [rootUser, earlierReply, earlierUser, parentUser, oldReply, placeholder],
      {
        messageId: -1,
        localUserMessageId: null,
        started: false,
        interrupted: false,
        clearInterruptOnEvent: false,
      }
    );

    send(ctx, {
      kind: "start",
      data: {
        streamSessionId: "s-3",
        userMessage: entity(4, { role: "user", status: "completed" }),
        assistantMessage: entity(6, { parentMessageId: 4 }),
      },
    });

    expect(ctx.messages[5]).toBeUndefined();
    expect(ctx.order).toEqual([1, 2, 3, 4, 6]);
    expect(ctx.messages[2]).toBe(earlierReply);
    expect(ctx.messages[3].role).toBe("user");
  });

  it("重新生成较早回复：旧回复的下游 user/assistant 一并移除，不留下幽灵轮次", () => {
    // 1(user) → 2(assistant) → 3(user) → 4(assistant)：对中间的 2 重新生成
    const ctx = ctxOf(
      [
        entity(1, { role: "user" }),
        entity(2, { parentMessageId: 1 }),
        entity(3, { role: "user", parentMessageId: 2 }),
        entity(4, { parentMessageId: 3 }),
        entity(-1, { parentMessageId: 1 }),
      ],
      {
        messageId: -1,
        localUserMessageId: null,
        started: false,
        interrupted: false,
        clearInterruptOnEvent: false,
      }
    );

    send(ctx, {
      kind: "start",
      data: {
        streamSessionId: "s-4",
        userMessage: entity(1, { role: "user", status: "completed" }),
        assistantMessage: entity(5, { parentMessageId: 1 }),
      },
    });

    // 服务端新链只有 1 → 5：旧分支的下游轮次不在新链上，本地同步裁掉
    expect(ctx.order).toEqual([1, 5]);
    expect(ctx.messages[2]).toBeUndefined();
    expect(ctx.messages[3]).toBeUndefined();
    expect(ctx.messages[4]).toBeUndefined();
    expect(ctx.state.messageId).toBe(5);
  });

  it("正常发送：parent 为链尾用户消息，既有消息一条不删", () => {
    const previousReply = entity(2, { parentMessageId: 1 });
    const ctx = ctxOf(
      [
        entity(1, { role: "user" }),
        previousReply,
        entity(-1, { role: "user", status: "completed" }),
        entity(-2, { parentMessageId: -1 }),
      ],
      {
        messageId: -2,
        localUserMessageId: -1,
        started: false,
        interrupted: false,
        clearInterruptOnEvent: false,
      }
    );

    send(ctx, {
      kind: "start",
      data: {
        streamSessionId: "s-5",
        userMessage: entity(3, {
          role: "user",
          status: "completed",
          parentMessageId: 2,
        }),
        assistantMessage: entity(4, { parentMessageId: 3 }),
      },
    });

    expect(ctx.order).toEqual([1, 2, 3, 4]);
    expect(ctx.messages[1].role).toBe("user");
    expect(ctx.messages[2]).toBe(previousReply);
  });
});

describe("流式归约：块增量", () => {
  it("block.start 追加块，block.delta 按 blockId 原位追加（不按 position 覆盖）", () => {
    const message = entity(1, { blocks: [block("a", "thinking", 1)] });
    const ctx = ctxOf([message]);

    send(ctx, {
      kind: "block-start",
      data: { messageId: 1, block: block("b", "text", 1) },
    });
    send(ctx, {
      kind: "block-delta",
      data: { messageId: 1, blockId: "b", delta: "你好" },
    });
    send(ctx, {
      kind: "block-delta",
      data: { messageId: 1, blockId: "b", delta: "，世界" },
    });
    // 已经开新块后再回写旧块，仍按 blockId 定位
    send(ctx, {
      kind: "block-delta",
      data: { messageId: 1, blockId: "a", delta: "思考" },
    });

    expect(message.blocks.map((item) => item.id)).toEqual(["a", "b"]);
    expect(message.blocks[0].text).toBe("思考");
    expect(message.blocks[1].text).toBe("你好，世界");
  });

  it("delta 原位写入保持块对象与数组引用（O(1) 追加，不重建）", () => {
    const textBlock = block("a", "text", 1);
    const message = entity(1, { blocks: [textBlock] });
    const blocksRef = message.blocks;
    const ctx = ctxOf([message]);

    send(ctx, {
      kind: "block-delta",
      data: { messageId: 1, blockId: "a", delta: "x" },
    });

    expect(message.blocks[0]).toBe(textBlock);
    expect(message.blocks).toBe(blocksRef);
  });

  it("tool_use 块的参数增量不入实体，等 block.stop 的完整载荷替换", () => {
    const message = entity(1, {
      blocks: [
        {
          id: "t",
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
      ],
    });
    const ctx = ctxOf([message]);

    send(ctx, {
      kind: "block-delta",
      data: { messageId: 1, blockId: "t", delta: '{"city":' },
    });
    expect(message.blocks[0].text).toBeNull();

    const finished: Block = {
      ...block("t", "tool_use", 1),
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
    };
    send(ctx, { kind: "block-stop", data: { messageId: 1, block: finished } });

    expect(message.blocks).toHaveLength(1);
    expect(message.blocks[0]).toBe(finished);
    expect(message.blocks[0].tool?.input).toEqual({ city: "北京" });
  });

  it("block.stop 用服务端完整块替换（写入耗时）", () => {
    const message = entity(1, { blocks: [block("a", "thinking", 1, "半截")] });
    const ctx = ctxOf([message]);
    const complete = {
      ...block("a", "thinking", 1, "完整思考"),
      durationMs: 900,
    };

    send(ctx, { kind: "block-stop", data: { messageId: 1, block: complete } });

    expect(message.blocks[0]).toBe(complete);
    expect(message.blocks[0].durationMs).toBe(900);
  });

  it("块事件指向未知消息时忽略（不凭空造实体）", () => {
    const ctx = ctxOf([]);
    const effects = send(ctx, {
      kind: "block-start",
      data: { messageId: 99, block: block("a", "text", 1) },
    });
    expect(effects).toEqual([]);
    expect(ctx.messages[99]).toBeUndefined();
  });
});

describe("流式归约：计划 / 中断 / 用量 / 推荐", () => {
  it("plan.update 全量替换 plan", () => {
    const message = entity(1);
    const ctx = ctxOf([message]);
    const plan = {
      phase: "plan",
      status: "running" as const,
      tasks: [{ id: "t1", description: "步骤", dependsOn: [] }],
      revisions: [{ revisionNo: 1, reason: "初始" }],
      awaitingApproval: false,
    };

    send(ctx, { kind: "plan", data: { messageId: 1, plan } });

    expect(message.plan).toEqual(plan);
  });

  it("interrupt 写 pendingInterrupt 并进入 interrupted 态", () => {
    const message = entity(1);
    const ctx = ctxOf([message]);
    const interrupt = {
      id: "i1",
      type: "confirm" as const,
      confirmKind: "tool_permission" as const,
      createdAt: "2026-01-01T00:00:00",
      expiresAt: "2026-01-01T00:10:00",
      actionable: true,
    };

    const effects = send(ctx, {
      kind: "interrupt",
      data: { messageId: 1, interrupt },
    });

    expect(message.pendingInterrupt).toEqual(interrupt);
    expect(ctx.state.interrupted).toBe(true);
    expect(effects).toEqual([{ type: "status", status: "interrupted" }]);
  });

  it("usage / suggestions 写到实体", () => {
    const message = entity(1);
    const ctx = ctxOf([message]);
    const usage = {
      inputTokens: 10,
      outputTokens: 5,
      cachedInputTokens: 2,
      credits: 3,
    };

    send(ctx, { kind: "usage", data: { messageId: 1, usage } });
    send(ctx, {
      kind: "suggestions",
      data: { messageId: 1, suggestions: ["然后呢", "换个话题"] },
    });

    expect(message.usage).toEqual(usage);
    expect(message.suggestions).toEqual(["然后呢", "换个话题"]);
  });
});

describe("流式归约：终态", () => {
  it("message.end 写 status / error / usage，并清掉已完成轮次的中断", () => {
    const message = entity(1, {
      pendingInterrupt: {
        id: "i1",
        type: "confirm",
        createdAt: "2026-01-01T00:00:00",
        expiresAt: "2026-01-01T00:10:00",
        actionable: true,
      },
    });
    const ctx = ctxOf([message]);
    const usage = {
      inputTokens: 1,
      outputTokens: 2,
      cachedInputTokens: 0,
      credits: 3,
    };

    const effects = send(ctx, {
      kind: "end",
      data: { messageId: 1, status: "completed", usage },
    });

    expect(message.status).toBe("completed");
    expect(message.usage).toEqual(usage);
    expect(message.pendingInterrupt).toBeNull();
    expect(effects).toEqual([
      { type: "end", status: "completed", error: undefined },
    ]);
  });

  it("message.end(awaiting_input) 保留 pendingInterrupt", () => {
    const message = entity(1, {
      pendingInterrupt: {
        id: "i1",
        type: "plan_approve",
        createdAt: "2026-01-01T00:00:00",
        expiresAt: "2026-01-01T00:10:00",
        actionable: true,
      },
    });
    const ctx = ctxOf([message]);

    send(ctx, {
      kind: "end",
      data: { messageId: 1, status: "awaiting_input", error: null },
    });

    expect(message.status).toBe("awaiting_input");
    expect(message.pendingInterrupt?.id).toBe("i1");
  });

  it("error 事件只记录原因，不自行置终态", () => {
    const message = entity(1);
    const ctx = ctxOf([message]);

    const effects = send(ctx, {
      kind: "error",
      data: { code: "A0601", message: "模型未配置售价", messageId: 1 },
    });

    expect(message.error).toBe("模型未配置售价");
    expect(message.status).toBe("running");
    expect(effects).toEqual([]);
  });

  it("canceled / failed 终态由 message.end 写入", () => {
    const canceled = entity(1);
    const failed = entity(2);
    const ctx = ctxOf([canceled, failed]);

    send(ctx, { kind: "end", data: { messageId: 1, status: "canceled" } });
    send(ctx, {
      kind: "end",
      data: { messageId: 2, status: "failed", error: "boom" },
    });

    expect(canceled.status).toBe("canceled");
    expect(failed.status).toBe("failed");
    expect(failed.error).toBe("boom");
  });
});
