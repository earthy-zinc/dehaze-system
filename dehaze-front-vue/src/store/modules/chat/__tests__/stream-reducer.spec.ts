// 流式归约器独立单测：不引入 SDK mock / 定时器 / pinia —— 归约函数为确定性纯函数，
// 仅以普通对象回放 SSE 事件即可断言状态迁移与副作用描述，时间由 now 注入。
import { describe, expect, it } from "vitest";
import { reduceStreamState, type StreamEvent } from "../stream-reducer";
import type { StreamViewState } from "../shared";

const NOW = 1_700_000_000_000;

function baseState(over: Partial<StreamViewState> = {}): StreamViewState {
  return {
    messageId: 4,
    textBuffer: "",
    thinking: null,
    thinkingBuffer: "",
    toolBlocks: new Map(),
    toolCalls: [],
    thoughts: [],
    interrupts: [],
    suggestions: [],
    streamingMessageId: 4,
    interruptedMessageId: null,
    ...over,
  };
}

const reduce = (state: StreamViewState, event: StreamEvent) =>
  reduceStreamState(state, event, NOW);

describe("reduceStreamState（纯归约）", () => {
  it("start：绑定 messageId 并同步 streamingMessageId；clearInterruptOnStart 清中断点与待批准", () => {
    const state = baseState({
      clearInterruptOnStart: true,
      interrupts: [{ type: "confirm", data: {} }],
      interruptedMessageId: 4,
      plan: {
        messageId: 4,
        tasks: [],
        status: "pending",
        revisions: [],
        awaitingApproval: true,
      },
    });
    const { state: next, effects } = reduce(state, {
      kind: "start",
      data: {
        messageId: 9,
        conversationId: 1,
        model: "m",
        streamSessionId: "s1",
      },
    });

    expect(next.messageId).toBe(9);
    expect(next.streamingMessageId).toBe(9);
    expect(next.interrupts).toEqual([]);
    expect(next.interruptedMessageId).toBeNull();
    expect(next.plan?.awaitingApproval).toBe(false);
    expect(effects).toEqual([
      { type: "bind", from: 4 },
      { type: "streaming-id" },
      { type: "clear-interrupts" },
      { type: "plan" },
    ]);
    // 入参未被改写
    expect(state.interrupts).toHaveLength(1);
    expect(state.plan?.awaitingApproval).toBe(true);
  });

  it("delta：text 增量仅累加缓冲并产出 flush（逐 delta 无额外状态迁移）", () => {
    const { state, effects } = reduce(baseState(), {
      kind: "delta",
      data: { index: 0, delta: { type: "text_delta", text: "你好" } },
    });
    expect(state.textBuffer).toBe("你好");
    expect(state.thinking).toBeNull();
    expect(effects).toEqual([{ type: "flush" }]);
  });

  it("思考块：start 开户 → delta 缓冲 → stop 并入并闭合，时间定格为注入的 now", () => {
    let result = reduce(baseState(), {
      kind: "block-start",
      data: { index: 1, type: "thinking" },
    });
    expect(result.effects).toEqual([{ type: "writing" }]);
    expect(result.state.thinking?.startAt).toBe(NOW);
    expect(result.state.thinking?.segments).toEqual([
      { text: "", closed: false },
    ]);

    result = reduce(result.state, {
      kind: "delta",
      data: { index: 1, delta: { type: "thinking_delta", thinking: "想" } },
    });
    expect(result.state.thinkingBuffer).toBe("想");
    // 未刷入前思考段文本仍为空（rAF 批量语义）
    expect(result.state.thinking?.segments[0].text).toBe("");

    result = reduce(result.state, { kind: "block-stop", data: { index: 1 } });
    expect(result.effects).toEqual([{ type: "writing" }]);
    expect(result.state.thinking?.segments).toEqual([
      { text: "想", closed: true },
    ]);
    expect(result.state.thinking?.endAt).toBe(NOW);
    expect(result.state.thinkingBuffer).toBe("");
  });

  it("工具块：input_json_delta 组装解析，block-stop 产出 toolCalls 并清草稿", () => {
    let result = reduce(baseState(), {
      kind: "block-start",
      data: { index: 2, type: "tool_use" },
    });
    result = reduce(result.state, {
      kind: "delta",
      data: {
        index: 2,
        delta: {
          type: "input_json_delta",
          name: "get_weather",
          partialJson: '{"city":',
        },
      },
    });
    result = reduce(result.state, {
      kind: "delta",
      data: {
        index: 2,
        delta: { type: "input_json_delta", partialJson: '"北京"}' },
      },
    });
    result = reduce(result.state, { kind: "block-stop", data: { index: 2 } });

    expect(result.effects).toEqual([{ type: "tool-calls" }]);
    expect(result.state.toolCalls).toEqual([
      { name: "get_weather", arguments: { city: "北京" } },
    ]);
    expect(result.state.toolBlocks.size).toBe(0);
  });

  it("thought：按 position 升序、同位覆盖", () => {
    let state = baseState();
    state = reduce(state, {
      kind: "thought",
      data: { position: 2, thought: "二", status: 1 },
    }).state;
    state = reduce(state, {
      kind: "thought",
      data: { position: 1, thought: "一", status: 1 },
    }).state;
    state = reduce(state, {
      kind: "thought",
      data: { position: 1, thought: "一改", status: 1 },
    }).state;
    expect(state.thoughts.map((item) => item.thought)).toEqual(["一改", "二"]);
  });

  it("plan_approve 中断：计划按统一形状归一、置待批准、挂起消息", () => {
    const { state, effects } = reduce(baseState(), {
      kind: "interrupt",
      data: {
        type: "plan_approve",
        data: {
          plan: {
            tasks: [{ id: "t1", description: "步骤", dependsOn: ["t0"] }],
            revisions: [],
          },
        },
      },
    });
    expect(state.interruptedMessageId).toBe(4);
    expect(state.streamingMessageId).toBeNull();
    expect(state.interrupts).toHaveLength(1);
    expect(state.plan?.awaitingApproval).toBe(true);
    expect(state.plan?.tasks).toEqual([
      { id: "t1", description: "步骤", dependsOn: ["t0"] },
    ]);
    expect(effects.map((effect) => effect.type)).toEqual([
      "interrupts",
      "plan",
      "finish",
    ]);
  });

  it("end：终态映射（stop→2 / canceled→4 / error→3）；中断挂起保持生成中并按需轮询", () => {
    const usage = {
      inputTokens: 1,
      outputTokens: 2,
      cachedInputTokens: 3,
      credits: 4,
    };
    expect(
      reduce(baseState(), { kind: "end", data: { stopReason: "stop", usage } })
        .effects
    ).toEqual([{ type: "end", usage, status: 2, pollAsync: false }]);
    expect(
      reduce(baseState(), {
        kind: "end",
        data: { stopReason: "canceled", usage },
      }).effects[0]
    ).toMatchObject({ status: 4 });
    expect(
      reduce(baseState(), { kind: "end", data: { stopReason: "error", usage } })
        .effects[0]
    ).toMatchObject({ status: 3 });

    const held = reduce(
      baseState({
        interrupts: [{ type: "async_wait", data: {} }],
        interruptedMessageId: 4,
      }),
      { kind: "end", data: { stopReason: "stop", usage } }
    );
    expect(held.effects[0]).toMatchObject({
      type: "end",
      status: null,
      pollAsync: true,
    });
  });

  it("纯函数：入参不被改写、同输入同输出", () => {
    const blocks = new Map([[2, { args: "x" }]]);
    const state = baseState({ toolBlocks: blocks });
    const event: StreamEvent = {
      kind: "delta",
      data: { index: 2, delta: { type: "input_json_delta", partialJson: "y" } },
    };
    const first = reduce(state, event);
    const second = reduce(state, event);

    expect(first.state.toolBlocks).not.toBe(blocks);
    expect(blocks.get(2)).toEqual({ args: "x" });
    expect(first.state.toolBlocks.get(2)).toEqual({ args: "xy" });
    expect(first.state).toEqual(second.state);
  });

  it("对抗：思考块 index 偏离契约显式报错（不静默容忍）", () => {
    expect(() =>
      reduce(baseState(), {
        kind: "block-start",
        data: { index: 0, type: "thinking" },
      })
    ).toThrow(/契约偏离/);
  });
});
