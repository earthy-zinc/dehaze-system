// 块派生单测：正文拼接、工具步骤投影（契约 ToolStatus 原样透传）、用量可见性。
import { describe, expect, it } from "vitest";
import type { Block, ToolCall, Usage } from "clarity-sdk-js";
import { hasUsage, stepsOfBlocks, textOfBlocks } from "../model/blocks";

function block(over: Partial<Block> & Pick<Block, "id" | "kind">): Block {
  return { seq: 1, ...over };
}

function tool(over: Partial<ToolCall> = {}): ToolCall {
  return {
    callId: "c1",
    name: "search",
    subAgent: false,
    status: "success",
    ...over,
  };
}

describe("textOfBlocks", () => {
  it("按 seq 拼接 text 块，非 text 块不进入正文", () => {
    const blocks = [
      block({ id: "t2", kind: "text", seq: 2, text: "后半" }),
      block({ id: "k1", kind: "thinking", seq: 1, text: "思考" }),
      block({ id: "t1", kind: "text", seq: 1, text: "前半" }),
    ];
    expect(textOfBlocks(blocks)).toBe("前半\n\n后半");
  });

  it("空文本块不产生多余空行", () => {
    expect(
      textOfBlocks([
        block({ id: "a", kind: "text", seq: 1, text: "" }),
        block({ id: "b", kind: "text", seq: 2, text: "正文" }),
      ])
    ).toBe("正文");
  });

  it("无 text 块返回空串", () => {
    expect(textOfBlocks([])).toBe("");
  });
});

describe("stepsOfBlocks", () => {
  it("tool_use 块投影为工具步骤：position=seq、status 原样字符串", () => {
    const steps = stepsOfBlocks([
      block({
        id: "tool",
        kind: "tool_use",
        seq: 2,
        tool: tool({
          name: "search",
          input: { q: "天气" },
          output: "结果",
          summary: "摘要",
          status: "failed",
          error: "超时",
          latencyMs: 800,
          agentCode: "sub",
          subAgent: true,
        }),
      }),
    ]);

    expect(steps).toEqual([
      {
        position: 2,
        status: "failed",
        tool: "search",
        toolInput: { q: "天气" },
        observation: "结果",
        error: "超时",
        latencyMs: 800,
      },
    ]);
  });

  it("跳过/执行中状态原样透传（不再归一为 1/2/3）", () => {
    const steps = stepsOfBlocks([
      block({
        id: "a",
        kind: "tool_use",
        seq: 1,
        tool: tool({ status: "skipped" }),
      }),
      block({
        id: "b",
        kind: "tool_use",
        seq: 2,
        tool: tool({ callId: "c2", status: "running" }),
      }),
    ]);
    expect(steps.map((step) => step.status)).toEqual(["skipped", "running"]);
  });

  it("output 缺失时回退 summary；tool 缺失时状态按 running", () => {
    const steps = stepsOfBlocks([
      block({
        id: "a",
        kind: "tool_use",
        seq: 1,
        tool: tool({ output: null, summary: "摘要" }),
      }),
      block({ id: "b", kind: "tool_use", seq: 2, tool: null }),
    ]);
    expect(steps[0].observation).toBe("摘要");
    expect(steps[1]).toEqual({ position: 2, status: "running" });
  });

  it("思考/正文块不产生步骤", () => {
    expect(
      stepsOfBlocks([
        block({ id: "t", kind: "text", seq: 1, text: "正文" }),
        block({ id: "k", kind: "thinking", seq: 2, text: "思考" }),
      ])
    ).toEqual([]);
  });
});

describe("hasUsage", () => {
  const zero: Usage = {
    inputTokens: 0,
    outputTokens: 0,
    cachedInputTokens: 0,
    credits: 0,
  };

  it("全 0 且无子智能体用量不展示", () => {
    expect(hasUsage(zero)).toBe(false);
    expect(hasUsage({ ...zero, subAgents: [] })).toBe(false);
    expect(hasUsage(null)).toBe(false);
    expect(hasUsage(undefined)).toBe(false);
  });

  it("任一 Token/积分为非 0 即展示", () => {
    expect(hasUsage({ ...zero, inputTokens: 1 })).toBe(true);
    expect(hasUsage({ ...zero, credits: 1 })).toBe(true);
  });

  it("仅子智能体用量也展示", () => {
    expect(
      hasUsage({
        ...zero,
        subAgents: [
          {
            agentCode: "planner",
            inputTokens: 0,
            outputTokens: 0,
            cachedInputTokens: 0,
            credits: 0,
          },
        ],
      })
    ).toBe(true);
  });
});
