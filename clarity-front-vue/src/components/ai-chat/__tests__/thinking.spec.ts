// 思考过程选择器单测：thinking 块按 seq 分段、durationMs 判定闭合与计时。
import { describe, expect, it } from "vitest";
import type { Block } from "clarity-sdk-js";
import { thinkingOfBlocks } from "../model/blocks";

function thinkingBlock(
  id: string,
  seq: number,
  text: string,
  durationMs: number | null = null
): Block {
  return { id, kind: "thinking", seq, text, durationMs };
}

const BASE = 1_700_000_000_000;

describe("thinkingOfBlocks", () => {
  it("无 thinking 块返回 null", () => {
    expect(thinkingOfBlocks([], BASE)).toBeNull();
    expect(
      thinkingOfBlocks([{ id: "a", kind: "text", seq: 1, text: "正文" }], BASE)
    ).toBeNull();
  });

  it("多段思考按 seq 升序，未收尾段标记 streaming", () => {
    const thinking = thinkingOfBlocks(
      [
        thinkingBlock("k2", 2, "第二段"),
        thinkingBlock("k1", 1, "第一段", 1200),
      ],
      BASE
    )!;

    expect(thinking.segments).toEqual([
      { text: "第一段", closed: true },
      { text: "第二段", closed: false },
    ]);
    expect(thinking.streaming).toBe(true);
    expect(thinking.startAt).toBe(BASE);
    expect(thinking.endAt).toBeNull();
  });

  it("首个思考块之前有工具块时，计时起点加上前置块时长", () => {
    const thinking = thinkingOfBlocks(
      [
        {
          id: "t1",
          kind: "tool_use",
          seq: 1,
          durationMs: 860,
          tool: {
            callId: "c1",
            name: "algorithm_recommend",
            status: "success",
            subAgent: false,
          },
        },
        thinkingBlock("k1", 1, "开始思考", 500),
      ],
      BASE
    )!;

    expect(thinking.startAt).toBe(BASE + 860);
    expect(thinking.endAt).toBe(BASE + 860 + 500);
  });

  it("全部收尾后按 durationMs 之和定格", () => {
    const thinking = thinkingOfBlocks(
      [
        thinkingBlock("k1", 1, "第一段", 1200),
        thinkingBlock("k2", 2, "第二段", 800),
      ],
      BASE
    )!;

    expect(thinking.streaming).toBe(false);
    expect(thinking.endAt).toBe(BASE + 2000);
  });

  it("对抗：text 为 null 的思考块按空串分段，不抛错", () => {
    const thinking = thinkingOfBlocks(
      [{ id: "k", kind: "thinking", seq: 1, text: null, durationMs: 10 }],
      BASE
    )!;
    expect(thinking.segments).toEqual([{ text: "", closed: true }]);
  });
});
