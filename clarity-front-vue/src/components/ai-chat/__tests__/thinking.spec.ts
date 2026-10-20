// 思考过程归约单测：分段 / 幂等 / 收尾 + 对抗性边界（空数组、未闭合即收尾、重复 start）。
import { describe, expect, it } from "vitest";
import type { ChatThinkingVM, ChatThoughtStepVM } from "../types";
import {
  appendThinkingDelta,
  buildThinkingFromThoughts,
  closeThinkingSegment,
  finalizeThinking,
  openThinkingSegment,
} from "../vm/thinking";

const base = (): ChatThinkingVM => ({
  segments: [],
  startAt: 100,
  endAt: null,
  streaming: false,
});

describe("thinking 归约", () => {
  describe("openThinkingSegment", () => {
    it("空态开首段", () => {
      const opened = openThinkingSegment(base());
      expect(opened.segments).toEqual([{ text: "", closed: false }]);
      expect(opened.streaming).toBe(true);
      expect(opened.startAt).toBe(100);
    });

    it("末段未关闭时幂等忽略（断线重放重复 start）", () => {
      const opened = openThinkingSegment(base());
      expect(openThinkingSegment(opened)).toBe(opened);
    });

    it("末段已关闭时追加新段（多段思考）", () => {
      let state = openThinkingSegment(base());
      state = closeThinkingSegment(state, 200);
      const reopened = openThinkingSegment(state);
      expect(reopened.segments).toEqual([
        { text: "", closed: true },
        { text: "", closed: false },
      ]);
      expect(reopened.endAt).toBe(200);
    });
  });

  describe("appendThinkingDelta", () => {
    it("增量累积到当前未闭合段", () => {
      let state = openThinkingSegment(base());
      state = appendThinkingDelta(state, "你好");
      state = appendThinkingDelta(state, "世界");
      expect(state.segments).toEqual([{ text: "你好世界", closed: false }]);
      expect(state.streaming).toBe(true);
    });

    it("空字符串增量无副作用（返回同引用）", () => {
      const state = openThinkingSegment(base());
      expect(appendThinkingDelta(state, "")).toBe(state);
    });

    it("无 start 的残留流：末段已闭合时开新段", () => {
      let state = openThinkingSegment(base());
      state = appendThinkingDelta(state, "第一段");
      state = closeThinkingSegment(state, 200);
      state = appendThinkingDelta(state, "第二段");
      expect(state.segments).toEqual([
        { text: "第一段", closed: true },
        { text: "第二段", closed: false },
      ]);
    });

    it("空态直接追加也能开段（无段兜底）", () => {
      const state = appendThinkingDelta(base(), "直接增量");
      expect(state.segments).toEqual([{ text: "直接增量", closed: false }]);
    });
  });

  describe("closeThinkingSegment", () => {
    it("闭合末段并定格计时", () => {
      let state = openThinkingSegment(base());
      state = appendThinkingDelta(state, "思考内容");
      const closed = closeThinkingSegment(state, 999);
      expect(closed.segments).toEqual([{ text: "思考内容", closed: true }]);
      expect(closed.endAt).toBe(999);
      expect(closed.streaming).toBe(false);
    });

    it("无可闭合段时幂等忽略（不写 endAt）", () => {
      const empty = base();
      expect(closeThinkingSegment(empty, 5)).toBe(empty);

      let state = openThinkingSegment(base());
      state = closeThinkingSegment(state, 200);
      const again = closeThinkingSegment(state, 300);
      expect(again).toBe(state);
      expect(again.endAt).toBe(200);
    });
  });

  describe("finalizeThinking", () => {
    it("段未关闭即收尾：闭合全部段并定格计时", () => {
      let state = openThinkingSegment(base());
      state = appendThinkingDelta(state, "未收尾的思考");
      const done = finalizeThinking(state, 500);
      expect(done.segments).toEqual([{ text: "未收尾的思考", closed: true }]);
      expect(done.endAt).toBe(500);
      expect(done.streaming).toBe(false);
    });

    it("已有 endAt 保持不变", () => {
      let state = openThinkingSegment(base());
      state = closeThinkingSegment(state, 500);
      expect(finalizeThinking(state, 800).endAt).toBe(500);
    });

    it("空态收尾：无段也定格 endAt", () => {
      const done = finalizeThinking(base(), 7);
      expect(done.segments).toEqual([]);
      expect(done.endAt).toBe(7);
      expect(done.streaming).toBe(false);
    });
  });

  describe("buildThinkingFromThoughts", () => {
    it("空数组返回 null", () => {
      expect(buildThinkingFromThoughts([])).toBeNull();
    });

    it("仅含工具步骤（无纯思考）返回 null", () => {
      const steps: ChatThoughtStepVM[] = [
        { position: 1, status: 1, tool: "search" },
      ];
      expect(buildThinkingFromThoughts(steps)).toBeNull();
    });

    it("抽取纯思考步骤为已闭合段，忽略工具步骤与空 thought", () => {
      const steps: ChatThoughtStepVM[] = [
        { position: 1, status: 1, thought: "先想" },
        { position: 2, status: 1, thought: "" },
        { position: 3, status: 1, thought: "带工具的思考", tool: "search" },
        { position: 4, status: 2, thought: "再想" },
      ];
      expect(buildThinkingFromThoughts(steps)).toEqual({
        segments: [
          { text: "先想", closed: true },
          { text: "再想", closed: true },
        ],
        startAt: 0,
        endAt: null,
        streaming: false,
      });
    });
  });
});
