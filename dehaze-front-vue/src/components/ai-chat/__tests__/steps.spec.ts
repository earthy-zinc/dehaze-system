// 推理步骤归约单测：排序 / 过滤 + 对抗性边界（乱序、重复 position、空数组）。
import { describe, expect, it } from "vitest";
import type { ChatThoughtStepVM } from "../types";
import { filterToolSteps, sortStepsByPosition } from "../vm/steps";

const step = (over: Partial<ChatThoughtStepVM>): ChatThoughtStepVM => ({
  position: 0,
  status: 1,
  ...over,
});

describe("steps 归约", () => {
  describe("sortStepsByPosition", () => {
    it("乱序按 position 升序", () => {
      const sorted = sortStepsByPosition([
        step({ position: 3 }),
        step({ position: 1 }),
        step({ position: 2 }),
      ]);
      expect(sorted.map((s) => s.position)).toEqual([1, 2, 3]);
    });

    it("重复 position 保持原相对顺序（稳定排序）", () => {
      const sorted = sortStepsByPosition([
        step({ position: 1, thought: "b" }),
        step({ position: 1, thought: "a" }),
      ]);
      expect(sorted.map((s) => s.thought)).toEqual(["b", "a"]);
    });

    it("空数组返回空数组", () => {
      expect(sortStepsByPosition([])).toEqual([]);
    });

    it("不修改入参", () => {
      const input = [step({ position: 2 }), step({ position: 1 })];
      const snapshot = [...input];
      sortStepsByPosition(input);
      expect(input).toEqual(snapshot);
    });
  });

  describe("filterToolSteps", () => {
    it("仅保留 tool 非空步骤", () => {
      const filtered = filterToolSteps([
        step({ position: 1 }),
        step({ position: 2, tool: "search" }),
        step({ position: 3, tool: "" }),
        step({ position: 4, tool: "task" }),
      ]);
      expect(filtered.map((s) => s.position)).toEqual([2, 4]);
    });

    it("空数组返回空数组", () => {
      expect(filterToolSteps([])).toEqual([]);
    });
  });
});
