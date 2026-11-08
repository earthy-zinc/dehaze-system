import { describe, expect, it } from "vitest";
import type { Interrupt, PlanTask } from "clarity-sdk-js";
import {
  appendPlanTask,
  buildPlanEdit,
  canApprovePlan,
  confirmActionOf,
  editablePlanOf,
  hasNewPlanTask,
  interruptTitle,
  isExpiredAt,
  isPlanEditEmpty,
  movePlanTask,
  parseExpiresAt,
  planIdsOf,
  planSignatureOf,
  removePlanTaskAt,
  type EditablePlanTask,
} from "../model/interrupt";

const interrupt = (over: Partial<Interrupt> = {}): Interrupt => ({
  id: "i1",
  type: "plan_approve",
  createdAt: "2026-01-01T00:00:00Z",
  expiresAt: "2099-01-01T00:00:00Z",
  actionable: true,
  ...over,
});

const plan: PlanTask[] = [
  { id: "t1", description: "步骤一" },
  { id: "t2", description: "步骤二" },
];

const rows = (): EditablePlanTask[] => editablePlanOf(plan);

describe("model/interrupt", () => {
  describe("计划编辑", () => {
    it("editablePlanOf：id/描述投影为可编辑行，新增标记为 false", () => {
      expect(rows()).toEqual([
        { id: "t1", description: "步骤一", isNew: false },
        { id: "t2", description: "步骤二", isNew: false },
      ]);
      expect(editablePlanOf(null)).toEqual([]);
    });

    it("planIdsOf：跳过无 id 的脏数据", () => {
      expect(planIdsOf(plan)).toEqual(["t1", "t2"]);
      expect(planIdsOf([{ id: "" }, ...plan])).toEqual(["t1", "t2"]);
    });

    it("movePlanTask：上移 / 下移 / 越界原样", () => {
      expect(movePlanTask(rows(), 1, -1).map((t) => t.id)).toEqual([
        "t2",
        "t1",
      ]);
      expect(movePlanTask(rows(), 0, 1).map((t) => t.id)).toEqual(["t2", "t1"]);
      expect(movePlanTask(rows(), 0, -1).map((t) => t.id)).toEqual([
        "t1",
        "t2",
      ]);
      expect(movePlanTask(rows(), 1, 1).map((t) => t.id)).toEqual(["t1", "t2"]);
    });

    it("movePlanTask：不修改入参", () => {
      const input = rows();
      movePlanTask(input, 1, -1);
      expect(input.map((t) => t.id)).toEqual(["t1", "t2"]);
    });

    it("removePlanTaskAt：移除指定行且不修改入参", () => {
      const input = rows();
      expect(removePlanTaskAt(input, 0).map((t) => t.id)).toEqual(["t2"]);
      expect(input).toHaveLength(2);
    });

    it("appendPlanTask：只允许一条待新增行", () => {
      const once = appendPlanTask(rows());
      expect(once).toHaveLength(3);
      expect(hasNewPlanTask(once)).toBe(true);
      expect(appendPlanTask(once)).toHaveLength(3);
    });

    it("canApprovePlan：有描述或有 id 即可批准", () => {
      expect(canApprovePlan(rows())).toBe(true);
      expect(canApprovePlan([{ description: "  ", isNew: true }])).toBe(false);
      expect(
        canApprovePlan([{ id: "t1", description: "", isNew: false }])
      ).toBe(true);
    });
  });

  describe("buildPlanEdit", () => {
    it("无改动时为空对象（批准只回 confirm=true）", () => {
      const edit = buildPlanEdit(rows(), planIdsOf(plan));
      expect(edit).toEqual({});
      expect(isPlanEditEmpty(edit)).toBe(true);
    });

    it("移除：remove 携带被删 id", () => {
      const edit = buildPlanEdit(removePlanTaskAt(rows(), 0), planIdsOf(plan));
      expect(edit).toEqual({ remove: ["t1"] });
      expect(isPlanEditEmpty(edit)).toBe(false);
    });

    it("排序：reorder 携带新顺序", () => {
      const edit = buildPlanEdit(movePlanTask(rows(), 0, 1), planIdsOf(plan));
      expect(edit).toEqual({ reorder: ["t2", "t1"] });
    });

    it("新增：add 为单对象且描述 trim", () => {
      const withNew = appendPlanTask(rows()).map((task) =>
        task.isNew ? { ...task, description: "  新任务  " } : task
      );
      expect(buildPlanEdit(withNew, planIdsOf(plan))).toEqual({
        add: { description: "新任务", dependsOn: [] },
      });
    });

    it("新增但描述为空：不产生 add", () => {
      const edit = buildPlanEdit(appendPlanTask(rows()), planIdsOf(plan));
      expect(edit).toEqual({});
    });

    it("删除后顺序变化不重复下发 reorder（以集合变动为准）", () => {
      const edit = buildPlanEdit(removePlanTaskAt(rows(), 1), planIdsOf(plan));
      expect(edit).toEqual({ remove: ["t2"] });
    });
  });

  describe("planSignatureOf", () => {
    it("内容相同签名相同（父级重渲染生成新对象不会重建编辑）", () => {
      expect(planSignatureOf(interrupt({ plan }))).toBe(
        planSignatureOf(interrupt({ plan: plan.map((task) => ({ ...task })) }))
      );
    });

    it("计划变化签名变化", () => {
      expect(planSignatureOf(interrupt({ plan }))).not.toBe(
        planSignatureOf(interrupt({ plan: [plan[0]] }))
      );
    });
  });

  describe("过期判定", () => {
    it("parseExpiresAt：非法或缺失返回 null", () => {
      expect(parseExpiresAt(null)).toBeNull();
      expect(parseExpiresAt("")).toBeNull();
      expect(parseExpiresAt("not-a-date")).toBeNull();
      expect(parseExpiresAt("2026-01-01T00:00:00Z")).toBe(
        Date.parse("2026-01-01T00:00:00Z")
      );
    });

    it("isExpiredAt：无到期时间不过期，到点即过期", () => {
      const at = Date.parse("2026-01-01T00:00:00Z");
      expect(isExpiredAt(null, at)).toBe(false);
      expect(isExpiredAt(at, at - 1)).toBe(false);
      expect(isExpiredAt(at, at)).toBe(true);
      expect(isExpiredAt(at, at + 1)).toBe(true);
    });
  });

  describe("confirmActionOf", () => {
    it.each([
      ["algorithm_recommend", "采纳推荐", "拒绝"],
      ["tool_permission", "允许", "拒绝"],
      ["dangerous_op", "确认执行", "取消"],
      ["write_conflict", "覆盖写入", "取消"],
    ] as const)("%s 的动作文案", (kind, accept, reject) => {
      const action = confirmActionOf(kind);
      expect(action.accept).toBe(accept);
      expect(action.reject).toBe(reject);
      expect(action.fallback).not.toBe("");
    });

    it("危险操作的接受按钮用 danger 样式", () => {
      expect(confirmActionOf("dangerous_op").acceptType).toBe("danger");
      expect(confirmActionOf("tool_permission").acceptType).toBe("primary");
    });
  });

  describe("interruptTitle", () => {
    it.each([
      ["plan_approve", undefined, "计划待确认"],
      ["quota", undefined, "配额不足"],
      ["async_wait", undefined, "异步任务处理中"],
      ["confirm", "algorithm_recommend", "需要你的确认"],
      ["confirm", "tool_permission", "工具授权确认"],
      ["confirm", "dangerous_op", "危险操作确认"],
      ["confirm", "write_conflict", "写入冲突"],
      ["confirm", undefined, "需要你的确认"],
    ] as const)("%s / %s → %s", (type, kind, expected) => {
      expect(interruptTitle(interrupt({ type }), kind)).toBe(expected);
    });

    it("契约外中断类型兜底标题", () => {
      expect(
        interruptTitle(interrupt({ type: "bogus" as Interrupt["type"] }))
      ).toBe("推理已暂停");
    });
  });
});
