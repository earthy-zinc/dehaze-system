import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import type { Interrupt } from "clarity-sdk-js";
import InterruptCard from "../components/InterruptCard.vue";
import { elStubs, findButton } from "./stubs";

const interrupt = (over: Partial<Interrupt> = {}): Interrupt => ({
  id: "i1",
  type: "confirm",
  createdAt: "2026-01-01T00:00:00Z",
  expiresAt: "2099-01-01T00:00:00Z",
  actionable: true,
  ...over,
});

const mountCard = (over: Partial<Interrupt> = {}) =>
  mount(InterruptCard, {
    props: { interrupt: interrupt(over) },
    global: { stubs: elStubs },
  });

describe("InterruptCard", () => {
  describe("confirm 按契约 confirmKind 路由", () => {
    it("algorithm_recommend：采纳推荐 / 选择备选 / 拒绝", async () => {
      const wrapper = mountCard({
        confirmKind: "algorithm_recommend",
        recommendation: {
          modelId: "aecrnet",
          modelName: "AECR-Net",
          reason: "最佳",
        },
        alternatives: [
          {
            modelId: "dcp",
            modelName: "DCP",
            matchScore: 0.5,
            reason: "次优",
          },
        ],
      });

      await findButton(wrapper, "采纳推荐")!.trigger("click");
      await findButton(wrapper, "DCP")!.trigger("click");
      await findButton(wrapper, "拒绝")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true, params: { modelId: "aecrnet" } }],
        [{ confirm: true, params: { modelId: "dcp" } }],
        [{ confirm: false }],
      ]);
    });

    it("tool_permission：允许 / 拒绝", async () => {
      const wrapper = mountCard({
        confirmKind: "tool_permission",
        detail: "调用 write_file",
      });
      expect(wrapper.text()).toContain("工具授权确认");

      await findButton(wrapper, "允许")!.trigger("click");
      await findButton(wrapper, "拒绝")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true }],
        [{ confirm: false }],
      ]);
    });

    it("dangerous_op：确认执行 / 取消", async () => {
      const wrapper = mountCard({
        confirmKind: "dangerous_op",
        detail: "将删除文件",
      });
      expect(wrapper.text()).toContain("危险操作确认");

      await findButton(wrapper, "确认执行")!.trigger("click");
      await findButton(wrapper, "取消")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true }],
        [{ confirm: false }],
      ]);
    });

    it("write_conflict：后端直出 confirmKind，覆盖写入仅需 confirm", async () => {
      const wrapper = mountCard({
        confirmKind: "write_conflict",
        action: "write_conflict",
        detail: "文件已被修改",
      });
      expect(wrapper.text()).toContain("写入冲突");

      await findButton(wrapper, "覆盖写入")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([[{ confirm: true }]]);
    });

    it("无 confirmKind：禁用操作，不发起 resume", async () => {
      const wrapper = mountCard({ reason: "需要确认" });
      expect(wrapper.text()).toContain("确认类型缺失");
      await findButton(wrapper, "继续")!.trigger("click");
      expect(wrapper.emitted("resume")).toBeUndefined();
    });

    it("未知 confirmKind：与缺失同处理，按钮禁用且点击不发起 resume", async () => {
      const wrapper = mountCard({
        confirmKind: "bogus" as Interrupt["confirmKind"],
        detail: "契约外子类型",
      });

      expect(wrapper.text()).toContain("确认类型缺失，无法继续");
      const proceed = findButton(wrapper, "继续")!;
      const cancel = findButton(wrapper, "拒绝")!;
      expect(proceed.attributes("disabled")).toBeDefined();
      expect(cancel.attributes("disabled")).toBeDefined();

      await proceed.trigger("click");
      await cancel.trigger("click");
      expect(wrapper.emitted("resume")).toBeUndefined();
    });
  });

  describe("plan_approve", () => {
    const planInterrupt = {
      type: "plan_approve" as const,
      plan: [
        { id: "t1", description: "步骤一" },
        { id: "t2", description: "步骤二" },
      ],
    };

    it("无改动批准：仅 confirm=true", async () => {
      const wrapper = mountCard(planInterrupt);
      await findButton(wrapper, "批准执行")!.trigger("click");
      expect(wrapper.emitted("resume")).toEqual([[{ confirm: true }]]);
    });

    it("移除任务后批准：planEdit.remove 携带被移除 id", async () => {
      const wrapper = mountCard(planInterrupt);
      const removeButtons = wrapper
        .findAll("button")
        .filter((b) => b.text().includes("移除"));
      await removeButtons[0].trigger("click");
      await findButton(wrapper, "批准执行")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true, planEdit: { remove: ["t1"] } }],
      ]);
    });

    it("下移任务后批准：planEdit.reorder 携带新顺序", async () => {
      const wrapper = mountCard(planInterrupt);
      const downButtons = wrapper
        .findAll("button")
        .filter((b) => b.text().includes("下移"));
      await downButtons[0].trigger("click");
      await findButton(wrapper, "批准执行")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true, planEdit: { reorder: ["t2", "t1"] } }],
      ]);
    });

    it("新增任务后批准：planEdit.add 为单对象（对齐后端 wire 契约）", async () => {
      const wrapper = mountCard(planInterrupt);
      await findButton(wrapper, "+ 添加任务")!.trigger("click");

      const inputs = wrapper.findAll(".el-input-stub");
      await inputs[inputs.length - 1].setValue("新任务");
      await findButton(wrapper, "批准执行")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [
          {
            confirm: true,
            planEdit: { add: { description: "新任务", dependsOn: [] } },
          },
        ],
      ]);
    });

    it("新增任务收敛为单条：再次点击不再追加新任务行", async () => {
      const wrapper = mountCard(planInterrupt);

      await findButton(wrapper, "+ 添加任务")!.trigger("click");
      await findButton(wrapper, "+ 添加任务")!.trigger("click");
      await findButton(wrapper, "+ 添加任务")!.trigger("click");

      // 初始 2 个任务 + 至多 1 条新增（后端 plan_edit.add 仅接受单对象）
      expect(wrapper.findAll(".el-input-stub")).toHaveLength(3);
    });

    it("取消：confirm=false", async () => {
      const wrapper = mountCard(planInterrupt);
      await findButton(wrapper, "取消")!.trigger("click");
      expect(wrapper.emitted("resume")).toEqual([[{ confirm: false }]]);
    });
  });

  it("quota：展示用量并重试续流", async () => {
    const wrapper = mountCard({
      type: "quota",
      upgradeTip: "积分不足",
      usedDaily: 10,
      dailyLimit: 10,
    });
    expect(wrapper.text()).toContain("今日已用 10 / 10");
    await findButton(wrapper, "重试")!.trigger("click");
    expect(wrapper.emitted("resume")).toEqual([[{}]]);
  });

  it("async_wait：纯信息展示，无动作按钮", () => {
    const wrapper = mountCard({
      type: "async_wait",
      estDuration: "30s",
      imageCount: 3,
    });
    expect(wrapper.text()).toContain("预计耗时 30s");
    expect(wrapper.findAll("button")).toHaveLength(0);
  });

  describe("过期态", () => {
    it("actionable=false：提示已失效、按钮禁用且点击不发起 resume", async () => {
      const wrapper = mountCard({
        confirmKind: "tool_permission",
        detail: "调用 write_file",
        actionable: false,
      });

      expect(wrapper.text()).toContain("已失效，不可继续");
      const allow = findButton(wrapper, "允许")!;
      expect(allow.attributes("disabled")).toBeDefined();
      await allow.trigger("click");
      expect(wrapper.emitted("resume")).toBeUndefined();
    });

    it("expiresAt 已过：即使 actionable=true 也按过期处理", async () => {
      const wrapper = mountCard({
        confirmKind: "tool_permission",
        reason: "需要确认",
        expiresAt: "2020-01-01T00:00:00Z",
      });

      expect(wrapper.text()).toContain("已失效，不可继续");
      await findButton(wrapper, "允许")!.trigger("click");
      expect(wrapper.emitted("resume")).toBeUndefined();
    });

    it("expiresAt 未到且 actionable=true：正常可操作", async () => {
      const wrapper = mountCard({
        confirmKind: "tool_permission",
        reason: "需要确认",
      });

      expect(wrapper.text()).not.toContain("已失效");
      await findButton(wrapper, "允许")!.trigger("click");
      expect(wrapper.emitted("resume")).toEqual([[{ confirm: true }]]);
    });
  });
});
