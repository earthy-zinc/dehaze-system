import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import InterruptCard from "../components/InterruptCard.vue";
import type { ChatInterruptVM } from "../types";
import { elStubs, findButton } from "./stubs";

const mountCard = (interrupt: ChatInterruptVM) =>
  mount(InterruptCard, {
    props: { interrupt },
    global: { stubs: elStubs },
  });

describe("InterruptCard", () => {
  describe("confirm 按 confirmKind 路由", () => {
    it("algorithm_recommend：采纳推荐 / 选择备选 / 拒绝", async () => {
      const wrapper = mountCard({
        type: "confirm",
        confirmKind: "algorithm_recommend",
        recommendation: {
          algorithmId: 7,
          algorithmName: "去雾",
          reason: "最佳",
        },
        alternatives: [
          {
            algorithmId: 8,
            algorithmName: "备选",
            matchScore: 0.5,
            reason: "次优",
          },
        ],
      });

      await findButton(wrapper, "采纳推荐")!.trigger("click");
      await findButton(wrapper, "备选")!.trigger("click");
      await findButton(wrapper, "拒绝")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true, params: { algorithmId: 7 } }],
        [{ confirm: true, params: { algorithmId: 8 } }],
        [{ confirm: false }],
      ]);
    });

    it("tool_permission：允许 / 拒绝", async () => {
      const wrapper = mountCard({
        type: "confirm",
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
        type: "confirm",
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

    it("write_conflict：覆盖写入携带 action 参数", async () => {
      const wrapper = mountCard({
        type: "confirm",
        confirmKind: "write_conflict",
        action: "write_conflict",
        detail: "文件已被修改",
      });
      expect(wrapper.text()).toContain("写入冲突");

      await findButton(wrapper, "覆盖写入")!.trigger("click");

      expect(wrapper.emitted("resume")).toEqual([
        [{ confirm: true, params: { action: "write_conflict" } }],
      ]);
    });

    it("无 confirmKind 的通用确认：继续 / 拒绝", async () => {
      const wrapper = mountCard({ type: "confirm", reason: "需要确认" });
      await findButton(wrapper, "继续")!.trigger("click");
      expect(wrapper.emitted("resume")).toEqual([[{ confirm: true }]]);
    });
  });

  describe("plan_approve", () => {
    const planInterrupt: ChatInterruptVM = {
      type: "plan_approve",
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
});
