import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiScheduleAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createScheduleForm,
  createScheduleQuery,
  createScheduleStatusForm,
  createScheduleUpdateForm,
} from "#/factories/ai-schedule";

/**
 * AI 定时调度（T-SC-050~064）
 * 创建需 VIP2+（A0503），普通用户/VIP1 被拒；数据前缀 test_task_，afterAll 软删除清理。
 */
describe("AI 定时调度 - AiScheduleAPI (T-SC-050~064)", () => {
  let scheduleId: number;
  let scheduleForm: ReturnType<typeof createScheduleForm>;
  const createdIds: number[] = [];

  const asVip2 = async () => login(USERS.VIP2.username);

  beforeAll(async () => {
    await asVip2();
  });

  afterAll(async () => {
    await asVip2().catch(() => {});
    for (const id of [...createdIds].reverse()) {
      await AiScheduleAPI.delete(id).catch(() => {});
    }
  });

  describe("POST /api/v1/ai/scheduled-tasks - 创建任务", () => {
    test("T-SC-050 正向：创建任务返回 nextTriggerTime", async () => {
      await asVip2();
      const form = createScheduleForm();
      const result = await AiScheduleAPI.create(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toBe(form.name);
      expect(result.cron).toBe(form.cron);
      expect(result.enabled).toBe(1);
      expect(result.status).toBe(1);
      expect(result.nextTriggerTime).toBeTruthy();
      scheduleId = result.id;
      scheduleForm = form;
      createdIds.push(result.id);
    });

    test("T-SC-054 负向：普通用户创建 → 拒绝", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiScheduleAPI.create(createScheduleForm()), ["A0503", "A0500", "A0403"]);
      await asVip2();
    });

    test("T-SC-054 负向：VIP1 创建 → 拒绝", async () => {
      await login(USERS.VIP1.username);
      await expectBizError(AiScheduleAPI.create(createScheduleForm()), ["A0503"]);
      await asVip2();
    });

    test("T-SC-064 负向：非法 Cron → 参数错误 A0400", async () => {
      await asVip2();
      await expectBizError(AiScheduleAPI.create(createScheduleForm({ cron: "invalid-cron" })), [
        "A0400",
        "B0001",
      ]);
    });
  });

  describe("GET /api/v1/ai/scheduled-tasks - 任务列表", () => {
    test("T-SC-050 列表分页结构 list/total", async () => {
      await asVip2();
      const result = await AiScheduleAPI.list(createScheduleQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        const item = result.list[0]!;
        expect(item.id).toBeGreaterThan(0);
      }
    });

    test("T-SC-050 列表包含刚创建的任务", async () => {
      await asVip2();
      const result = await AiScheduleAPI.list(createScheduleQuery({ pageSize: 100 }));
      const found = result.list.find((s) => s.id === scheduleId);
      expect(found).toBeDefined();
      expect(found!.name).toBe(scheduleForm.name);
    });
  });

  describe("GET /api/v1/ai/scheduled-tasks/next-times - Cron 预览", () => {
    test("T-SC-064 正向：返回 description + nextTimes", async () => {
      await asVip2();
      const result = await AiScheduleAPI.previewNextTimes("0 9 * * *", 5);
      expect(typeof result.description).toBe("string");
      expect(result.description.length).toBeGreaterThan(0);
      expect(Array.isArray(result.nextTimes)).toBe(true);
      expect(result.nextTimes.length).toBeGreaterThan(0);
    });

    test("T-SC-064 负向：非法 Cron → A0400", async () => {
      await asVip2();
      await expectBizError(AiScheduleAPI.previewNextTimes("bad cron"), ["A0400"]);
    });
  });

  describe("GET /api/v1/ai/scheduled-tasks/{id} - 任务详情", () => {
    test("T-SC-050 正向：查询详情", async () => {
      await asVip2();
      const detail = await AiScheduleAPI.detail(scheduleId);
      expect(detail.id).toBe(scheduleId);
      expect(detail.userId).toBe(USERS.VIP2.id);
      expect(detail.name).toBe(scheduleForm.name);
    });

    test("T-SC-072 边界：查询不存在任务 → A0401", async () => {
      await asVip2();
      await expectBizError(AiScheduleAPI.detail(99999999), ["A0401"]);
    });
  });

  describe("PUT /api/v1/ai/scheduled-tasks/{id} - 更新任务", () => {
    test("T-SC-050 正向：更新名称/Cron 重算下次触发", async () => {
      await asVip2();
      const updated = await AiScheduleAPI.update(scheduleId, createScheduleUpdateForm());
      expect(updated.id).toBe(scheduleId);
      expect(updated.nextTriggerTime).toBeTruthy();
      expect(updated.name).toContain("test_task_updated");
    });
  });

  describe("PATCH /api/v1/ai/scheduled-tasks/{id}/status - 启停", () => {
    test("T-SC-068 正向：禁用后 enabled=0", async () => {
      await asVip2();
      await AiScheduleAPI.setStatus(scheduleId, createScheduleStatusForm({ enabled: 0 }));
      const detail = await AiScheduleAPI.detail(scheduleId);
      expect(detail.enabled).toBe(0);
    });

    test("T-SC-061 正向：重新启用 enabled=1", async () => {
      await asVip2();
      await AiScheduleAPI.setStatus(scheduleId, createScheduleStatusForm({ enabled: 1 }));
      const detail = await AiScheduleAPI.detail(scheduleId);
      expect(detail.enabled).toBe(1);
    });
  });

  describe("POST /api/v1/ai/scheduled-tasks/{id}/run - 手动触发", () => {
    test("T-SC-062 正向：受理返回 {accepted: true}", async () => {
      await asVip2();
      const result = await AiScheduleAPI.run(scheduleId);
      expect(result.accepted).toBe(true);
    });
  });

  describe("GET /api/v1/ai/scheduled-tasks/{id}/history - 执行历史", () => {
    test("T-SC-056 正向：分页结构", async () => {
      await asVip2();
      const result = await AiScheduleAPI.history(scheduleId, { pageNum: 1, pageSize: 20 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });
  });

  describe("DELETE /api/v1/ai/scheduled-tasks/{id} - 删除任务", () => {
    test("T-SC-072 正向：软删除任务", async () => {
      await asVip2();
      const created = await AiScheduleAPI.create(createScheduleForm());
      expect(created.id).toBeGreaterThan(0);
      await AiScheduleAPI.delete(created.id);
      await expectBizError(AiScheduleAPI.detail(created.id), ["A0401"]);
    });

    test("T-SC-072 边界：删除不存在任务 → A0401", async () => {
      await asVip2();
      await expectBizError(AiScheduleAPI.delete(99999999), ["A0401"]);
    });
  });
});
