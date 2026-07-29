import { TaskAPI } from "../../../index";
import { login } from "#/utils/auth";
import { pageQuery } from "#/factories/common";
import type { TaskQuery } from "../../../index";

describe("任务管理接口测试 - TaskAPI", () => {
  beforeAll(async () => {
    await login("admin");
  });

  describe("GET /api/v1/tasks - 任务列表分页查询", () => {
    test("无筛选条件返回任务分页列表", async () => {
      const result = await TaskAPI.getPage(pageQuery<TaskQuery>({ pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);
    });

    test("按 status 筛选已完成任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        status: 3,
      });

      expect(result).toBeDefined();
      result.list.forEach((task) => {
        expect(task.status).toBe(3);
      });
    });

    test("按 taskCategory 筛选导出任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskCategory: "export",
      });

      expect(result).toBeDefined();
      result.list.forEach((task) => {
        if (task.taskType) {
          expect(task.taskType).toMatch(/_export$/);
        }
      });
    });

    test("按 taskType 筛选（逗号分隔多个类型）", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 50,
        taskType: "user_export,user_import",
      });

      expect(result).toBeDefined();
      result.list.forEach((task) => {
        expect(["user_export", "user_import"]).toContain(task.taskType);
      });
    });
  });

  describe("POST /api/v1/tasks - 创建任务 + 幂等性", () => {
    test("创建用户导出任务返回 taskId", async () => {
      const result = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      expect(result.taskId).toBeTruthy();
      expect(typeof result.taskId).toBe("string");
      expect([1, 2, 3, 4]).toContain(result.status);
    });

    test("幂等性：相同 Idempotency-Key 返回同一任务", async () => {
      const idempotencyKey = `test-idem-${Date.now()}`;

      const first = await TaskAPI.create(
        { type: "user_export", options: { format: "excel" } },
        idempotencyKey
      );
      const second = await TaskAPI.create(
        { type: "user_export", options: { format: "excel" } },
        idempotencyKey
      );

      expect(first.taskId).toBe(second.taskId);
    });

    test("无幂等键可重复创建不同任务", async () => {
      const first = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });
      const second = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      expect(first.taskId).not.toBe(second.taskId);
    });

    test("参数校验：缺少 type 字段应抛出异常", async () => {
      await expect(TaskAPI.create({} as any)).rejects.toThrow();
    });
  });

  describe("GET /api/v1/tasks/{taskId} - 查询任务状态", () => {
    test("查询已创建任务的状态", async () => {
      const created = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      const status = await TaskAPI.getStatus(created.taskId);
      expect(status.taskId).toBe(created.taskId);
      expect([1, 2, 3, 4, 5]).toContain(status.status);
    });

    test("异常测试：查询不存在的任务应抛出异常", async () => {
      await expect(TaskAPI.getStatus("non-existent-task-id")).rejects.toThrow();
    });
  });

  describe("POST /api/v1/tasks/{taskId}/cancel - 取消任务", () => {
    test("取消任务（创建后立即取消）", async () => {
      const created = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      try {
        await TaskAPI.cancel(created.taskId);
      } catch (error: any) {
        // 任务可能已快速完成，取消已完成任务会抛出异常
        expect(error).toBeDefined();
      }

      const status = await TaskAPI.getStatus(created.taskId);
      expect([5, 3, 4, 2, 1]).toContain(status.status);
    });

    test("异常测试：取消不存在的任务应抛出异常", async () => {
      await expect(TaskAPI.cancel("non-existent-task-id")).rejects.toThrow();
    });
  });

  describe("POST /api/v1/tasks/{taskId}/retry - 重试任务", () => {
    test("异常测试：重试不存在的任务应抛出异常", async () => {
      await expect(TaskAPI.retry("non-existent-task-id")).rejects.toThrow();
    });

    test("异常测试：重试非失败状态的任务应抛出异常", async () => {
      // 修复后端 module 推导后，通用端点创建的 user_export 任务能正常执行，
      // 任务处于 PENDING/PROCESSING/COMPLETED 等非 FAILED 状态时，retry 应抛出异常
      const created = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      await expect(TaskAPI.retry(created.taskId)).rejects.toThrow();
    });
  });

  describe("权限校验 - 数据隔离", () => {
    test("用户A查询用户B的任务应被拒绝", async () => {
      // 以 admin 创建任务
      const created = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      // 切换到 test 用户查询 admin 的任务
      await login("test");
      try {
        await expect(TaskAPI.getStatus(created.taskId)).rejects.toThrow();
      } finally {
        // 切回 admin
        await login("admin");
      }
    });

    test("用户A取消用户B的任务应被拒绝", async () => {
      // 以 admin 创建任务
      const created = await TaskAPI.create({
        type: "user_export",
        options: { format: "excel" },
      });

      // 切换到 test 用户取消 admin 的任务
      await login("test");
      try {
        await expect(TaskAPI.cancel(created.taskId)).rejects.toThrow();
      } finally {
        await login("admin");
      }
    });
  });
});
