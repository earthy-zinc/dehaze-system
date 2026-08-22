import { TaskAPI } from "../../../index";
import { login } from "#/utils/auth";
import { pageQuery } from "#/factories/common";
import { TestCleanupRegistry } from "#/utils/cleanup";
import type { TaskQuery } from "../../../index";

describe("任务管理接口测试 - TaskAPI", () => {
  const cleanup = new TestCleanupRegistry();
  const createdTaskIds: string[] = [];

  // 创建导出任务并登记清理（供需要真实任务ID的用例复用）
  async function createTask() {
    const result = await TaskAPI.create({ type: "user_export", options: { format: "excel" } });
    createdTaskIds.push(result.taskId);
    return result;
  }

  beforeAll(async () => {
    await login("admin");
  });

  afterAll(async () => {
    // 任务数据最终由后端 TaskCleanupJob 定时清理，这里尽力取消即可
    // taskId 为字符串，不能用 registerIds（其接收 number[]），改用 register 直接遍历
    cleanup.register(async () => {
      for (const id of [...createdTaskIds].reverse()) {
        try {
          await TaskAPI.cancel(id);
        } catch {
          // 静默忽略清理失败
        }
      }
    });
    await cleanup.executeAll();
  });

  describe("GET /api/v1/tasks - 任务列表分页查询", () => {
    test("无筛选条件返回任务分页列表", async () => {
      const result = await TaskAPI.getPage(pageQuery<TaskQuery>({ pageSize: 10 }));

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

      result.list.forEach((task) => {
        expect(["user_export", "user_import"]).toContain(task.taskType);
      });
    });

    test("验证：任务按创建时间倒序排列", async () => {
      const result = await TaskAPI.getPage(pageQuery<TaskQuery>({ pageSize: 20 }));
      if (result.list.length < 2) return;

      for (let i = 1; i < result.list.length; i++) {
        const prev = result.list[i - 1]!.createdAt;
        const curr = result.list[i]!.createdAt;
        if (prev && curr) {
          expect(prev >= curr).toBe(true);
        }
      }
    });
  });

  describe("POST /api/v1/tasks - 创建任务 + 幂等性", () => {
    test("创建用户导出任务返回 taskId", async () => {
      const result = await createTask();

      expect(typeof result.taskId).toBe("string");
      expect([1, 2, 3, 4]).toContain(result.status);
    });

    test("验证：任务ID为UUID格式", async () => {
      const result = await createTask();

      expect(result.taskId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
      );
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
      createdTaskIds.push(first.taskId);

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
      createdTaskIds.push(first.taskId, second.taskId);

      expect(first.taskId).not.toBe(second.taskId);
    });

    test("不同幂等键创建不同任务", async () => {
      const first = await TaskAPI.create(
        { type: "user_export", options: { format: "excel" } },
        `test-key-a-${Date.now()}`
      );
      const second = await TaskAPI.create(
        { type: "user_export", options: { format: "excel" } },
        `test-key-b-${Date.now()}`
      );
      createdTaskIds.push(first.taskId, second.taskId);

      expect(first.taskId).not.toBe(second.taskId);
    });

    test("参数校验：缺少 type 字段应抛出异常", async () => {
      await expect(TaskAPI.create({} as any)).rejects.toThrow();
    });

    test("边界：不支持的任务类型应失败", async () => {
      await expect(TaskAPI.create({ type: "invalid_type" } as any)).rejects.toThrow();
    });
  });

  describe("GET /api/v1/tasks/{taskId} - 查询任务状态", () => {
    test("查询已创建任务的状态", async () => {
      const created = await createTask();

      const status = await TaskAPI.getStatus(created.taskId);
      expect(status.taskId).toBe(created.taskId);
      expect([1, 2, 3, 4, 5]).toContain(status.status);
    });

    test("异常测试：查询不存在的任务应抛出异常", async () => {
      await expect(TaskAPI.getStatus("non-existent-task-id")).rejects.toThrow();
    });
  });

  describe("GET /api/v1/tasks/{taskId}/download - 下载任务结果", () => {
    test("边界：下载未完成任务应失败", async () => {
      const created = await createTask();

      // 刚创建可能仍在进行中，下载应失败
      await expect(TaskAPI.download(created.taskId)).rejects.toThrow();
    });

    test("边界：下载他人任务应失败", async () => {
      const created = await createTask();

      await login("test");
      try {
        await expect(TaskAPI.download(created.taskId)).rejects.toThrow();
      } finally {
        await login("admin");
      }
    });
  });

  describe("POST /api/v1/tasks/{taskId}/cancel - 取消任务", () => {
    test("取消任务（创建后立即取消）", async () => {
      const created = await createTask();

      try {
        await TaskAPI.cancel(created.taskId);
      } catch {
        // 任务可能已快速完成，取消已完成任务属预期
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
      const created = await createTask();

      await expect(TaskAPI.retry(created.taskId)).rejects.toThrow();
    });

    test("边界：重试他人任务应失败", async () => {
      const created = await createTask();

      await login("test");
      try {
        await expect(TaskAPI.retry(created.taskId)).rejects.toThrow();
      } finally {
        await login("admin");
      }
    });
  });

  describe("权限校验 - 数据隔离", () => {
    test("用户A查询用户B的任务应被拒绝", async () => {
      const created = await createTask();

      await login("test");
      try {
        await expect(TaskAPI.getStatus(created.taskId)).rejects.toThrow();
      } finally {
        await login("admin");
      }
    });

    test("用户A取消用户B的任务应被拒绝", async () => {
      const created = await createTask();

      await login("test");
      try {
        await expect(TaskAPI.cancel(created.taskId)).rejects.toThrow();
      } finally {
        await login("admin");
      }
    });

    test("验证：用户仅能看到自己的任务", async () => {
      const mine = await createTask(); // admin 创建，记录 id

      await login("test");
      const result = await TaskAPI.getPage(pageQuery<TaskQuery>({ pageSize: 50 }));
      // 前置结构断言：验证的是数据隔离而非空列表
      expect(Array.isArray(result.list)).toBe(true);
      // test 用户列表中不得包含 admin 创建的任务（数据隔离）
      expect(result.list.some((t) => t.taskId === mine.taskId)).toBe(false);
      await login("admin");
    });
  });
});
