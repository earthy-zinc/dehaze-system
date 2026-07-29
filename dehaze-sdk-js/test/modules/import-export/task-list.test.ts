import { TaskAPI, TaskQuery } from "../../../index";
import { pageQuery } from "#/factories/common";

describe("任务列表筛选接口测试 - TaskAPI.getPage", () => {
  describe("GET /api/v1/tasks - 基础分页查询", () => {
    test("无筛选条件返回任务分页列表", async () => {
      const result = await TaskAPI.getPage(pageQuery<TaskQuery>({ pageSize: 10 }));

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);
    });
  });

  describe("taskCategory 筛选", () => {
    test("筛选 export 类别任务返回导出任务列表", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskCategory: "export",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        if (task.taskType) {
          expect(task.taskType).toMatch(/_export$/);
        }
      });
    });

    test("筛选 import 类别任务返回导入任务列表", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskCategory: "import",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        if (task.taskType) {
          expect(task.taskType).toMatch(/_import$/);
        }
      });
    });
  });

  describe("taskType 筛选", () => {
    test("筛选 user_export 任务类型返回用户导出任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskType: "user_export",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.taskType).toBe("user_export");
      });
    });

    test("筛选 user_import 任务类型返回用户导入任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskType: "user_import",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.taskType).toBe("user_import");
      });
    });

    test("筛选 dataset_export 任务类型返回数据集导出任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskType: "dataset_export",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.taskType).toBe("dataset_export");
      });
    });

    test("逗号分隔筛选多个任务类型 user_export,user_import", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 50,
        taskType: "user_export,user_import",
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(["user_export", "user_import"]).toContain(task.taskType);
      });
    });
  });

  describe("status 筛选", () => {
    test("筛选 COMPLETED 状态任务返回已完成任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        status: 3,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.status).toBe(3);
      });
    });

    test("筛选 PENDING 状态任务返回待执行任务", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        status: 1,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.status).toBe(1);
      });
    });
  });

  describe("组合筛选", () => {
    test("taskCategory=export + status=COMPLETED 组合筛选", async () => {
      const result = await TaskAPI.getPage({
        pageNum: 1,
        pageSize: 20,
        taskCategory: "export",
        status: 3,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((task) => {
        expect(task.status).toBe(3);
        if (task.taskType) {
          expect(task.taskType).toMatch(/_export$/);
        }
      });
    });
  });
});
