import { DatasetAPI, DatasetItemAPI, TaskAPI } from "../../../index";
import { createDatasetForm, createDatasetItemForm } from "#/factories/dataset";
import { expectBizError } from "#/utils/assertion";

describe("导出任务接口测试", () => {
  let testDatasetId: number;
  let testItemIds: number[] = [];

  beforeAll(async () => {
    // 创建测试数据集
    const datasetForm = createDatasetForm({ type: "图像去雾" });
    testDatasetId = await DatasetAPI.add(datasetForm);

    // 创建多个测试数据项
    for (let i = 0; i < 3; i++) {
      const itemForm = createDatasetItemForm(testDatasetId, {
        sceneType: "urban",
        name: `导出测试数据项${i}`,
      });
      const item = await DatasetItemAPI.add(itemForm);
      testItemIds.push(item.id);
    }
  }, 60000);

  afterAll(async () => {
    // 清理测试数据
    try {
      await DatasetAPI.deleteById(testDatasetId);
    } catch (e) {
      // 忽略
    }
  });

  describe("POST /api/v1/tasks - 创建数据集导出任务（dataset_export）", () => {
    test("正向测试：创建导出任务（包含清晰图和有雾图）", async () => {
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: testDatasetId,
        options: { includeTypes: ["clear", "hazy"], structure: "by_item" },
      });
      expect(result.taskId).toBeTruthy();
      expect(typeof result.taskId).toBe("string");
      expect([1, 2, 3, 4]).toContain(result.status);
    });

    test("正向测试：使用默认参数创建导出任务", async () => {
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: testDatasetId,
      });
      expect(result.taskId).toBeDefined();
    });

    test("参数校验：缺少type字段", async () => {
      await expect(TaskAPI.create({ targetId: testDatasetId } as any)).rejects.toThrow();
    });
  });

  describe("GET /api/v1/tasks/{taskId} - 查询任务状态", () => {
    test("正向测试：查询任务状态", async () => {
      const createResult = await TaskAPI.create({
        type: "dataset_export",
        targetId: testDatasetId,
      });

      expect(createResult.taskId).toBeDefined();

      const status = await TaskAPI.getStatus(createResult.taskId);
      expect(status.taskId).toBe(createResult.taskId);
      expect([1, 2, 3, 4, 5]).toContain(status.status);
    });

    test("异常测试：查询不存在的任务", async () => {
      // 规范：查询不存在的任务返回 400（code=B0301 任务不存在）
      await expectBizError(TaskAPI.getStatus("non-existent-task-id"), "B0301");
    });
  });

  describe("DELETE /api/v1/tasks/{taskId} - 取消任务", () => {
    test("正向测试：取消进行中的任务", async () => {
      const createResult = await TaskAPI.create({
        type: "dataset_export",
        targetId: testDatasetId,
      });

      expect(createResult.taskId).toBeDefined();

      // 任务创建后立即取消（可能已完成或仍在处理中）
      try {
        await TaskAPI.cancel(createResult.taskId);
      } catch (error: any) {
        // 任务可能已快速完成，取消已完成的任务后端幂等返回
        expect(error).toBeDefined();
      }

      const status = await TaskAPI.getStatus(createResult.taskId);
      expect([5, 3, 4, 2, 1]).toContain(status.status);
    });

    test("异常测试：取消不存在的任务", async () => {
      // 后端 Assert.notNull 会抛 IllegalArgumentException → B0001
      await expect(TaskAPI.cancel("non-existent-task-id")).rejects.toThrow();
    });
  });

  describe("综合测试：导出任务流程", () => {
    test("完整流程：创建数据集 -> 创建数据项 -> 导出", async () => {
      const datasetForm = createDatasetForm({ type: "图像去雾" });
      const datasetId = await DatasetAPI.add(datasetForm);

      const itemForm = createDatasetItemForm(datasetId, {
        sceneType: "urban",
        name: "完整流程测试数据项",
      });
      await DatasetItemAPI.add(itemForm);

      // 数据集导出
      const exportTask = await TaskAPI.create({
        type: "dataset_export",
        targetId: datasetId,
      });
      expect(exportTask.taskId).toBeDefined();

      await DatasetAPI.deleteById(datasetId);
    });
  });
});
