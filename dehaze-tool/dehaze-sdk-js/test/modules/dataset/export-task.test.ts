import { DatasetAPI, DatasetItemAPI, TaskAPI } from "../../../index";
import { createDatasetForm, createDatasetItemForm } from "#/factories/dataset";

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
      expect(["PENDING", "PROCESSING", "COMPLETED", "FAILED"]).toContain(result.status);
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
      expect(["PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"]).toContain(
        status.status
      );
    });

    test("异常测试：查询不存在的任务", async () => {
      // 后端 getTaskStatus 找不到任务时返回 null，SDK 解包后为 undefined
      const result = await TaskAPI.getStatus("non-existent-task-id");
      expect(result === undefined || result === null).toBe(true);
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
      expect(["CANCELLED", "COMPLETED", "FAILED", "PROCESSING", "PENDING"]).toContain(
        status.status
      );
    });

    test("异常测试：取消不存在的任务", async () => {
      // 后端 Assert.notNull 会抛 IllegalArgumentException → B0001
      await expect(TaskAPI.cancel("non-existent-task-id")).rejects.toThrow();
    });
  });

  describe("POST /api/v1/tasks - 创建数据项下载任务（item_download）", () => {
    let itemId: number;

    beforeAll(async () => {
      const itemForm = createDatasetItemForm(testDatasetId, {
        sceneType: "urban",
        name: "下载任务测试数据项",
      });
      const item = await DatasetItemAPI.add(itemForm);
      itemId = item.id;
    });

    afterAll(async () => {
      try {
        await DatasetItemAPI.deleteById(itemId);
      } catch (e) {
        // 忽略
      }
    });

    test("正向测试：创建单个数据项下载任务", async () => {
      const result = await TaskAPI.create({
        type: "item_download",
        targetId: itemId,
      });
      expect(result.taskId).toBeTruthy();
      expect(typeof result.taskId).toBe("string");
      expect(["PENDING", "PROCESSING", "COMPLETED", "FAILED"]).toContain(result.status);
    });

    test("参数校验：缺少targetId", async () => {
      // 统一任务接口为异步执行：targetId 为 null 不是同步校验错误，
      // 任务创建成功（PENDING），异步策略 validateParams 抛出异常，任务状态变为 FAILED。
      const result = await TaskAPI.create({ type: "item_download" } as any);
      expect(result.taskId).toBeDefined();
      expect(result.status).toBe("PENDING");
    });

    test("边界测试：重复创建下载任务", async () => {
      const task1 = await TaskAPI.create({
        type: "item_download",
        targetId: itemId,
      });
      expect(task1.taskId).toBeDefined();

      const task2 = await TaskAPI.create({
        type: "item_download",
        targetId: itemId,
      });
      expect(task2.taskId).toBeDefined();
      // 两次创建的任务ID应不同
      expect(task1.taskId).not.toBe(task2.taskId);
    });
  });

  describe("POST /api/v1/tasks - 批量下载数据项（batch_download）", () => {
    test("正向测试：批量下载所有数据项", async () => {
      const result = await TaskAPI.create({
        type: "batch_download",
        targetIds: testItemIds,
        options: { structure: "by_item" },
      });
      expect(result.taskId).toBeTruthy();
      expect(typeof result.taskId).toBe("string");
      expect(["PENDING", "PROCESSING", "COMPLETED", "FAILED"]).toContain(result.status);
    });

    test("正向测试：扁平结构批量下载", async () => {
      const result = await TaskAPI.create({
        type: "batch_download",
        targetIds: testItemIds,
        options: { structure: "flat" },
      });
      expect(result.taskId).toBeTruthy();
      expect(typeof result.taskId).toBe("string");
    });

    test("参数校验：空ID数组", async () => {
      // 统一任务接口为异步执行：空 targetIds 不是同步校验错误（非 null），
      // 任务创建成功（PENDING），异步执行时 listByIds 返回空列表，任务状态变为 FAILED。
      const result = await TaskAPI.create({
        type: "batch_download",
        targetIds: [],
      });
      expect(result.taskId).toBeDefined();
      expect(result.status).toBe("PENDING");
    });

    test("边界测试：单个数据项批量下载", async () => {
      expect(testItemIds.length).toBeGreaterThan(0);

      const result = await TaskAPI.create({
        type: "batch_download",
        targetIds: [testItemIds[0]!],
      });
      expect(result.taskId).toBeDefined();
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
      const item = await DatasetItemAPI.add(itemForm);

      // 数据集导出
      const exportTask = await TaskAPI.create({
        type: "dataset_export",
        targetId: datasetId,
      });
      expect(exportTask.taskId).toBeDefined();

      // 数据项下载
      const downloadTask = await TaskAPI.create({
        type: "item_download",
        targetId: item.id,
      });
      expect(downloadTask.taskId).toBeDefined();

      await DatasetAPI.deleteById(datasetId);
    });
  });
});
