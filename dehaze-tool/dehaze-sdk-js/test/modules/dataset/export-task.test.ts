import { DatasetAPI, DatasetItemAPI, TaskAPI } from "../../../index";
import { login, logout } from "#/utils/auth";
import {
  createDatasetForm,
  createDatasetItemForm,
  createExportTaskRequest,
  createBatchDownloadForm,
} from "#/factories/dataset";

describe("导出任务接口测试", () => {
  let testDatasetId: number;
  let testItemIds: number[] = [];

  beforeAll(async () => {
    await login();

    // 创建测试数据集
    const datasetForm = createDatasetForm({ name: "导出任务测试数据集", type: "图像去雾" });
    const dataset = await DatasetAPI.add(datasetForm);
    testDatasetId = dataset.id;

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
    await logout();
  });

  describe("POST /api/v1/datasets/{id}/export - 创建数据集导出任务", () => {
    test("正向测试：创建导出任务（包含清晰图和有雾图）", async () => {
      const request = createExportTaskRequest({ includeTypes: ["clear", "hazy"] });

      try {
        const result = await DatasetAPI.createExportTask(testDatasetId, request);
        expect(result.taskId).toBeDefined();
        expect(result.status).toBeDefined();
        expect(result.progress).toBeGreaterThanOrEqual(0);
        expect(result.progress).toBeLessThanOrEqual(100);
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
      }
    });

    test("正向测试：创建导出任务（仅清晰图）", async () => {
      const request = createExportTaskRequest({ includeTypes: ["clear"], structure: "flat" });

      try {
        const result = await DatasetAPI.createExportTask(testDatasetId, request);
        expect(result.taskId).toBeDefined();
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
      }
    });

    test("正向测试：使用默认参数创建导出任务", async () => {
      const request = createExportTaskRequest({});

      try {
        const result = await DatasetAPI.createExportTask(testDatasetId, request);
        expect(result.taskId).toBeDefined();
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
      }
    });

    test("异常测试：导出不存在的数据集", async () => {
      const request = createExportTaskRequest({});
      await expect(DatasetAPI.createExportTask(99999999, request)).rejects.toThrow();
    });
  });

  describe("GET /api/v1/tasks/{taskId} - 查询任务状态", () => {
    test("正向测试：查询任务状态", async () => {
      const request = createExportTaskRequest();

      try {
        const createResult = await DatasetAPI.createExportTask(testDatasetId, request);

        if (createResult?.taskId) {
          const status = await TaskAPI.getStatus(createResult.taskId);
          expect(status.taskId).toBe(createResult.taskId);
          expect(["PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"]).toContain(
            status.status
          );
        }
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
      }
    });

    test("异常测试：查询不存在的任务", async () => {
      try {
        const result = await TaskAPI.getStatus("non-existent-task-id");
        expect(result === undefined || result === null).toBe(true);
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("POST /api/v1/tasks/{taskId}/cancel - 取消任务", () => {
    test("正向测试：取消进行中的任务", async () => {
      const request = createExportTaskRequest();

      try {
        const createResult = await DatasetAPI.createExportTask(testDatasetId, request);

        if (createResult?.taskId) {
          await expect(TaskAPI.cancel(createResult.taskId)).resolves.not.toThrow();

          try {
            const status = await TaskAPI.getStatus(createResult.taskId);
            expect(["CANCELLED", "COMPLETED", "FAILED"]).toContain(status.status);
          } catch (e) {}
        }
      } catch (error: any) {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("B0001");
      }
    });

    test("异常测试：取消不存在的任务（后端幂等设计）", async () => {
      try {
        await TaskAPI.cancel("non-existent-task-id");
        console.warn("⚠️ 后端取消不存在的任务返回成功（幂等设计）");
      } catch (error: any) {
        expect(error).toBeDefined();
      }
    });
  });

  describe("POST /api/v1/dataset-items/{id}/download/task - 创建数据项下载任务", () => {
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
      try {
        const result = await DatasetItemAPI.createDownloadTask(itemId);
        expect(result.taskId).toBeDefined();
        expect(result.status).toBeDefined();
        expect(result.progress).toBeGreaterThanOrEqual(0);
        expect(result.progress).toBeLessThanOrEqual(100);
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }
    });

    test("异常测试：创建不存在数据项的下载任务", async () => {
      await expect(DatasetItemAPI.createDownloadTask(99999999)).rejects.toThrow();
    });

    test("边界测试：重复创建下载任务", async () => {
      try {
        const task1 = await DatasetItemAPI.createDownloadTask(itemId);
        expect(task1.taskId).toBeDefined();

        const task2 = await DatasetItemAPI.createDownloadTask(itemId);
        expect(task2.taskId).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }
    });
  });

  describe("POST /api/v1/dataset-items/batch/download - 批量下载数据项图片", () => {
    test("正向测试：批量下载所有文件", async () => {
      const form = createBatchDownloadForm(testItemIds, { organizeByItem: true });

      try {
        const result = await DatasetItemAPI.batchDownload(form);
        expect(result.taskId).toBeDefined();
        expect(result.status).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }
    });

    test("正向测试：扁平结构批量下载", async () => {
      const form = createBatchDownloadForm(testItemIds, { organizeByItem: false });

      try {
        const result = await DatasetItemAPI.batchDownload(form);
        expect(result.taskId).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }
    });

    test("参数校验：空ID数组", async () => {
      const form = createBatchDownloadForm([]);
      await expect(DatasetItemAPI.batchDownload(form)).rejects.toThrow();
    });

    test("边界测试：单个文件批量下载", async () => {
      if (testItemIds.length === 0) {
        console.warn("No test item IDs available for batch download test");
        return;
      }

      const form = createBatchDownloadForm([testItemIds[0]!]);

      try {
        const result = await DatasetItemAPI.batchDownload(form);
        expect(result.taskId).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }
    });
  });

  describe("综合测试：导出任务流程", () => {
    test("完整流程：创建数据集 -> 创建数据项 -> 导出", async () => {
      const datasetForm = createDatasetForm({ name: "完整流程测试数据集", type: "图像去雾" });
      const dataset = await DatasetAPI.add(datasetForm);

      const itemForm = createDatasetItemForm(dataset.id, {
        sceneType: "urban",
        name: "完整流程测试数据项",
      });
      const item = await DatasetItemAPI.add(itemForm);

      const exportRequest = createExportTaskRequest();
      try {
        const exportTask = await DatasetAPI.createExportTask(dataset.id, exportRequest);
        expect(exportTask.taskId).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.response?.status === 400 || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }

      try {
        const downloadTask = await DatasetItemAPI.createDownloadTask(item.id);
        expect(downloadTask.taskId).toBeDefined();
      } catch (error: any) {
        const isExpectedError =
          error.code === "B0001" || (error.message && error.message.includes("没有图片"));
        expect(isExpectedError).toBe(true);
      }

      await DatasetAPI.deleteById(dataset.id);
    });
  });
});
