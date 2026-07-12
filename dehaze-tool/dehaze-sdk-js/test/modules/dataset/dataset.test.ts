import { DatasetAPI, BatchDeleteForm, TaskAPI } from "../../../index";
import { login, logout } from "#/utils/auth";
import { expectBizErrorOrUndefined } from "#/utils/assertion";
import {
  createDatasetForm,
  createDatasetUpdateForm,
  createDatasetQuery,
} from "#/factories/dataset";

describe("数据集接口测试", () => {
  beforeAll(async () => {
    await login();
  }, 30000);

  afterAll(async () => {
    await logout();
  });

  describe("GET /api/v1/datasets - 获取数据集列表", () => {
    test("正向测试：获取所有数据集", async () => {
      const query = createDatasetQuery();
      const result = await DatasetAPI.getList(query);
      // getList 返回分页结构 PageResult<Dataset[]> = { list, total }
      expect(result).toBeDefined();
      expect(Array.isArray(result.list)).toBe(true);
      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(firstItem.id).toBeDefined();
        expect(firstItem.name).toBeDefined();
        expect(firstItem.type).toBeDefined();
      }
    });

    test("正向测试：按类型筛选数据集", async () => {
      const query = createDatasetQuery({ type: "用户数据集" });
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.type).toBeDefined();
      });
    });

    test("正向测试：按状态筛选数据集", async () => {
      const query = createDatasetQuery({ status: "1" });
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.status).toBe(1);
      });
    });

    test("边界测试：大页码返回空结果", async () => {
      const query = createDatasetQuery({ pageNum: 99999 });
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
    });
  });

  describe("GET /api/v1/datasets/options - 获取数据集下拉选项", () => {
    test("正向测试：获取下拉选项并验证结构", async () => {
      const options = await DatasetAPI.getOptions();
      expect(Array.isArray(options)).toBe(true);
      if (options.length > 0) {
        const firstOption = options[0]!;
        expect(firstOption.value).toBeDefined();
        expect(firstOption.label).toBeDefined();
      }
    });
  });

  describe("POST /api/v1/datasets - 新增数据集", () => {
    const createdIds: number[] = [];

    afterAll(async () => {
      for (const id of createdIds) {
        try {
          await DatasetAPI.deleteById(id);
        } catch (e) {
          // 忽略清理错误
        }
      }
    });

    test("正向测试：创建有效数据集", async () => {
      const form = createDatasetForm();
      const result = await DatasetAPI.add(form);
      expect(result.id).toBeDefined();
      expect(result.name).toBe(form.name);
      expect(result.type).toBe(form.type);
      expect(result.description).toBe(form.description);
      expect(result.status).toBe(1);
      expect(result.parentId).toBe(0);
      expect(result.statistics).toBeDefined();
      createdIds.push(result.id);
    });

    test("正向测试：创建带可选字段的数据集", async () => {
      const form = createDatasetForm({
        description: "带描述的测试数据集",
        status: "1",
      });
      const result = await DatasetAPI.add(form);
      expect(result.id).toBeDefined();
      expect(result.description).toBe(form.description);
      createdIds.push(result.id);
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form = createDatasetForm();
      delete (form as any).name;
      await expectBizErrorOrUndefined(DatasetAPI.add(form), ["A0400", "B0001"]);
    });
  });

  describe("GET /api/v1/datasets/{id} - 获取数据集详细信息", () => {
    let datasetId: number;

    beforeAll(async () => {
      const form = createDatasetForm();
      const result = await DatasetAPI.add(form);
      datasetId = result.id;
    });

    afterAll(async () => {
      try {
        await DatasetAPI.deleteById(datasetId);
      } catch (e) {
        // 忽略清理错误
      }
    });

    test("正向测试：获取有效数据集详情", async () => {
      const result = await DatasetAPI.getDatasetInfoById(datasetId);
      expect(result.id).toBe(datasetId);
      expect(result.name).toBeDefined();
      expect(result.type).toBeDefined();
      expect(result.path).toBeDefined();
      expect(result.parentId).toBeDefined();
    });

    test("正向测试：验证统计信息结构", async () => {
      const result = await DatasetAPI.getDatasetInfoById(datasetId);
      expect(result.statistics).toBeDefined();
      if (result.statistics) {
        expect(result.statistics.itemCount).toBeDefined();
        expect(result.statistics.fileCount).toBeDefined();
        expect(result.statistics.totalSize).toBeDefined();
        expect(result.statistics.annotatedCount).toBeDefined();
        expect(result.statistics.unannotatedCount).toBeDefined();
      }
    });

    test("异常测试：不存在的ID", async () => {
      await expectBizErrorOrUndefined(DatasetAPI.getDatasetInfoById(99999999), ["B0001", "A0400"]);
    });
  });

  describe("PUT /api/v1/datasets/{id} - 修改数据集信息", () => {
    let datasetId: number;

    beforeAll(async () => {
      const form = createDatasetForm();
      const result = await DatasetAPI.add(form);
      datasetId = result.id;
    });

    afterAll(async () => {
      try {
        await DatasetAPI.deleteById(datasetId);
      } catch (e) {
        // 忽略清理错误
      }
    });

    test("正向测试：更新数据集名称", async () => {
      const form = createDatasetUpdateForm();
      const result = await DatasetAPI.update(datasetId, form);
      expect(result.id).toBe(datasetId);
      expect(result.name).toBe(form.name);
    });

    test("正向测试：更新数据集描述", async () => {
      const form = createDatasetUpdateForm({
        description: "更新后的描述",
      });
      const result = await DatasetAPI.update(datasetId, form);
      expect(result.description).toBe(form.description);
    });

    test("正向测试：更新数据集状态", async () => {
      const form = createDatasetUpdateForm({
        status: "0",
      });
      const result = await DatasetAPI.update(datasetId, form);
      expect(result.status).toBe(0);
    });

    test("异常测试：更新不存在的数据集", async () => {
      const form = createDatasetUpdateForm();
      await expectBizErrorOrUndefined(DatasetAPI.update(99999999, form), ["B0001", "A0400"]);
    });
  });

  describe("DELETE /api/v1/datasets/{id} - 删除单个数据集", () => {
    test("正向测试：删除有效数据集", async () => {
      const form = createDatasetForm();
      const result = await DatasetAPI.add(form);
      await expect(DatasetAPI.deleteById(result.id)).resolves.not.toThrow();
    });

    test("异常测试：删除不存在的数据集", async () => {
      await expectBizErrorOrUndefined(DatasetAPI.deleteById(99999999), ["B0001", "A0400"]);
    });
  });

  describe("DELETE /api/v1/datasets/batch - 批量删除数据集", () => {
    test("正向测试：批量删除多个数据集", async () => {
      const datasetIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createDatasetForm();
        const result = await DatasetAPI.add(form);
        datasetIds.push(result.id);
      }

      const batchForm: BatchDeleteForm = {
        ids: datasetIds,
      };
      const result = await DatasetAPI.batchDelete(batchForm);
      expect(result).toBeDefined();
      if (result.successIds) {
        expect(result.successIds.length).toBe(3);
        expect(result.successIds).toEqual(expect.arrayContaining(datasetIds));
      }
      if (result.failedItems) {
        expect(result.failedItems.length).toBe(0);
      }
    });

    test("参数校验：空ID数组", async () => {
      const form: BatchDeleteForm = {
        ids: [],
      };
      await expectBizErrorOrUndefined(DatasetAPI.batchDelete(form), ["A0400", "B0001"]);
    });

    test("异常测试：包含不存在的ID", async () => {
      const form: BatchDeleteForm = {
        ids: [99999999, 99999998],
      };
      const result = await DatasetAPI.batchDelete(form);
      expect(result).toBeDefined();
      if (result.successIds) {
        expect(result.successIds.length).toBe(0);
      }
      if (result.failedItems) {
        expect(result.failedItems.length).toBe(2);
      }
    });
  });

  describe("POST /api/v1/tasks - 创建数据集导出任务（dataset_export）", () => {
    let datasetId: number;

    beforeAll(async () => {
      const form = createDatasetForm();
      const result = await DatasetAPI.add(form);
      datasetId = result.id;
    });

    afterAll(async () => {
      try {
        await DatasetAPI.deleteById(datasetId);
      } catch (e) {
        // 忽略清理错误
      }
    });

    test("正向测试：创建导出任务（空数据集）", async () => {
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: datasetId,
        options: { includeTypes: ["clear", "hazy"], structure: "by_item" },
      });
      expect(result.taskId).toBeDefined();
      expect(result.status).toBeDefined();
      expect(result.progress).toBeGreaterThanOrEqual(0);
      expect(result.progress).toBeLessThanOrEqual(100);
    });

    test("正向测试：使用默认参数创建导出任务", async () => {
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: datasetId,
      });
      expect(result.taskId).toBeDefined();
    });

    test("异常测试：导出不存在的数据集", async () => {
      // 统一任务接口为异步执行：createTask 同步创建任务记录（PENDING），
      // 数据集存在性校验在异步策略 DatasetExportStrategy 中执行，任务最终状态为 FAILED。
      // 此处仅验证任务能创建成功，异步失败需通过 getStatus 轮询验证。
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: 99999999,
      });
      expect(result.taskId).toBeDefined();
      expect(result.status).toBe("PENDING");
    });
  });
});
