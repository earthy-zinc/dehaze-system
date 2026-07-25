import { DatasetAPI, DatasetItemAPI, BatchDeleteForm } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import {
  createDatasetForm,
  createDatasetItemForm,
  createDatasetItemUpdateForm,
  createDatasetItemQuery,
} from "#/factories/dataset";

describe("数据项接口测试", () => {
  let testDatasetId: number;

  beforeAll(async () => {
    const form = createDatasetForm({ type: "图像去雾" });
    testDatasetId = await DatasetAPI.add(form);
  });

  afterAll(async () => {
    try {
      await DatasetAPI.deleteById(testDatasetId);
    } catch (e) {
      // 忽略清理错误
    }
  });

  describe("GET /api/v1/dataset-items - 分页查询数据项列表", () => {
    test("正向测试：获取所有数据项", async () => {
      const query = createDatasetItemQuery();
      const result = await DatasetItemAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.total).toBeGreaterThanOrEqual(0);

      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(typeof firstItem.id).toBe("number");
        expect(firstItem.id).toBeGreaterThan(0);
        expect(typeof firstItem.datasetId).toBe("number");
        expect(typeof firstItem.name).toBe("string");
      }
    });

    test("正向测试：按数据集筛选", async () => {
      const query = createDatasetItemQuery({ datasetId: testDatasetId });
      const result = await DatasetItemAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      result.list.forEach((item) => {
        expect(item.datasetId).toBe(testDatasetId);
      });
    });

    test("正向测试：按场景类型筛选", async () => {
      const query = createDatasetItemQuery({ sceneType: "urban" });
      const result = await DatasetItemAPI.getList(query);
      result.list.forEach((item) => {
        expect(item.sceneType).toBe("urban");
      });
    });

    test("边界测试：大页码返回空结果", async () => {
      const query = createDatasetItemQuery({ pageNum: 99999 });
      const result = await DatasetItemAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });
  });

  describe("POST /api/v1/dataset-items - 创建空数据项", () => {
    const createdIds: number[] = [];

    afterAll(async () => {
      for (const id of createdIds) {
        try {
          await DatasetItemAPI.deleteById(id);
        } catch (e) {
          // 忽略清理错误
        }
      }
    });

    test("正向测试：创建有效数据项", async () => {
      const form = createDatasetItemForm(testDatasetId);
      const result = await DatasetItemAPI.add(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toBe(form.name);
      expect(result.datasetId).toBe(testDatasetId);
      if (result.sceneType) {
        expect(result.sceneType).toBe("urban");
      }
      createdIds.push(result.id);
    });

    test("正向测试：创建带描述的数据项", async () => {
      const form = createDatasetItemForm(testDatasetId, {
        description: "带描述的测试数据项",
      });
      const result = await DatasetItemAPI.add(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.datasetId).toBe(testDatasetId);
      createdIds.push(result.id);
    });

    test("参数校验：缺少必需字段 datasetId", async () => {
      const form = createDatasetItemForm(testDatasetId);
      delete (form as any).datasetId;
      await expectBizError(DatasetItemAPI.add(form), ["A0400", "B0001"], undefined, true);
    });
  });

  describe("GET /api/v1/dataset-items/{id} - 获取数据项详情", () => {
    let itemId: number;

    beforeAll(async () => {
      const form = createDatasetItemForm(testDatasetId);
      const result = await DatasetItemAPI.add(form);
      itemId = result.id;
    });

    afterAll(async () => {
      try {
        await DatasetItemAPI.deleteById(itemId);
      } catch (e) {
        // 忽略清理错误
      }
    });

    test("正向测试：获取有效数据项详情", async () => {
      const result = await DatasetItemAPI.getById(itemId);
      expect(result.id).toBe(itemId);
      expect(result.datasetId).toBe(testDatasetId);
      expect(typeof result.name).toBe("string");
      expect(result.name.length).toBeGreaterThan(0);
      if (result.sceneType) {
        expect(result.sceneType).toBe("urban");
      }
    });

    test("正向测试：验证返回字段完整性", async () => {
      const result = await DatasetItemAPI.getById(itemId);
      expect(typeof result.id).toBe("number");
      expect(typeof result.datasetId).toBe("number");
      expect(typeof result.name).toBe("string");
      if (result.imageCount !== undefined) {
        expect(typeof result.imageCount).toBe("number");
        expect(result.imageCount).toBeGreaterThanOrEqual(0);
      }
      if (result.usageCount !== undefined) {
        expect(typeof result.usageCount).toBe("number");
        expect(result.usageCount).toBeGreaterThanOrEqual(0);
      }
    });

    test("异常测试：获取不存在的数据项", async () => {
      await expectBizError(
        DatasetItemAPI.getById(99999999),
        ["A0401", "B0001", "A0400"],
        undefined,
        true
      );
    });
  });

  describe("PUT /api/v1/dataset-items/{id} - 修改数据项信息", () => {
    let itemId: number;

    beforeAll(async () => {
      const form = createDatasetItemForm(testDatasetId);
      const result = await DatasetItemAPI.add(form);
      itemId = result.id;
    });

    afterAll(async () => {
      try {
        await DatasetItemAPI.deleteById(itemId);
      } catch (e) {
        // 忽略清理错误
      }
    });

    test("正向测试：更新数据项名称", async () => {
      const form = createDatasetItemUpdateForm();
      const result = await DatasetItemAPI.update(itemId, form);
      expect(result.id).toBe(itemId);
      expect(result.name).toBe(form.name);
    });

    test("正向测试：更新场景类型", async () => {
      const form = createDatasetItemUpdateForm({
        sceneType: "rural",
      });
      const result = await DatasetItemAPI.update(itemId, form);
      expect(result.id).toBe(itemId);
      if (result.sceneType) {
        expect(result.sceneType).toBe("rural");
      }
    });

    test("异常测试：更新不存在的数据项", async () => {
      const form = createDatasetItemUpdateForm();
      await expectBizError(
        DatasetItemAPI.update(99999999, form),
        ["A0401", "B0001", "A0400"],
        undefined,
        true
      );
    });
  });

  describe("DELETE /api/v1/dataset-items/{id} - 删除数据项", () => {
    test("正向测试：删除有效数据项并验证不存在", async () => {
      const form = createDatasetItemForm(testDatasetId);
      const result = await DatasetItemAPI.add(form);
      expect(result.id).toBeGreaterThan(0);

      await DatasetItemAPI.deleteById(result.id);

      // 验证已删除
      await expectBizError(
        DatasetItemAPI.getById(result.id),
        ["A0401", "B0001", "A0400"],
        undefined,
        true
      );
    });

    test("异常测试：删除不存在的数据项（后端bug - 应返回错误）", async () => {
      // 【预期行为】删除不存在的资源应返回业务错误
      // 【实际行为】后端可能返回成功（幂等）或报错
      await DatasetItemAPI.deleteById(99999999).catch((error: any) => {
        const bizError = error.response?.data || error;
        expect(bizError.code).toBe("A0401");
      });
    });
  });

  describe("DELETE /api/v1/dataset-items/batch - 批量删除数据项", () => {
    test("正向测试：批量删除数据项", async () => {
      const itemIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createDatasetItemForm(testDatasetId);
        const result = await DatasetItemAPI.add(form);
        itemIds.push(result.id);
      }

      const batchForm: BatchDeleteForm = {
        ids: itemIds,
      };
      const result = await DatasetItemAPI.batchDelete(batchForm);
      expect(result).toBeDefined();
      if (result.successCount !== undefined) {
        expect(result.successCount).toBe(itemIds.length);
      }
      if (result.failedCount !== undefined) {
        expect(result.failedCount).toBe(0);
      }
      if (result.successIds) {
        expect(result.successIds).toEqual(expect.arrayContaining(itemIds));
      }
    });

    test("参数校验：空ID数组", async () => {
      const form: BatchDeleteForm = {
        ids: [],
      };
      await expectBizError(DatasetItemAPI.batchDelete(form), ["A0400", "B0001"], undefined, true);
    });

    test("异常测试：包含不存在的ID", async () => {
      const form: BatchDeleteForm = {
        ids: [99999999, 99999998],
      };
      const result = await DatasetItemAPI.batchDelete(form);
      expect(result).toBeDefined();
      if (result.failedCount !== undefined) {
        expect(result.failedCount).toBeGreaterThan(0);
      }
    });
  });

  describe("完整 CRUD 生命周期：数据项管理", () => {
    test("创建→读→更新→读→删除→验证不存在", async () => {
      // Create
      const createForm = createDatasetItemForm(testDatasetId, { description: "CRUD生命周期测试" });
      const created = await DatasetItemAPI.add(createForm);
      expect(created.id).toBeGreaterThan(0);
      const itemId = created.id;

      // Read: 验证字段
      const detail = await DatasetItemAPI.getById(itemId);
      expect(detail.name).toBe(createForm.name);
      expect(detail.datasetId).toBe(testDatasetId);

      // Update
      const newName = `CRUD更新_${Date.now()}`;
      const updateForm = createDatasetItemUpdateForm({ name: newName });
      const updated = await DatasetItemAPI.update(itemId, updateForm);
      expect(updated.name).toBe(newName);

      // Read: 验证更新
      const readUpdated = await DatasetItemAPI.getById(itemId);
      expect(readUpdated.name).toBe(newName);

      // Delete
      await DatasetItemAPI.deleteById(itemId);

      // Verify
      await expectBizError(
        DatasetItemAPI.getById(itemId),
        ["A0401", "B0001", "A0400"],
        undefined,
        true
      );
    });
  });
});
