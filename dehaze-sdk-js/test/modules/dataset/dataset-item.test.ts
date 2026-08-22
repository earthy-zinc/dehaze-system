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

  // add 失败会抛错，返回值必为有效 id
  async function createTestItem(): Promise<number> {
    const result = await DatasetItemAPI.add(createDatasetItemForm(testDatasetId));
    expect(result.id).toBeGreaterThan(0);
    return result.id;
  }

  async function cleanupTestItem(id: number): Promise<void> {
    try {
      await DatasetItemAPI.deleteById(id);
    } catch {
      // 清理失败不影响用例结果
    }
  }

  beforeAll(async () => {
    testDatasetId = await DatasetAPI.add(createDatasetForm({ type: "图像去雾" }));
  });

  afterAll(async () => {
    try {
      await DatasetAPI.deleteById(testDatasetId);
    } catch {
      // 清理失败不影响用例结果
    }
  });

  describe("GET /api/v1/dataset-items - 分页查询数据项列表", () => {
    test("正向测试：获取所有数据项", async () => {
      const result = await DatasetItemAPI.getList(createDatasetItemQuery());
      expect(Array.isArray(result.list)).toBe(true);

      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(firstItem.id).toBeGreaterThan(0);
        expect(typeof firstItem.datasetId).toBe("number");
        expect(typeof firstItem.name).toBe("string");
      }
    });

    test("正向测试：按数据集筛选", async () => {
      const result = await DatasetItemAPI.getList(
        createDatasetItemQuery({ datasetId: testDatasetId })
      );
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.datasetId).toBe(testDatasetId);
      });
    });

    test("正向测试：按场景类型筛选", async () => {
      const result = await DatasetItemAPI.getList(createDatasetItemQuery({ sceneType: "urban" }));
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.sceneType).toBe("urban");
      });
    });
  });

  describe("POST /api/v1/dataset-items - 创建空数据项", () => {
    const createdIds: number[] = [];

    afterAll(async () => {
      for (const id of createdIds) {
        await cleanupTestItem(id);
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
      await expectBizError(DatasetItemAPI.add(form), ["A0400", "B0001"]);
    });
  });

  describe("GET /api/v1/dataset-items/{id} - 获取数据项详情", () => {
    let itemId: number;

    beforeAll(async () => {
      itemId = await createTestItem();
    });

    afterAll(async () => {
      await cleanupTestItem(itemId);
    });

    test("正向测试：获取有效数据项详情", async () => {
      const result = await DatasetItemAPI.getById(itemId);
      expect(result.id).toBe(itemId);
      expect(result.datasetId).toBe(testDatasetId);
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
      }
      if (result.usageCount !== undefined) {
        expect(typeof result.usageCount).toBe("number");
      }
    });

    test("异常测试：获取不存在的数据项", async () => {
      await expectBizError(DatasetItemAPI.getById(99999999), ["A0401", "B0001", "A0400"]);
    });
  });

  describe("PUT /api/v1/dataset-items/{id} - 修改数据项信息", () => {
    let itemId: number;

    beforeAll(async () => {
      itemId = await createTestItem();
    });

    afterAll(async () => {
      await cleanupTestItem(itemId);
    });

    test("正向测试：更新数据项名称", async () => {
      const form = createDatasetItemUpdateForm();
      const result = await DatasetItemAPI.update(itemId, form);
      expect(result.id).toBe(itemId);
      expect(result.name).toBe(form.name);
    });

    test("正向测试：更新场景类型", async () => {
      const form = createDatasetItemUpdateForm({ sceneType: "rural" });
      const result = await DatasetItemAPI.update(itemId, form);
      expect(result.id).toBe(itemId);
      if (result.sceneType) {
        expect(result.sceneType).toBe("rural");
      }
    });

    test("异常测试：更新不存在的数据项", async () => {
      const form = createDatasetItemUpdateForm();
      await expectBizError(DatasetItemAPI.update(99999999, form), ["A0401", "B0001", "A0400"]);
    });
  });

  describe("DELETE /api/v1/dataset-items/{id} - 删除数据项", () => {
    test("正向测试：删除有效数据项并验证不存在", async () => {
      const itemId = await createTestItem();
      await DatasetItemAPI.deleteById(itemId);
      await expectBizError(DatasetItemAPI.getById(itemId), ["A0401", "B0001", "A0400"]);
    });

    test("异常测试：删除不存在的数据项", async () => {
      await expectBizError(DatasetItemAPI.deleteById(99999999), ["A0401"]);
    });
  });

  describe("DELETE /api/v1/dataset-items/batch - 批量删除数据项", () => {
    test("正向测试：批量删除数据项", async () => {
      const itemIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        itemIds.push(await createTestItem());
      }

      const result = await DatasetItemAPI.batchDelete({ ids: itemIds } as BatchDeleteForm);
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
      await expectBizError(DatasetItemAPI.batchDelete({ ids: [] } as BatchDeleteForm), [
        "A0400",
        "B0001",
      ]);
    });

    test("异常测试：包含不存在的ID", async () => {
      const result = await DatasetItemAPI.batchDelete({
        ids: [99999999, 99999998],
      } as BatchDeleteForm);
      if (result.failedCount !== undefined) {
        expect(result.failedCount).toBeGreaterThan(0);
      }
    });
  });

  describe("完整 CRUD 生命周期：数据项管理", () => {
    test("创建→读→更新→读→删除→验证不存在", async () => {
      const createForm = createDatasetItemForm(testDatasetId, { description: "CRUD生命周期测试" });
      const created = await DatasetItemAPI.add(createForm);
      expect(created.id).toBeGreaterThan(0);
      const itemId = created.id;

      const detail = await DatasetItemAPI.getById(itemId);
      expect(detail.name).toBe(createForm.name);
      expect(detail.datasetId).toBe(testDatasetId);

      const newName = `CRUD更新_${Date.now()}`;
      const updateForm = createDatasetItemUpdateForm({ name: newName });
      const updated = await DatasetItemAPI.update(itemId, updateForm);
      expect(updated.name).toBe(newName);

      const readUpdated = await DatasetItemAPI.getById(itemId);
      expect(readUpdated.name).toBe(newName);

      await DatasetItemAPI.deleteById(itemId);
      await expectBizError(DatasetItemAPI.getById(itemId), ["A0401", "B0001", "A0400"]);
    });
  });
});
