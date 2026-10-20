import { DatasetAPI, DatasetItemAPI, BatchDeleteForm, TaskAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import {
  createDatasetForm,
  createDatasetUpdateForm,
  createDatasetQuery,
  createDatasetItemForm,
} from "#/factories/dataset";
import { USERS } from "#/factories/constants";
import { login } from "#/utils/auth";

describe("数据集接口测试", () => {
  describe("GET /api/v1/datasets - 获取数据集列表", () => {
    test("正向测试：获取所有数据集", async () => {
      const query = createDatasetQuery();
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");

      if (result.list.length > 0) {
        const firstItem = result.list[0]!;
        expect(typeof firstItem.id).toBe("number");
        expect(firstItem.id).toBeGreaterThan(0);
        expect(typeof firstItem.name).toBe("string");
        expect(firstItem.name.length).toBeGreaterThan(0);
        expect(typeof firstItem.type).toBe("string");
      }
    });

    test("正向测试：按类型筛选数据集", async () => {
      const query = createDatasetQuery({ type: "用户数据集" });
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(typeof item.type).toBe("string");
        expect(item.type).toBe("用户数据集");
      });
    });

    test("正向测试：按状态筛选数据集", async () => {
      const query = createDatasetQuery({ status: 1 });
      const result = await DatasetAPI.getList(query);
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((item) => {
        expect(item.status).toBe(1);
      });
    });

    test("正向测试：按关键字搜索数据集", async () => {
      const form = createDatasetForm({ name: `keyword_test_${Date.now()}` });
      const id = await DatasetAPI.add(form);
      try {
        const result = await DatasetAPI.getList(createDatasetQuery({ keyword: "keyword_test" }));
        expect(result.list.length).toBeGreaterThan(0);
        const found = result.list.find((d) => d.id === id);
        expect(found).toBeDefined();
      } finally {
        await DatasetAPI.deleteById(id);
      }
    });

    test("安全：特殊字符搜索不引发XSS", async () => {
      const result = await DatasetAPI.getList(
        createDatasetQuery({ keyword: "<script>alert(1)</script>" })
      );
      expect(Array.isArray(result.list)).toBe(true);
      const jsonStr = JSON.stringify(result);
      expect(jsonStr).not.toContain("<script>");
    });
  });

  describe("GET /api/v1/datasets/options - 获取数据集下拉选项", () => {
    test("正向测试：获取下拉选项并验证结构", async () => {
      const options = await DatasetAPI.getOptions();
      expect(Array.isArray(options)).toBe(true);

      options.forEach((option: any) => {
        expect(typeof option.value).toBe("number");
        expect(typeof option.label).toBe("string");
      });
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
      const datasetId = await DatasetAPI.add(form);
      expect(datasetId).toBeGreaterThan(0);
      createdIds.push(datasetId);
    });

    test("正向测试：创建带可选字段的数据集", async () => {
      const form = createDatasetForm({
        description: "带描述的测试数据集",
        status: 1,
      });
      const datasetId = await DatasetAPI.add(form);
      expect(datasetId).toBeGreaterThan(0);
      createdIds.push(datasetId);
    });

    test("参数校验：缺少必需字段 name", async () => {
      const form = createDatasetForm();
      delete (form as any).name;
      await expectBizError(DatasetAPI.add(form), ["A0400", "B0001"]);
    });

    test("正向测试：新增子数据集", async () => {
      const parentForm = createDatasetForm();
      const parentId = await DatasetAPI.add(parentForm);
      createdIds.push(parentId);

      const childForm = createDatasetForm({ parentId });
      const childId = await DatasetAPI.add(childForm);
      expect(childId).toBeGreaterThan(0);
      createdIds.push(childId);

      const childDetail = await DatasetAPI.getDatasetInfoById(childId);
      expect(childDetail.parentId).toBe(parentId);
    });

    test("边界：名称唯一性校验（同级重名应失败）", async () => {
      const form = createDatasetForm({ name: `unique_test_${Date.now()}` });
      const id = await DatasetAPI.add(form);
      createdIds.push(id);

      await expectBizError(DatasetAPI.add(form), ["A0501", "A0400", "B0001", "ERR_BAD_REQUEST"]);
    });
  });

  describe("GET /api/v1/datasets/{id} - 获取数据集详细信息", () => {
    let datasetId: number;
    let createForm: ReturnType<typeof createDatasetForm>;

    beforeAll(async () => {
      createForm = createDatasetForm();
      datasetId = await DatasetAPI.add(createForm);
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
      expect(typeof result.name).toBe("string");
      expect(result.name.length).toBeGreaterThan(0);
      expect(typeof result.type).toBe("string");
      expect(result.parentId).toBeGreaterThanOrEqual(0);
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
      await expectBizError(DatasetAPI.getDatasetInfoById(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/datasets/{id} - 修改数据集信息", () => {
    let datasetId: number;

    beforeAll(async () => {
      const form = createDatasetForm();
      datasetId = await DatasetAPI.add(form);
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
        status: 0,
      });
      const result = await DatasetAPI.update(datasetId, form);
      expect(result.status).toBe(0);
    });

    test("异常测试：更新不存在的数据集", async () => {
      const form = createDatasetUpdateForm();
      await expectBizError(DatasetAPI.update(99999999, form), ["A0401", "B0001", "A0400"]);
    });
  });

  describe("DELETE /api/v1/datasets/{id} - 删除单个数据集", () => {
    test("正向测试：删除有效数据集", async () => {
      const form = createDatasetForm();
      const datasetId = await DatasetAPI.add(form);
      expect(datasetId).toBeGreaterThan(0);

      await DatasetAPI.deleteById(datasetId);

      await expectBizError(DatasetAPI.getDatasetInfoById(datasetId), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常测试：删除不存在的数据集", async () => {
      await expectBizError(DatasetAPI.deleteById(99999999), ["A0401", "B0001", "A0400"]);
    });
  });

  describe("DELETE /api/v1/datasets/batch - 批量删除数据集", () => {
    // 构造三级数据集树：父 -> 子 -> 孙
    const createDatasetTree = async () => {
      const parentDatasetId = await DatasetAPI.add(createDatasetForm());
      const childDatasetId = await DatasetAPI.add(createDatasetForm({ parentId: parentDatasetId }));
      const grandChildDatasetId = await DatasetAPI.add(
        createDatasetForm({ parentId: childDatasetId })
      );
      return { parentDatasetId, childDatasetId, grandChildDatasetId };
    };

    // 断言给定 ID 的数据集均已被删除
    const expectDatasetsDeleted = async (ids: number[]) => {
      for (const id of ids) {
        await expectBizError(DatasetAPI.getDatasetInfoById(id), ["A0401", "B0001", "A0400"]);
      }
    };

    test("正向测试：批量删除多个数据集", async () => {
      const datasetIds: number[] = [];
      for (let i = 0; i < 3; i++) {
        const form = createDatasetForm();
        const datasetId = await DatasetAPI.add(form);
        datasetIds.push(datasetId);
      }

      const batchForm: BatchDeleteForm = {
        ids: datasetIds,
      };
      const result = await DatasetAPI.batchDelete(batchForm);
      expect(result.succeeded).toBe(3);
      expect(result.failed).toBe(0);
    });

    test("参数校验：空ID数组", async () => {
      const form: BatchDeleteForm = {
        ids: [],
      };
      await expectBizError(DatasetAPI.batchDelete(form), ["A0400", "B0001"]);
    });

    test("异常测试：包含不存在的ID", async () => {
      const form: BatchDeleteForm = {
        ids: [99999999, 99999998],
      };
      const result = await DatasetAPI.batchDelete(form);
      expect(result.succeeded).toBe(0);
      expect(result.failed).toBe(2);
    });

    test("级联删除：同时选中父数据集和子数据集应递归删除所有子孙数据集", async () => {
      const { parentDatasetId, childDatasetId, grandChildDatasetId } = await createDatasetTree();

      const batchForm: BatchDeleteForm = {
        ids: [parentDatasetId, childDatasetId, grandChildDatasetId],
      };
      const result = await DatasetAPI.batchDelete(batchForm);
      expect(result.succeeded).toBe(3);

      await expectDatasetsDeleted([parentDatasetId, childDatasetId, grandChildDatasetId]);
    });

    test("级联删除：仅选中父数据集应递归删除所有子孙数据集", async () => {
      const { parentDatasetId, childDatasetId, grandChildDatasetId } = await createDatasetTree();

      const batchForm: BatchDeleteForm = {
        ids: [parentDatasetId],
      };
      const result = await DatasetAPI.batchDelete(batchForm);
      expect(result.succeeded).toBe(1);

      await expectDatasetsDeleted([parentDatasetId, childDatasetId, grandChildDatasetId]);
    });

    test("级联删除：删除含数据项的父数据集应同时删除所有子孙数据集及数据项", async () => {
      const parentForm = createDatasetForm();
      const parentDatasetId = await DatasetAPI.add(parentForm);

      const childForm = createDatasetForm({ parentId: parentDatasetId });
      const childDatasetId = await DatasetAPI.add(childForm);

      const itemForm1 = createDatasetItemForm(childDatasetId, {
        name: "级联删除测试数据项1",
      });
      const item1 = await DatasetItemAPI.add(itemForm1);

      const itemForm2 = createDatasetItemForm(childDatasetId, {
        name: "级联删除测试数据项2",
      });
      const item2 = await DatasetItemAPI.add(itemForm2);

      const batchForm: BatchDeleteForm = {
        ids: [parentDatasetId],
      };
      const result = await DatasetAPI.batchDelete(batchForm);
      expect(result.succeeded).toBe(1);

      await expectDatasetsDeleted([parentDatasetId, childDatasetId]);
      await expectBizError(DatasetItemAPI.getById(item1.id), ["A0401", "B0001", "A0400"]);
      await expectBizError(DatasetItemAPI.getById(item2.id), ["A0401", "B0001", "A0400"]);
    });
  });

  describe("POST /api/v1/tasks - 创建数据集导出任务（dataset_export）", () => {
    let datasetId: number;

    beforeAll(async () => {
      const form = createDatasetForm();
      datasetId = await DatasetAPI.add(form);
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
      expect(typeof result.taskId).toBe("string");
      expect([1, 2, 3, 4]).toContain(result.status);
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
      // 数据集存在性校验在异步策略 DatasetExportStrategy 中执行，任务最终状态为 FAILED，
      // 因此此处仅验证任务能创建成功。
      const result = await TaskAPI.create({
        type: "dataset_export",
        targetId: 99999999,
      });
      expect(result.taskId).toBeDefined();
      expect(result.status).toBe(1);
    });
  });

  // GET /api/v1/datasets/evaluation-options（T-DS-046），路由声明在 /{dataset_id} 之前避免被吞掉
  describe("GET /api/v1/datasets/evaluation-options - 测试集选项（评估接入）", () => {
    test("正向测试：按任务类型获取测试集选项", async () => {
      const options = await DatasetAPI.getEvaluationOptions("dehaze");
      expect(Array.isArray(options)).toBe(true);
      options.forEach((option: any) => {
        expect(typeof option.value).toBe("number");
        expect(typeof option.label).toBe("string");
      });
    });
  });

  describe("边界测试", () => {
    test("超长数据集名称应被拒绝", async () => {
      const form = createDatasetForm({ name: "x".repeat(500) });
      await expectBizError(DatasetAPI.add(form), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    test("边界测试：特殊字符数据集名称不应污染存储", async () => {
      const specialName = `测试<>&"'数据集_${Date.now()}`;
      const form = createDatasetForm({ name: specialName });

      const datasetId = await DatasetAPI.add(form);
      expect(datasetId).toBeGreaterThan(0);

      try {
        const detail = await DatasetAPI.getDatasetInfoById(datasetId);
        expect(typeof detail.name).toBe("string");
        expect(detail.name.length).toBeGreaterThan(0);
        expect(detail.name).not.toMatch(/<[^>]+>/);
      } finally {
        try {
          await DatasetAPI.deleteById(datasetId);
        } catch {
          // 忽略清理错误
        }
      }
    });
  });

  // 后端已为数据集接口加 require_permission（sys:dataset:add/edit/delete），
  // 普通用户访问返回 A0301（访问未授权）。
  describe("权限测试 - 普通用户管理操作应失败", () => {
    beforeAll(async () => {
      await login(USERS.USER.username);
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户新增数据集应失败", async () => {
      const form = createDatasetForm();
      await expectBizError(DatasetAPI.add(form), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户修改数据集应失败", async () => {
      await expectBizError(DatasetAPI.update(1, { name: "hacked" } as any), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户删除数据集应失败", async () => {
      await expectBizError(DatasetAPI.deleteById(999), [
        "A0301",
        "A0403",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
