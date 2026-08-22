import { ImageInputHistoryAPI, HistoryForm } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { TestCleanupRegistry } from "#/utils/cleanup";
import { USERS } from "#/factories/constants";

function buildForm(overrides: Partial<HistoryForm> = {}): HistoryForm {
  return {
    originalImageUrl: "/images/test_haze.jpg",
    algorithmId: 1,
    algorithmName: "DCP",
    status: 1,
    inputSource: "upload",
    ...overrides,
  };
}

describe("图像输入历史记录 API 测试", () => {
  const cleanup = new TestCleanupRegistry();
  const createdIds: number[] = [];

  afterAll(async () => {
    cleanup.registerIds(
      () => createdIds,
      (id) => ImageInputHistoryAPI.deleteById(Number(id))
    );
    await cleanup.executeAll();
  });

  describe("POST /api/v1/image-input/history - 创建历史记录", () => {
    test("正向测试：创建一条处理成功的历史记录", async () => {
      const form = buildForm({
        resultImageUrl: "/images/test_dehazed.jpg",
        processingTime: 1520,
      });

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });

    test("正向测试：创建一条处理失败的历史记录", async () => {
      const form = buildForm({
        originalImageUrl: "/images/test_bad.jpg",
        processingTime: 300,
        status: 2,
        inputSource: "sample",
      });

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });

    test("正向测试：创建一条来源为 camera 的处理中记录", async () => {
      const form = buildForm({
        originalImageUrl: "/images/camera_test.jpg",
        algorithmId: 2,
        algorithmName: "AODNet",
        status: 3,
        inputSource: "camera",
      });

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });
  });

  describe("GET /api/v1/image-input/history - 分页查询", () => {
    test("正向测试：分页查询自己的历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 20 });

      expect(Array.isArray(page.list)).toBe(true);
      // 验证至少包含我们创建的数据
      expect(page.total).toBeGreaterThanOrEqual(createdIds.length);

      if (page.list.length > 0) {
        const item = page.list[0]!;
        expect(typeof item.id).toBe("number");
        expect(item.id).toBeGreaterThan(0);
        expect(typeof item.originalImageUrl).toBe("string");
        expect(typeof item.status).toBe("number");
        expect(typeof item.inputSource).toBe("string");
        expect(typeof item.createTime).toBe("string");
      }
    });

    test("正向测试：按状态筛选历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 10, status: 1 });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((item) => {
        expect(item.status).toBe(1);
      });
    });

    test("正向测试：按来源筛选历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({
        pageNum: 1,
        pageSize: 10,
        inputSource: "upload",
      });
      expect(Array.isArray(page.list)).toBe(true);
      page.list.forEach((item) => {
        expect(item.inputSource).toBe("upload");
      });
    });

    test("正向测试：按时间范围筛选历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({
        pageNum: 1,
        pageSize: 10,
        startTime: "2025-01-01 00:00:00",
        endTime: "2099-12-31 23:59:59",
      });
      expect(Array.isArray(page.list)).toBe(true);
    });

    test("验证：历史记录按创建时间倒序排列", async () => {
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 20 });
      if (page.list.length < 2) return;
      for (let i = 1; i < page.list.length; i++) {
        const prev = page.list[i - 1]!.createTime;
        const curr = page.list[i]!.createTime;
        if (prev && curr) {
          expect(prev >= curr).toBe(true);
        }
      }
    });

    test("边界：大页码返回空列表", async () => {
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 10000, pageSize: 10 });
      expect(page.list.length).toBe(0);
    });
  });

  describe("GET /api/v1/image-input/history/{id} - 获取详情", () => {
    test("正向测试：获取已创建的历史记录详情并验证字段值", async () => {
      expect(createdIds.length).toBeGreaterThan(0);

      const detail = await ImageInputHistoryAPI.getById(createdIds[0]!);

      expect(detail.id).toBe(createdIds[0]);
      expect(detail.originalImageUrl).toBe("/images/test_haze.jpg");
      expect(detail.algorithmName).toBe("DCP");
      expect(detail.status).toBe(1);
      expect(detail.inputSource).toBe("upload");
      expect(typeof detail.createTime).toBe("string");
    });

    test("异常测试：访问不存在的记录应报错", async () => {
      await expectBizError(ImageInputHistoryAPI.getById(99999999), [
        "A0401",
        "A0400",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/image-input/history/{id} - 删除记录", () => {
    test("正向测试：删除单条历史记录", async () => {
      const form = buildForm({
        originalImageUrl: "/images/delete_test.jpg",
        algorithmName: "Test",
        status: 3,
      });
      const id = await ImageInputHistoryAPI.create(form);

      await ImageInputHistoryAPI.deleteById(id);

      await expectBizError(ImageInputHistoryAPI.getById(id), ["A0401", "A0400", "ERR_BAD_REQUEST"]);
    });

    test("幂等性测试：删除已不存在的记录应成功", async () => {
      // 已删除的记录再删不应报错（幂等：promise resolve 即为成功）
      await ImageInputHistoryAPI.deleteById(99999999);
    });
  });

  describe("DELETE /api/v1/image-input/history/batch - 批量删除", () => {
    test("正向测试：批量删除多条记录", async () => {
      const batchIds: number[] = [];
      for (let i = 0; i < 2; i++) {
        const id = await ImageInputHistoryAPI.create(
          buildForm({ originalImageUrl: `/images/batch_${i}.jpg`, algorithmName: "Test" })
        );
        batchIds.push(id);
      }

      const deletedCount = await ImageInputHistoryAPI.batchDelete(batchIds);
      expect(deletedCount).toBeGreaterThanOrEqual(batchIds.length);

      for (const id of batchIds) {
        await expectBizError(ImageInputHistoryAPI.getById(id), [
          "A0401",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      }
    });

    test("边界：空数组批量删除", async () => {
      const result = await ImageInputHistoryAPI.batchDelete([]);
      expect(typeof result).toBe("number");
    });
  });

  describe("DELETE /api/v1/image-input/history/clear - 清空", () => {
    test("正向测试：清空当前用户所有历史记录", async () => {
      const count = await ImageInputHistoryAPI.clearAll(true);
      expect(typeof count).toBe("number");

      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 1 });
      expect(page.total).toBe(0);
    });
  });

  describe("数据隔离", () => {
    test("边界：查看他人历史记录应失败", async () => {
      const id = await ImageInputHistoryAPI.create(
        buildForm({ originalImageUrl: "/images/iso_test.jpg" })
      );

      try {
        await login(USERS.USER.username);
        await expectBizError(ImageInputHistoryAPI.getById(id), [
          "A0401",
          "B0300",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      } finally {
        await login(USERS.ADMIN.username);
        await ImageInputHistoryAPI.deleteById(id);
      }
    });
  });
});
