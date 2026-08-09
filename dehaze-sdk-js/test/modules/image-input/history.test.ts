import { ImageInputHistoryAPI, HistoryForm } from "../../../index";
import { expectBizError } from "#/utils/assertion";

describe("图像输入历史记录 API 测试", () => {
  const createdIds: number[] = [];

  afterAll(async () => {
    for (const id of createdIds) {
      try {
        await ImageInputHistoryAPI.deleteById(id);
      } catch {}
    }
  });

  describe("POST /api/v1/image-input/history - 创建历史记录", () => {
    test("正向测试：创建一条处理成功的历史记录", async () => {
      const form: HistoryForm = {
        originalImageUrl: "/images/test_haze.jpg",
        resultImageUrl: "/images/test_dehazed.jpg",
        algorithmId: 1,
        algorithmName: "DCP",
        processingTime: 1520,
        status: 1, // 成功
        inputSource: "upload",
      };

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeDefined();
      expect(typeof id).toBe("number");
      expect(id).toBeGreaterThan(0);

      createdIds.push(id);
    });

    test("正向测试：创建一条处理失败的历史记录", async () => {
      const form: HistoryForm = {
        originalImageUrl: "/images/test_bad.jpg",
        algorithmId: 1,
        algorithmName: "DCP",
        processingTime: 300,
        status: 2, // 失败
        inputSource: "sample",
      };

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });

    test("正向测试：创建一条来源为 camera 的处理中记录", async () => {
      const form: HistoryForm = {
        originalImageUrl: "/images/camera_test.jpg",
        algorithmId: 2,
        algorithmName: "AODNet",
        status: 3, // 处理中
        inputSource: "camera",
      };

      const id = await ImageInputHistoryAPI.create(form);
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });
  });

  describe("GET /api/v1/image-input/history - 分页查询", () => {
    test("正向测试：分页查询自己的历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 20 });

      expect(page).toBeDefined();
      expect(Array.isArray(page.list)).toBe(true);
      expect(typeof page.total).toBe("number");

      // 验证至少包含我们创建的数据
      expect(page.total).toBeGreaterThanOrEqual(createdIds.length);

      // 验证列表项结构
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
      expect(page).toBeDefined();
      if (page.list.length > 0) {
        page.list.forEach((item) => {
          expect(item.status).toBe(1);
        });
      }
    });

    test("正向测试：按来源筛选历史记录", async () => {
      const page = await ImageInputHistoryAPI.getPage({
        pageNum: 1,
        pageSize: 10,
        inputSource: "upload",
      });
      expect(page).toBeDefined();
      if (page.list.length > 0) {
        page.list.forEach((item) => {
          expect(item.inputSource).toBe("upload");
        });
      }
    });
  });

  describe("GET /api/v1/image-input/history/{id} - 获取详情", () => {
    test("正向测试：获取已创建的历史记录详情并验证字段值", async () => {
      expect(createdIds.length).toBeGreaterThan(0);

      const detail = await ImageInputHistoryAPI.getById(createdIds[0]!);

      expect(detail).toBeDefined();
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
      // 创建一条用于删除的记录
      const form: HistoryForm = {
        originalImageUrl: "/images/delete_test.jpg",
        algorithmId: 1,
        algorithmName: "Test",
        status: 3,
        inputSource: "upload",
      };
      const id = await ImageInputHistoryAPI.create(form);

      // 删除
      await ImageInputHistoryAPI.deleteById(id);

      // 验证已删除
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
        const form: HistoryForm = {
          originalImageUrl: `/images/batch_${i}.jpg`,
          algorithmId: 1,
          algorithmName: "Test",
          status: 1,
          inputSource: "upload",
        };
        const id = await ImageInputHistoryAPI.create(form);
        batchIds.push(id);
      }

      const deletedCount = await ImageInputHistoryAPI.batchDelete(batchIds);
      expect(deletedCount).toBeGreaterThanOrEqual(batchIds.length);

      // 验证已删除
      for (const id of batchIds) {
        await expectBizError(ImageInputHistoryAPI.getById(id), [
          "A0401",
          "A0400",
          "ERR_BAD_REQUEST",
        ]);
      }
    });
  });

  describe("DELETE /api/v1/image-input/history/clear - 清空", () => {
    test("正向测试：清空当前用户所有历史记录", async () => {
      const count = await ImageInputHistoryAPI.clearAll(true);
      expect(count).toBeDefined();
      expect(typeof count).toBe("number");

      // 验证已清空
      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 1 });
      expect(page.total).toBe(0);
    });
  });
});
