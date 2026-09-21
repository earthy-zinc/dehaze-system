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

    test("边界：空数组批量删除应被拒绝（A0400，与全项目批量删除口径一致）", async () => {
      await expectBizError(ImageInputHistoryAPI.batchDelete([]), ["A0400"]);
    });
  });

  describe("DELETE /api/v1/image-input/history/clear - 清空", () => {
    test("正向测试：清空当前用户所有历史记录", async () => {
      const count = await ImageInputHistoryAPI.clearAll();
      expect(typeof count).toBe("number");

      const page = await ImageInputHistoryAPI.getPage({ pageNum: 1, pageSize: 1 });
      expect(page.total).toBe(0);
    });
  });

  describe("对抗性输入校验（后端参数校验 A0400）", () => {
    test("异常测试：无效 status（超出 1-3）应被拒绝", async () => {
      await expectBizError(
        ImageInputHistoryAPI.create(buildForm({ status: 99 as unknown as number })),
        ["A0400"]
      );
    });

    test("异常测试：无效 inputSource（非枚举值）应被拒绝", async () => {
      await expectBizError(
        ImageInputHistoryAPI.create(buildForm({ inputSource: "hacked" as unknown as string })),
        ["A0400"]
      );
    });

    test("异常测试：超长 originalImageUrl（504 字符，超出列宽 500）应被拒绝", async () => {
      await expectBizError(
        ImageInputHistoryAPI.create(
          buildForm({ originalImageUrl: `/images/${"a".repeat(492)}.jpg` })
        ),
        ["A0400"]
      );
    });

    test("边界：originalImageUrl 恰好 500 字符应创建成功", async () => {
      // "/images/" 8 + padding + ".jpg" 4 = 500
      const padding = 500 - "/images/".length - ".jpg".length;
      const id = await ImageInputHistoryAPI.create(
        buildForm({ originalImageUrl: `/images/${"a".repeat(padding)}.jpg` })
      );
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);
    });

    test("异常测试：超长 algorithmName（101 字符，超出列宽 100）应被拒绝", async () => {
      await expectBizError(
        ImageInputHistoryAPI.create(buildForm({ algorithmName: "A".repeat(101) })),
        ["A0400"]
      );
    });

    test("异常测试：非法 algorithmParams（非 JSON 字符串）应被拒绝", async () => {
      await expectBizError(
        ImageInputHistoryAPI.create(buildForm({ algorithmParams: "not-a-json{[" })),
        ["A0400"]
      );
    });

    test("边界：合法 algorithmParams（JSON 字符串）应创建成功且回读一致", async () => {
      const id = await ImageInputHistoryAPI.create(
        buildForm({ algorithmParams: '{"topK":5,"mode":"fast"}' })
      );
      expect(id).toBeGreaterThan(0);
      createdIds.push(id);

      const detail = await ImageInputHistoryAPI.getById(id);
      // 回读为合法 JSON 字符串即可，key 顺序不保证，按解析后对象比较
      expect(JSON.parse(detail.algorithmParams as string)).toEqual({
        topK: 5,
        mode: "fast",
      });
    });

    test("异常测试：负数 processingTime 应被拒绝", async () => {
      await expectBizError(ImageInputHistoryAPI.create(buildForm({ processingTime: -100 })), [
        "A0400",
      ]);
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

    test("边界：批量删除他人记录应不生效（仅删除本人记录，返回实际删除数）", async () => {
      const id = await ImageInputHistoryAPI.create(
        buildForm({ originalImageUrl: "/images/batch_iso_test.jpg" })
      );

      try {
        await login(USERS.USER.username);
        // 请求里混入他人记录 ID：user_id 过滤后实际删除数为 0，他人记录不受影响
        const deleted = await ImageInputHistoryAPI.batchDelete([id]);
        expect(deleted).toBe(0);
      } finally {
        await login(USERS.ADMIN.username);
        // 他人记录依然存在且可访问，随后清理
        const detail = await ImageInputHistoryAPI.getById(id);
        expect(detail.id).toBe(id);
        await ImageInputHistoryAPI.deleteById(id);
      }
    });
  });
});
