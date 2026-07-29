import { AlgorithmAPI, Algorithm, ImportExportAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";

describe("算法管理新增端点测试", () => {
  let testAlgorithmId: number;

  // 创建一个测试算法（通过前端的 AlgorithmFormDialog 新增后自动状态=1 草稿）
  beforeAll(async () => {
    const form: Partial<Algorithm> = {
      parentId: 0,
      name: `SdkTest_${Date.now()}`,
      type: "TEST",
      description: "SDK 测试用算法",
      status: 1,
    };
    const id = (await AlgorithmAPI.add(form)) as unknown as number;
    testAlgorithmId = typeof id === "number" ? id : Number(id);
  });

  afterAll(async () => {
    if (testAlgorithmId) {
      try {
        // 恢复为草稿状态后再删除（DELETABLE_STATUSES 仅允许草稿/已停用/已归档删除）
        await AlgorithmAPI.updateStatus(testAlgorithmId, 1).catch(() => {});
        await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
      } catch (_) {}
    }
  });

  describe("PUT /api/v1/algorithms/{id}/status - 状态变更", () => {
    test("正向测试：将草稿(1)切换为测试中(2)", async () => {
      // 草稿 → 测试中
      await AlgorithmAPI.updateStatus(testAlgorithmId, 2);

      // 验证状态持久化
      const info = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(info.status).toBe(2);

      // 注：按设计文档 SSOT，测试中(2) 仅可流转到 待审核(3)，不可回退到草稿(1)
    });

    test("参数校验：无效状态值应提示错误", async () => {
      await expectBizError(AlgorithmAPI.updateStatus(testAlgorithmId, 99), [
        "A0502",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      await expectBizError(AlgorithmAPI.updateStatus(99999999, 1), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/algorithms/{id}/versions - 版本管理", () => {
    test("正向测试：获取版本历史列表", async () => {
      const versions = await AlgorithmAPI.getVersions(testAlgorithmId);
      expect(versions).toBeDefined();
      expect(Array.isArray(versions)).toBe(true);

      versions.forEach((v: any) => {
        expect(v.version).toBeTruthy();
        expect(typeof v.version).toBe("string");
      });
    });

    test("异常测试：不存在的算法ID应报错或返回空数组", async () => {
      // 不存在的算法ID：可能返回错误，也可能返回空数组（两种都是合理的业务行为）
      const versions = await AlgorithmAPI.getVersions(99999999).catch(() => null);
      if (versions === null) {
        // 后端抛出错误，合理
        return;
      }
      expect(Array.isArray(versions)).toBe(true);
      // 不存在的算法不应有版本数据
      expect(versions.length).toBe(0);
    });
  });

  describe("GET /api/v1/algorithms/{id}/monitor - 监控数据", () => {
    test("正向测试：获取监控数据", async () => {
      const monitor = await AlgorithmAPI.getMonitorData(testAlgorithmId);

      expect(monitor).toBeDefined();
      expect(typeof monitor.callCount).toBe("number");
      expect(monitor.callCount).toBeGreaterThanOrEqual(0);
      expect(typeof monitor.avgTime).toBe("number");
      expect(monitor.avgTime).toBeGreaterThanOrEqual(0);
      expect(typeof monitor.successRate).toBe("number");
      expect(monitor.successRate).toBeGreaterThanOrEqual(0);
      expect(monitor.successRate).toBeLessThanOrEqual(100);
      expect(typeof monitor.todayCallCount).toBe("number");
      expect(monitor.todayCallCount).toBeGreaterThanOrEqual(0);
    });

    test("正向测试：获取监控统计报表", async () => {
      const stats = await AlgorithmAPI.getMonitorStats(testAlgorithmId);
      expect(stats).toBeDefined();
      expect(Array.isArray(stats)).toBe(true);
      expect(stats.length).toBeGreaterThan(0);
      // 每条记录应包含 date、callCount、avgTime、successRate 字段
      const first = stats[0] as Record<string, unknown>;
      expect(typeof first.date).toBe("string");
      expect(typeof first.callCount).toBe("number");
      expect(typeof first.avgTime).toBe("number");
      expect(typeof first.successRate).toBe("number");
      expect(first.successRate as number).toBeLessThanOrEqual(100);
    });
  });

  describe("GET /api/v1/algorithms/_export - 算法导出（通过 ImportExportAPI）", () => {
    test("正向测试：导出算法（同步返回 Blob 或异步返回任务）", async () => {
      const result = await ImportExportAPI.export("algorithm", { id: testAlgorithmId });
      expect(result).toBeDefined();
      const size =
        (result as { size?: number }).size ??
        (result as { length?: number }).length ??
        (result as { byteLength?: number }).byteLength;
      const isBlob = typeof size === "number" && size > 0;
      const isTaskResult = !isBlob && typeof (result as { taskId?: string }).taskId === "string";
      expect(isBlob || isTaskResult).toBe(true);
    });
  });
});
