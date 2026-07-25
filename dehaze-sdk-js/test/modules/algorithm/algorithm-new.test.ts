import { AlgorithmAPI, Algorithm } from "../../../index";
import { expectBizError } from "#/utils/assertion";

describe("算法管理新增端点测试", () => {
  let testAlgorithmId: number;

  // 创建一个测试算法（通过前端的 AlgorithmFormDialog 新增后自动状态=0 草稿）
  beforeAll(async () => {
    const form: Partial<Algorithm> = {
      parentId: 0,
      name: `SdkTest_${Date.now()}`,
      type: "TEST",
      description: "SDK 测试用算法",
      status: 0,
    };
    const id = (await AlgorithmAPI.add(form)) as unknown as number;
    testAlgorithmId = typeof id === "number" ? id : Number(id);
  });

  afterAll(async () => {
    if (testAlgorithmId) {
      try {
        await AlgorithmAPI.deleteByIds([testAlgorithmId.toString()]);
      } catch (_) {}
    }
  });

  describe("PUT /api/v1/algorithms/{id}/status - 状态变更", () => {
    test("正向测试：将草稿(0)切换为测试中(1)", async () => {
      // 草稿 → 测试中
      await AlgorithmAPI.updateStatus(testAlgorithmId, 1);

      // 验证状态持久化
      const info = await AlgorithmAPI.getAlgorithmInfoById(testAlgorithmId);
      expect(info.status).toBe(1);

      // 注：按设计文档 SSOT，测试中(1) 仅可流转到 待审核(2)，不可回退到草稿(0)
    });

    test("参数校验：无效状态值应提示错误", async () => {
      await expectBizError(
        AlgorithmAPI.updateStatus(testAlgorithmId, 99),
        ["A0502", "A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });

    test("异常测试：不存在的算法ID应报错", async () => {
      await expectBizError(
        AlgorithmAPI.updateStatus(99999999, 1),
        ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
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
      expect(typeof stats.callCount).toBe("number");
      expect(stats.callCount).toBeGreaterThanOrEqual(0);
    });
  });

  describe("GET /api/v1/algorithms/{id}/_export - 导出", () => {
    test("正向测试：导出算法 JSON（返回 Blob）", async () => {
      const blob = await AlgorithmAPI.exportAlgorithm(testAlgorithmId);
      expect(blob).toBeDefined();
      // Node.js 环境下可能是 Blob/Buffer/string，验证有内容即可
      const size = (blob as any)?.size ?? (blob as any)?.length ?? (blob as any)?.byteLength;
      expect(size).toBeGreaterThan(0);
    });
  });

  describe("POST /api/v1/algorithms/_import/validate - 导入校验", () => {
    test("正向测试：校验合法 JSON 文件", async () => {
      const jsonContent = JSON.stringify({
        name: `TestImport_${Date.now()}`,
        type: "TEST",
        description: "导入测试",
        version: "0.0.1",
      });
      const file = new File([jsonContent], "test_algorithm.json", { type: "application/json" });

      const result = await AlgorithmAPI.validateImport(file);
      expect(result).toBeDefined();
      expect(typeof result).toBe("string");
    });

    test("参数校验：空文件应报错", async () => {
      const emptyFile = new File([], "empty.json", { type: "application/json" });
      await expectBizError(
        AlgorithmAPI.validateImport(emptyFile),
        ["A0400", "B0001", "ERR_BAD_REQUEST"],
        undefined,
        true
      );
    });
  });
});
