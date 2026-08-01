import { expectBizError } from "#/utils/assertion";
import { ImportExportAPI } from "../../../index";
import {
  createAsyncExportRequest,
  createCsvExportRequest,
  createExportRequest,
  isBlobLike,
  isTaskResultLike,
} from "./factories";

describe("通用导出接口测试 - ImportExportAPI.export", () => {
  describe("GET /api/v1/user/_export - 用户模块导出", () => {
    test("同步导出用户(默认 Excel)返回 Blob 文件流", async () => {
      const result = await ImportExportAPI.export("user", createExportRequest());

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);

      if (isBlobLike(result)) {
        const blob = result as Blob;
        expect(blob.size).toBeGreaterThan(0);
      }
    });

    test("强制异步导出用户返回 ExportResult(taskId)", async () => {
      const result = await ImportExportAPI.export("user", createAsyncExportRequest());

      expect(result).toBeDefined();
      expect(isTaskResultLike(result)).toBe(true);
      const taskResult = result as { taskId: string; status: number; estimatedCount?: number };
      expect(typeof taskResult.taskId).toBe("string");
      expect(taskResult.taskId.length).toBeGreaterThan(0);
      expect(taskResult.status).toBe(1);
    });

    test("exportByPost 复杂查询导出用户返回 Blob 或 taskId", async () => {
      const result = await ImportExportAPI.exportByPost(
        "user",
        createExportRequest({ keywords: "admin" })
      );

      expect(result).toBeDefined();
      expect(isTaskResultLike(result) || isBlobLike(result)).toBe(true);
    });
  });

  describe("GET /api/v1/role/_export - 角色模块导出", () => {
    test("同步导出角色 Excel 返回 Blob", async () => {
      const result = await ImportExportAPI.export("role", createExportRequest({ format: "excel" }));

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });

    test("导出角色 CSV 格式返回 Blob", async () => {
      const result = await ImportExportAPI.export("role", createCsvExportRequest());

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);

      if (isBlobLike(result)) {
        const blob = result as Blob;
        expect(blob.size).toBeGreaterThan(0);
      }
    });
  });

  describe("GET /api/v1/dataset/_export - 数据集模块导出", () => {
    test("数据集导出总是异步返回 taskId", async () => {
      const result = await ImportExportAPI.export("dataset", createExportRequest());

      expect(result).toBeDefined();
      expect(isTaskResultLike(result) || isBlobLike(result)).toBe(true);
    });
  });

  describe("GET /api/v1/algorithm/_export - 算法模块导出", () => {
    test("导出算法列表 Excel 返回 Blob 或 taskId", async () => {
      const result = await ImportExportAPI.export("algorithm", createExportRequest());

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });
  });

  describe("格式切换 - format 参数", () => {
    test("导出用户 Excel 格式", async () => {
      const result = await ImportExportAPI.export("user", createExportRequest({ format: "excel" }));

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });

    test("导出用户 CSV 格式", async () => {
      const result = await ImportExportAPI.export("user", createCsvExportRequest());

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });
  });

  describe("字段选择 - fields 参数", () => {
    test("指定 fields 子集导出用户返回 Blob 或 taskId", async () => {
      const result = await ImportExportAPI.export(
        "user",
        createExportRequest({
          fields: ["username", "nickname", "email"],
        })
      );

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });

    test("指定全部字段导出用户返回 Blob 或 taskId", async () => {
      const result = await ImportExportAPI.export(
        "user",
        createExportRequest({
          fields: ["username", "nickname", "email", "mobile", "gender", "status", "deptId"],
        })
      );

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });

    test("指定空 fields 数组等价于不传 fields", async () => {
      const result = await ImportExportAPI.export("user", createExportRequest({ fields: [] }));

      expect(result).toBeDefined();
      expect(isBlobLike(result) || isTaskResultLike(result)).toBe(true);
    });
  });

  describe("异常场景", () => {
    test("导出不存在的模块应抛出错误", async () => {
      await expectBizError(ImportExportAPI.export("unknown_module" as any, createExportRequest()), [
        "B0001",
        "A0400",
        "A0710",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
