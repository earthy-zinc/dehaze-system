import { expectBizError } from "#/utils/assertion";
import { ImportExportAPI } from "../../../index";
import { isBlobLike } from "./factories";

describe("模板下载接口测试 - ImportExportAPI.downloadTemplate", () => {
  describe("GET /api/v1/user/template - 用户模块模板", () => {
    test("下载用户 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("user", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });

    test("下载用户 CSV 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("user", "csv");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });

    test("不传 format 默认下载 Excel 模板", async () => {
      const result = await ImportExportAPI.downloadTemplate("user");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/role/template - 角色模块模板", () => {
    test("下载角色 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("role", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });

    test("下载角色 CSV 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("role", "csv");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/dept/template - 部门模块模板", () => {
    test("下载部门 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("dept", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/menu/template - 菜单模块模板", () => {
    test("下载菜单 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("menu", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/dict/template - 字典模块模板", () => {
    test("下载字典 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("dict", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("GET /api/v1/algorithm/template - 算法模块模板", () => {
    test("下载算法 Excel 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("algorithm", "excel");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });

    test("下载算法 CSV 模板返回 Blob", async () => {
      const result = await ImportExportAPI.downloadTemplate("algorithm", "csv");

      expect(result).toBeDefined();
      expect(isBlobLike(result)).toBe(true);
      const blob = result as Blob;
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("异常场景", () => {
    test("下载不支持导入的模块(dataset)模板应抛出错误", async () => {
      await expectBizError(
        ImportExportAPI.downloadTemplate("dataset" as any, "excel"),
        ["A0710", "B0001", "ERR_BAD_REQUEST"],
      );
    });
  });
});
