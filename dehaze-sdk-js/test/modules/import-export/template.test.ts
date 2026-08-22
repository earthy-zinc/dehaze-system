import { expectBizError } from "#/utils/assertion";
import { ImportExportAPI, ImportModule } from "../../../index";
import { isBlobLike } from "./factories";

const expectTemplateBlob = async (module: ImportModule, format?: "excel" | "csv") => {
  const result = await ImportExportAPI.downloadTemplate(module, format);
  expect(isBlobLike(result)).toBe(true);
  const blob = result as Blob;
  expect(blob.size).toBeGreaterThan(0);
};

describe("模板下载接口测试 - ImportExportAPI.downloadTemplate", () => {
  describe("GET /api/v1/user/template - 用户模块模板", () => {
    test("下载用户 Excel 模板返回 Blob", () => expectTemplateBlob("user", "excel"));
    test("下载用户 CSV 模板返回 Blob", () => expectTemplateBlob("user", "csv"));
    test("不传 format 默认下载 Excel 模板", () => expectTemplateBlob("user"));
  });

  describe("GET /api/v1/role/template - 角色模块模板", () => {
    test("下载角色 Excel 模板返回 Blob", () => expectTemplateBlob("role", "excel"));
    test("下载角色 CSV 模板返回 Blob", () => expectTemplateBlob("role", "csv"));
  });

  describe("GET /api/v1/dept/template - 部门模块模板", () => {
    test("下载部门 Excel 模板返回 Blob", () => expectTemplateBlob("dept", "excel"));
  });

  describe("GET /api/v1/menu/template - 菜单模块模板", () => {
    test("下载菜单 Excel 模板返回 Blob", () => expectTemplateBlob("menu", "excel"));
  });

  describe("GET /api/v1/dict/template - 字典模块模板", () => {
    test("下载字典 Excel 模板返回 Blob", () => expectTemplateBlob("dict", "excel"));
  });

  describe("GET /api/v1/algorithm/template - 算法模块模板", () => {
    test("下载算法 Excel 模板返回 Blob", () => expectTemplateBlob("algorithm", "excel"));
    test("下载算法 CSV 模板返回 Blob", () => expectTemplateBlob("algorithm", "csv"));
  });

  describe("异常场景", () => {
    test("下载不支持导入的模块(dataset)模板应抛出错误", async () => {
      await expectBizError(ImportExportAPI.downloadTemplate("dataset" as any, "excel"), [
        "A0710",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
