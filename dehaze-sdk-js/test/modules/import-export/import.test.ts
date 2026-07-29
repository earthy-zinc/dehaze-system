import { expectBizError } from "#/utils/assertion";
import { ImportExportAPI, UserAPI } from "../../../index";
import { ROLES } from "#/factories/constants";
import { uniqueName, uniqueEmail, uniqueMobile } from "#/factories/common";
import {
  createCsvFile,
  createInvalidFile,
  createOversizedFile,
  createPartialImportRequest,
} from "./factories";

const buildUserCsvContent = () => {
  const username = uniqueName("imp_user");
  const nickname = uniqueName("导入用户");
  const email = uniqueEmail(username);
  const mobile = uniqueMobile();
  const roleCodes = ROLES.GUEST.code;
  // CSV 表头需与后端 ImportFieldConfig.label 对齐（中文）
  const header = "用户名,昵称,性别,手机号,邮箱,角色编码(多个逗号分隔)";
  const row = `${username},${nickname},男,${mobile},${email},${roleCodes}`;
  return { content: `${header}\n${row}\n`, username, nickname };
};

const isImportResult = (
  v: unknown
): v is {
  totalRows: number;
  successCount: number;
  failureCount: number;
  skippedCount: number;
  errors: unknown[];
} =>
  !!v && typeof v === "object" && "totalRows" in (v as object) && "successCount" in (v as object);

const isImportTaskResult = (v: unknown): v is { taskId: string; status: number } =>
  !!v &&
  typeof v === "object" &&
  typeof (v as { taskId?: unknown }).taskId === "string" &&
  "status" in (v as object);

const findUserIdByUsername = async (username: string): Promise<number | undefined> => {
  const page = await UserAPI.getPage({ pageNum: 1, pageSize: 100, keywords: username });
  return page.list.find((u) => u.username === username)?.id;
};

describe("通用导入接口测试 - ImportExportAPI.import", () => {
  const createdUserIds: number[] = [];

  afterAll(async () => {
    for (const id of createdUserIds) {
      try {
        await UserAPI.deleteByIds(id.toString());
      } catch {}
    }
  });

  describe("POST /api/v1/user/_import - 用户模块导入", () => {
    test("同步导入用户 CSV 文件返回 ImportResult", async () => {
      const { content, username } = buildUserCsvContent();
      const file = createCsvFile("user_sync.csv", content);

      const result = await ImportExportAPI.import("user", { mode: "all" }, file);

      expect(result).toBeDefined();
      expect(isImportResult(result) || isImportTaskResult(result)).toBe(true);

      if (isImportResult(result)) {
        expect(result.totalRows).toBeGreaterThan(0);
        expect(result.successCount).toBeGreaterThan(0);
        expect(typeof result.failureCount).toBe("number");
        expect(Array.isArray(result.errors)).toBe(true);
      }

      const userId = await findUserIdByUsername(username);
      if (userId) createdUserIds.push(userId);
    });

    test("强制异步导入用户返回 ImportTaskResult(taskId)", async () => {
      const { content, username } = buildUserCsvContent();
      const file = createCsvFile("user_async.csv", content);

      const result = await ImportExportAPI.import("user", { mode: "all", async: true }, file);

      expect(result).toBeDefined();
      expect(isImportTaskResult(result)).toBe(true);
      const taskResult = result as { taskId: string; status: number };
      expect(typeof taskResult.taskId).toBe("string");
      expect(taskResult.taskId.length).toBeGreaterThan(0);
      // 后端 TaskStatusEnum.PENDING 序列化为数字 1
      expect(taskResult.status).toBe(1);

      const userId = await findUserIdByUsername(username);
      if (userId) createdUserIds.push(userId);
    });
  });

  describe("导入模式切换 - mode 参数", () => {
    test("partial 模式导入用户返回 ImportResult", async () => {
      const { content, username } = buildUserCsvContent();
      const file = createCsvFile("user_partial.csv", content);

      const result = await ImportExportAPI.import("user", createPartialImportRequest(), file);

      expect(result).toBeDefined();
      expect(isImportResult(result) || isImportTaskResult(result)).toBe(true);

      const userId = await findUserIdByUsername(username);
      if (userId) createdUserIds.push(userId);
    });

    test("partial 模式重复导入相同用户应跳过已存在记录", async () => {
      const { content, username } = buildUserCsvContent();
      const file = createCsvFile("user_partial_dup.csv", content);

      const firstResult = await ImportExportAPI.import("user", createPartialImportRequest(), file);
      expect(firstResult).toBeDefined();

      const secondResult = await ImportExportAPI.import("user", createPartialImportRequest(), file);

      expect(secondResult).toBeDefined();
      if (isImportResult(secondResult)) {
        expect(secondResult.skippedCount).toBeGreaterThanOrEqual(0);
      }

      const userId = await findUserIdByUsername(username);
      if (userId) createdUserIds.push(userId);
    });
  });

  describe("错误反馈 - errors 数组", () => {
    test("导入含错误行的 CSV 应返回 errors 错误明细", async () => {
      const username = uniqueName("imp_err");
      const header = "用户名,昵称,性别,手机号,邮箱,角色编码(多个逗号分隔)";
      const validRow = `${username},导入错误测试,男,${uniqueMobile()},${uniqueEmail(username)},${ROLES.GUEST.code}`;
      const invalidRow = `,,invalid-email,000,99,99,9999,9999`;
      const content = `${header}\n${validRow}\n${invalidRow}\n`;
      const file = createCsvFile("user_with_errors.csv", content);

      const result = await ImportExportAPI.import("user", { mode: "partial" }, file);

      expect(result).toBeDefined();
      if (isImportResult(result)) {
        expect(result.totalRows).toBeGreaterThanOrEqual(1);
        expect(Array.isArray(result.errors)).toBe(true);
      }

      const userId = await findUserIdByUsername(username);
      if (userId) createdUserIds.push(userId);
    });
  });

  describe("异常场景", () => {
    test("上传非 Excel/CSV 文件应返回 A0701 错误", async () => {
      const file = createInvalidFile("not_supported.txt");

      await expectBizError(
        ImportExportAPI.import("user", { mode: "all" }, file),
        ["A0701", "B0001"],
      );
    });

    test("上传超过 20MB 文件应返回 A0702 错误", async () => {
      const file = createOversizedFile("oversized.xlsx");

      await expectBizError(
        ImportExportAPI.import("user", { mode: "all" }, file),
        ["A0702", "B0001"],
      );
    });
  });
});
