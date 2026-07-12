import { FileAPI } from "../../../index";
import { login, logout } from "#/utils/auth";
import { expectBizErrorOrUndefined } from "#/utils/assertion";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

describe("文件管理接口测试", () => {
  const uploadedFileIds: number[] = [];
  let testFilePath: string;
  let testFileMd5: string;

  beforeAll(async () => {
    await login();

    // 创建测试文件
    const testDir = path.join(__dirname, "../../../temp");
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    testFilePath = path.join(testDir, `test_file_${Date.now()}.txt`);
    const testContent = "This is a test file for file upload testing.";
    fs.writeFileSync(testFilePath, testContent);

    // 计算 MD5
    const buffer = fs.readFileSync(testFilePath);
    testFileMd5 = crypto.createHash("md5").update(buffer).digest("hex");
  }, 30000);

  afterAll(async () => {
    // 清理上传的文件
    for (const fileId of uploadedFileIds) {
      try {
        await FileAPI.deleteById(fileId);
      } catch (e) {
        // 忽略清理失败
      }
    }

    // 删除本地测试文件
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }

    await logout();
  });

  describe("GET /api/v1/files/check - 文件上传检查", () => {
    test("正向测试：检查不存在文件的MD5应返回空", async () => {
      const nonExistentMd5 = "nonexistent_" + Date.now();
      const result = await FileAPI.uploadCheck(nonExistentMd5);

      // 后端返回 data:null，Jackson 省略 null 字段，SDK 解析为 undefined
      expect(result).toBeUndefined();
    });

    test("正向测试：检查已存在文件的MD5应返回文件信息", async () => {
      // 先上传一个文件
      const file = fs.readFileSync(testFilePath);
      const blob = new Blob([file]);
      const formFile = new File([blob], path.basename(testFilePath), { type: "text/plain" });

      const uploadResult = await FileAPI.upload(formFile);
      expect(uploadResult).toBeDefined();
      expect(uploadResult.id).toBeDefined();
      uploadedFileIds.push(uploadResult.id);

      // 检查已上传文件的 MD5
      const result = await FileAPI.uploadCheck(testFileMd5);

      expect(result).not.toBeNull();
      expect(result?.id).toBeDefined();
      expect(result?.url).toBeTruthy();
    });
  });

  describe("POST /api/v1/files - 上传文件", () => {
    test("正向测试：上传文件并验证返回数据", async () => {
      const file = fs.readFileSync(testFilePath);
      const blob = new Blob([file]);
      const formFile = new File([blob], path.basename(testFilePath), { type: "text/plain" });

      const result = await FileAPI.upload(formFile);

      expect(result).toBeDefined();
      expect(result.name).toBeTruthy();
      expect(result.path).toBeTruthy();
      expect(result.url).toBeTruthy();
      expect(result.id).toBeDefined();
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传文件并指定modelId", async () => {
      const file = fs.readFileSync(testFilePath);
      const blob = new Blob([file]);
      const formFile = new File([blob], path.basename(testFilePath), { type: "text/plain" });

      const result = await FileAPI.upload(formFile, 1);

      expect(result).toBeDefined();
      expect(result.name).toBeTruthy();
      expect(result.path).toBeTruthy();
      expect(result.url).toBeTruthy();
      expect(result.id).toBeDefined();
      uploadedFileIds.push(result.id);
    });

    test("参数校验：未提供文件应抛出业务错误", async () => {
      await expectBizErrorOrUndefined(FileAPI.upload(null as any), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/files - 删除文件", () => {
    test("正向测试：删除文件后MD5检查应返回null", async () => {
      // 先上传一个文件
      const file = fs.readFileSync(testFilePath);
      const blob = new Blob([file]);
      const formFile = new File([blob], path.basename(testFilePath), { type: "text/plain" });

      const uploadResult = await FileAPI.upload(formFile);
      expect(uploadResult).toBeDefined();
      expect(uploadResult.id).toBeDefined();
      const fileId = uploadResult.id;

      // 删除文件
      await FileAPI.deleteById(fileId);

      // 验证文件已删除（MD5检查应返回空）
      const result = await FileAPI.uploadCheck(testFileMd5);
      expect(result).toBeUndefined();
    });

    test("异常测试：删除不存在的文件ID", async () => {
      const nonExistId = 999999999;
      // 删除不存在的文件，后端可能返回成功（幂等）或错误
      await expectBizErrorOrUndefined(FileAPI.deleteById(nonExistId), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });
});
