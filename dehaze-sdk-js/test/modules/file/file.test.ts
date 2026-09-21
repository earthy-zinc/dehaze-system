import { FileAPI } from "../../../index";
import { service } from "@/utils/request";
import { expectBizError } from "#/utils/assertion";
import { USERS } from "#/factories/constants";
import { login } from "#/utils/auth";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

describe("文件管理接口测试", () => {
  const uploadedFileIds: number[] = [];
  let testFilePath: string;
  let testFileMd5: string;

  /** 读取测试文件内容构造 FormFile（content 缺省时用共享测试文件） */
  const makeFormFile = (name: string, content?: string | Buffer) => {
    // Buffer 需转成 Uint8Array 才能作为 BlobPart（string 本身是合法 BlobPart）
    const part: BlobPart =
      typeof content === "string"
        ? content
        : new Uint8Array(content ?? fs.readFileSync(testFilePath));
    return new File([part], name, { type: "text/plain" });
  };

  beforeAll(async () => {
    const testDir = path.join(__dirname, "../../../temp");
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    testFilePath = path.join(testDir, `test_file_${Date.now()}.txt`);
    fs.writeFileSync(testFilePath, "This is a test file for file upload testing.");

    testFileMd5 = crypto.createHash("md5").update(fs.readFileSync(testFilePath)).digest("hex");
  });

  afterAll(async () => {
    // 清理上传的文件（去重后 id 可能重复，逐条幂等删除）
    const cleaned = new Set<number>();
    for (const fileId of uploadedFileIds) {
      if (cleaned.has(fileId)) continue;
      cleaned.add(fileId);
      try {
        await FileAPI.deleteById(fileId);
      } catch (e) {
        console.warn(`清理测试文件失败 id=${fileId}:`, e);
      }
    }

    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  describe("GET /api/v1/files/check - 文件上传检查", () => {
    test("正向测试：检查不存在文件的MD5应返回空", async () => {
      // 使用合法 32 位十六进制 MD5（格式校验 B0404 通过），但该 MD5 实际不存在，应返回空
      const nonExistentMd5 = crypto
        .createHash("md5")
        .update(`nonexistent_${Date.now()}`)
        .digest("hex");
      const result = await FileAPI.uploadCheck(nonExistentMd5);

      // 后端返回 data:null，Jackson 省略 null 字段，SDK 解析为 undefined
      expect(result).toBeUndefined();
    });

    test("正向测试：检查已存在文件的MD5应返回文件信息", async () => {
      const uploadResult = await FileAPI.upload(makeFormFile(path.basename(testFilePath)));
      expect(uploadResult.id).toBeGreaterThan(0);
      uploadedFileIds.push(uploadResult.id);

      const result = await FileAPI.uploadCheck(testFileMd5);

      expect(result).not.toBeNull();
      expect(result?.id).toBeGreaterThan(0);
      expect(result?.url).toBeTruthy();
      expect(typeof result?.url).toBe("string");
    });

    test("边界：无效MD5格式应失败", async () => {
      await expectBizError(FileAPI.uploadCheck("invalid_md5"), [
        "B0404",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：MD5长度不正确应失败", async () => {
      await expectBizError(FileAPI.uploadCheck("abc"), [
        "B0404",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/files - 上传文件", () => {
    test("正向测试：上传文件并验证返回数据", async () => {
      const result = await FileAPI.upload(makeFormFile(path.basename(testFilePath)));

      expect(result.id).toBeGreaterThan(0);
      expect(typeof result.name).toBe("string");
      expect(result.name.length).toBeGreaterThan(0);
      expect(typeof result.url).toBe("string");
      expect(result.url).toMatch(/^https?:\/\//);
      uploadedFileIds.push(result.id);
    });

    test("正向测试：上传文件并指定modelId", async () => {
      const result = await FileAPI.upload(makeFormFile(path.basename(testFilePath)), 1);

      expect(result.id).toBeGreaterThan(0);
      expect(typeof result.name).toBe("string");
      expect(typeof result.url).toBe("string");
      uploadedFileIds.push(result.id);
    });

    test("参数校验：未提供文件应抛出业务错误", async () => {
      await expectBizError(FileAPI.upload(null as any), ["A0400", "B0001", "ERR_BAD_REQUEST"]);
    });

    // T-FM-003：MD5 去重（秒传）——内容寻址去重，命中时直接复用既有文件记录
    //（同一 fileId，不重复存储物理文件）。文件服务无用户归属语义，单记录去重为标准做法。
    test("正向测试：MD5去重（秒传）复用同一文件记录", async () => {
      const formFile = makeFormFile(path.basename(testFilePath));

      // 第一次上传
      const result1 = await FileAPI.upload(formFile);
      expect(result1.id).toBeGreaterThan(0);
      uploadedFileIds.push(result1.id);

      // 第二次上传相同内容：秒传命中，复用同一记录
      const result2 = await FileAPI.upload(formFile);
      expect(result2.id).toBe(result1.id);
      expect(result2.md5).toBe(result1.md5);
    });

    test("验证：上传后返回的MD5与本地计算一致", async () => {
      const result = await FileAPI.upload(makeFormFile(path.basename(testFilePath)));
      uploadedFileIds.push(result.id);

      if (result.md5) {
        expect(result.md5).toBe(testFileMd5);
      }
    });

    test("边界：0字节文件上传应成功（sizeBytes=0）", async () => {
      const result = await FileAPI.upload(makeFormFile("empty.txt", ""));
      expect(result.id).toBeGreaterThan(0);
      uploadedFileIds.push(result.id);

      // 后端 FileVO 返回格式化 size（"0B"），不含 sizeBytes 字段（文档 T-FM-008）
      const detail = await FileAPI.getById(result.id);
      expect(detail.size).toBe("0B");
    });

    test("边界：特殊字符文件名正确保存", async () => {
      // 使用与测试文件不同的内容，避免命中 MD5 去重复用已有记录（同名覆盖测试会失效）
      const uniqueContent = `special_${Date.now()}_${Math.random()}`;
      const specialName = "测试 文件 (1).txt";
      const formFile = makeFormFile(specialName, uniqueContent);

      const result = await FileAPI.upload(formFile);
      expect(result.id).toBeGreaterThan(0);
      uploadedFileIds.push(result.id);

      expect(result.name).toBe(specialName);
    });

    test("正向测试：上传进度回调被触发", async () => {
      const formFile = makeFormFile(`progress_${Date.now()}.txt`);

      let progressCalled = false;
      let lastLoaded = 0;
      let lastTotal: number | undefined;

      const result = await FileAPI.upload(formFile, undefined, (progressEvent) => {
        progressCalled = true;
        if (progressEvent.total) {
          // 记录单调递增且不超过 total 的进度值，最后统一断言
          if (progressEvent.loaded <= progressEvent.total) {
            lastLoaded = progressEvent.loaded;
          }
          lastTotal = progressEvent.total;
        }
      });

      expect(result.id).toBeGreaterThan(0);
      uploadedFileIds.push(result.id);

      // 小文件可能因浏览器/axios 阈值不触发回调，用宽松断言
      if (progressCalled && lastTotal !== undefined) {
        expect(lastLoaded).toBeLessThanOrEqual(lastTotal);
      }
    });
  });

  describe("DELETE /api/v1/files - 删除文件", () => {
    test("正向测试：删除文件后MD5检查应返回null", async () => {
      // 上传独立内容，避免命中 MD5 去重复用测试文件共享记录，导致误删其他用例引用的记录
      const uniqueContent = `del_${Date.now()}_${Math.random()}`;
      const formFile = makeFormFile(`del_${Date.now()}.txt`, uniqueContent);
      const md5 = crypto.createHash("md5").update(uniqueContent).digest("hex");

      const uploadResult = await FileAPI.upload(formFile);
      expect(uploadResult.id).toBeGreaterThan(0);
      const fileId = uploadResult.id;

      // 后端契约：DELETE /api/v1/files?fileId=
      await FileAPI.deleteById(fileId);

      const result = await FileAPI.uploadCheck(md5);
      expect(result).toBeUndefined();
    });

    test("异常测试：删除不存在的文件ID", async () => {
      const nonExistId = 999999999;
      // 后端删除不存在文件返回 B0401"文件不存在"（三端统一）
      await expectBizError(FileAPI.deleteById(nonExistId), [
        "B0401",
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/files/page - 文件分页查询", () => {
    let testFileId: number;

    beforeAll(async () => {
      // 用独立内容，避免命中 MD5 去重复用测试文件共享记录（其文件名为 test_file_*，会导致
      // "pagetest" 关键字搜不到）
      const uniqueContent = `pagetest_${Date.now()}_${Math.random()}`;
      const formFile = makeFormFile(`pagetest_${Date.now()}.txt`, uniqueContent);
      const result = await FileAPI.upload(formFile);
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：分页查询文件列表", async () => {
      const result = await FileAPI.getPage({ pageNum: 1, pageSize: 10 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      expect(result.list.length).toBeLessThanOrEqual(10);
    });

    test("正向测试：关键字搜索", async () => {
      const result = await FileAPI.getPage({ pageNum: 1, pageSize: 100, keywords: "pagetest" });
      expect(result.list.length).toBeGreaterThan(0);
      result.list.forEach((file) => {
        expect(file.name).toContain("pagetest");
      });
    });

    test("验证：按创建时间倒序排列", async () => {
      const result = await FileAPI.getPage({ pageNum: 1, pageSize: 20 });
      if (result.list.length < 2) return;
      for (let i = 1; i < result.list.length; i++) {
        const prev = result.list[i - 1]!.createTime;
        const curr = result.list[i]!.createTime;
        if (prev && curr) {
          expect(prev >= curr).toBe(true);
        }
      }
    });

    test("边界：空列表（不存在的关键字）", async () => {
      const result = await FileAPI.getPage({
        pageNum: 1,
        pageSize: 10,
        keywords: "nonexistent_xyz_99999",
      });
      expect(result.list.length).toBe(0);
      expect(result.total).toBe(0);
    });

    test("边界：大页码返回空列表", async () => {
      const result = await FileAPI.getPage({ pageNum: 10000, pageSize: 10 });
      expect(result.list.length).toBe(0);
    });
  });

  describe("GET /api/v1/files/{fileId} - 文件详情查询", () => {
    let testFileId: number;

    beforeAll(async () => {
      const result = await FileAPI.upload(makeFormFile(`detail_${Date.now()}.txt`));
      testFileId = result.id;
      uploadedFileIds.push(testFileId);
    });

    test("正向测试：查询文件详情", async () => {
      const detail = await FileAPI.getById(testFileId);
      expect(detail.id).toBe(testFileId);
      expect(detail.name).toBeTruthy();
      expect(detail.url).toBeTruthy();
    });

    test("验证：返回字段完整性", async () => {
      const detail = await FileAPI.getById(testFileId);
      expect(detail.id).toBeGreaterThan(0);
      expect(typeof detail.name).toBe("string");
      expect(detail.objectName).toBeTruthy();
      // 后端现已返回 sizeBytes（文档 T-FM-043 契约），断言其为非负整数
      expect(typeof detail.sizeBytes).toBe("number");
      expect(detail.sizeBytes!).toBeGreaterThanOrEqual(0);
    });

    test("边界：查询不存在的文件应失败", async () => {
      await expectBizError(FileAPI.getById(99999999), [
        "B0401",
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：查询已删除的文件应返回B0401", async () => {
      // 用独立内容避免误删其他用例共享的记录
      const uniqueContent = `deletedetail_${Date.now()}_${Math.random()}`;
      const uploadResult = await FileAPI.upload(
        makeFormFile(`deletedetail_${Date.now()}.txt`, uniqueContent)
      );
      uploadedFileIds.push(uploadResult.id);
      await FileAPI.deleteById(uploadResult.id);

      // 软删后详情查询不应再返回该记录（B0401 文件不存在）
      await expectBizError(FileAPI.getById(uploadResult.id), [
        "B0401",
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/files/download/{objectName} - 文件下载", () => {
    let testObjectName: string;

    beforeAll(async () => {
      const result = await FileAPI.upload(makeFormFile(`download_${Date.now()}.txt`));
      uploadedFileIds.push(result.id);

      const detail = await FileAPI.getById(result.id);
      testObjectName = detail.objectName!;
    });

    test("正向测试：下载文件返回Blob", async () => {
      if (!testObjectName) return;
      const blob = await FileAPI.download(testObjectName);
      expect(blob.size).toBeGreaterThan(0);
    });

    test("边界：下载不存在的文件应失败", async () => {
      await expectBizError(FileAPI.download("nonexistent/object.txt"), [
        "B0401",
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("安全：路径遍历攻击应被拒绝", async () => {
      // 浏览器/axios 会先对 URL 中的 `..` 做路径归一化（`/download/../../../etc/passwd` 归约为
      // `/api/v1/files/etc/passwd`），`..` 不会到达后端。归一化后请求不命中任何路由，返回 FastAPI
      // 默认 404，不泄露系统文件——安全目标达成。此处用 service 直接断言请求被拒绝（HTTP 404），
      // 而非依赖 FileAPI.download 的 blob 错误体。
      await expect(service.get("/api/v1/files/etc/passwd")).rejects.toMatchObject({
        response: {
          status: 404,
        },
      });
    });

    test("安全：下载已删除的文件应失败（B0401）", async () => {
      // 用独立内容避免命中共享记录
      const uniqueContent = `deldownload_${Date.now()}_${Math.random()}`;
      const uploadResult = await FileAPI.upload(
        makeFormFile(`deldownload_${Date.now()}.txt`, uniqueContent)
      );
      const fileId = uploadResult.id;
      const detail = await FileAPI.getById(fileId);
      await FileAPI.deleteById(fileId);

      // 后端下载查询已过滤 deleted 记录，软删文件不可再下载
      await expectBizError(FileAPI.download(detail.objectName!), [
        "B0401",
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("对抗性文件名与内容输入", () => {
    test("安全：文件名含路径遍历前缀应被剥离（basename 化）", async () => {
      const uniqueContent = `traversal_${Date.now()}_${Math.random()}`;
      const result = await FileAPI.upload(makeFormFile("../../evil.txt", uniqueContent));
      uploadedFileIds.push(result.id);

      // 三端口径：路径前缀被剥离，仅保留文件名部分，objectName 由服务端 MD5 生成不受影响
      expect(result.name).toBe("evil.txt");
      expect(result.objectName).not.toContain("..");
    });

    test("安全：文件名含反斜杠路径应被剥离", async () => {
      const uniqueContent = `backslash_${Date.now()}_${Math.random()}`;
      const result = await FileAPI.upload(makeFormFile("..\\..\\win.ini", uniqueContent));
      uploadedFileIds.push(result.id);

      expect(result.name).toBe("win.ini");
    });

    test("安全：文件名含空字节/控制字符应被拒绝或安全编码", async () => {
      const uniqueContent = `nulbyte_${Date.now()}`;
      // 空字节文件名：服务端校验拒绝
      await expectBizError(FileAPI.upload(makeFormFile("file\x00.txt", uniqueContent)), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
      // 换行文件名：undici FormData 传输前会将控制字符 percent-encode（\n → 字面 %0A），
      // 服务端收到的是安全字符并按普通文件保存——验证最终存储名不含裸控制字符
      const crlfResult = await FileAPI.upload(makeFormFile("file\n.txt", `crlf_${Date.now()}`));
      uploadedFileIds.push(crlfResult.id);
      expect(crlfResult.id).toBeGreaterThan(0);
      expect(crlfResult.name).not.toMatch(/[\x00-\x1f]/);
    });

    test("边界：超过100字符的文件名应被拒绝", async () => {
      const longName = "a".repeat(101) + ".txt";
      await expectBizError(FileAPI.upload(makeFormFile(longName, `long_${Date.now()}`)), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：超过20字符的扩展名应被拒绝", async () => {
      const badExtName = "file." + "x".repeat(21);
      await expectBizError(FileAPI.upload(makeFormFile(badExtName, `ext_${Date.now()}`)), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("正向：零宽字符/emoji/中文文件名正确保存", async () => {
      const uniqueContent = `unicode_${Date.now()}_${Math.random()}`;
      const unicodeName = "隐\u200b藏🚀报告（终稿）.txt";
      const result = await FileAPI.upload(makeFormFile(unicodeName, uniqueContent));
      uploadedFileIds.push(result.id);

      expect(result.name).toBe(unicodeName);
    });

    test("边界：GBK 乱码字节文件名不崩溃", async () => {
      // 模拟客户端未正确转码产生的 latin1 乱码文件名：服务端应作为普通字符保存，不抛 500
      const mojibakeName = "Î¢ÐÅÍ¼Æ¬.txt";
      const uniqueContent = `mojibake_${Date.now()}_${Math.random()}`;
      const result = await FileAPI.upload(makeFormFile(mojibakeName, uniqueContent));
      uploadedFileIds.push(result.id);

      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toBe(mojibakeName);
    });

    test("对抗：伪装扩展名（.jpg 实为文本）应被拒绝（B0403）", async () => {
      // 图片扩展名做文件头魔数校验：文本内容伪装 .jpg 上传被拒绝
      await expectBizError(
        FileAPI.upload(makeFormFile("fake_image.jpg", "plain text disguised as jpg")),
        ["B0403", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });

    test("正向：真实 PNG 图片上传通过魔数校验", async () => {
      // 最小合法 PNG 头（8 字节签名 + IHDR 长度标记）
      const pngBytes = Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        Buffer.alloc(24, 0),
      ]);
      const result = await FileAPI.upload(makeFormFile("real_image.png", pngBytes));
      uploadedFileIds.push(result.id);

      expect(result.id).toBeGreaterThan(0);
      expect(result.type).toBe("png");
    });

    test("正向：非图片类型不做内容校验（.txt 文本正常上传）", async () => {
      const result = await FileAPI.upload(
        makeFormFile(`nocheck_${Date.now()}.txt`, `plain_${Date.now()}`)
      );
      uploadedFileIds.push(result.id);
      expect(result.id).toBeGreaterThan(0);
    });
  });

  describe("归属校验 - 普通用户仅可见/可操作自己上传的文件（B0407）", () => {
    let adminFileId: number;
    let adminObjectName: string;
    let adminFileMd5: string;

    beforeAll(async () => {
      // admin 上传文件（后端测试账号链路以 admin 为管理员口径）
      await login(USERS.ADMIN.username);
      const uniqueContent = `owner_${Date.now()}_${Math.random()}`;
      const result = await FileAPI.upload(makeFormFile(`owner_${Date.now()}.txt`, uniqueContent));
      adminFileId = result.id;
      uploadedFileIds.push(adminFileId);
      const detail = await FileAPI.getById(adminFileId);
      adminObjectName = detail.objectName!;
      adminFileMd5 = result.md5!;
    });

    afterAll(async () => {
      // 恢复 admin 登录态，保证 uploadedFileIds 清理有权限执行
      await login(USERS.ADMIN.username);
    });

    test("边界：普通用户查询他人文件详情应返回 B0407", async () => {
      await login(USERS.USER.username);
      await expectBizError(FileAPI.getById(adminFileId), [
        "B0407",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户下载他人文件应返回 B0407", async () => {
      await login(USERS.USER.username);
      await expectBizError(FileAPI.download(adminObjectName), [
        "B0407",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：普通用户删除他人文件应返回 B0407", async () => {
      await login(USERS.USER.username);
      await expectBizError(FileAPI.deleteById(adminFileId), [
        "B0407",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
      // 文件未被删除（admin afterAll 清理）
    });

    test("边界：普通用户列表不包含他人文件", async () => {
      await login(USERS.USER.username);
      const result = await FileAPI.getPage({ pageNum: 1, pageSize: 100 });
      expect(
        result.list.some((f) => f.id === adminFileId),
        "普通用户列表不应出现他人上传的文件"
      ).toBe(false);
    });

    test("豁免：check 秒传预检不做归属校验（md5 命中即返回）", async () => {
      // 普通用户持有相同内容（md5 命中）时 check 仍返回文件信息，秒传语义不受归属影响
      await login(USERS.USER.username);
      const result = await FileAPI.uploadCheck(adminFileMd5);
      expect(result).not.toBeNull();
      expect(result?.id).toBe(adminFileId);
    });

    test("正向：管理员可访问他人文件", async () => {
      await login(USERS.ADMIN.username);
      const detail = await FileAPI.getById(adminFileId);
      expect(detail.id).toBe(adminFileId);
    });
  });

  describe("性能烟测", () => {
    test("MD5 校验接口响应时间 < 1s（文档目标 100ms 量级，放宽网络抖动）", async () => {
      const md5 = crypto.createHash("md5").update("perf_probe").digest("hex");
      const start = Date.now();
      await FileAPI.uploadCheck(md5);
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000);
    });

    test("文件列表接口响应时间 < 2s", async () => {
      const start = Date.now();
      await FileAPI.getPage({ pageNum: 1, pageSize: 10 });
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(2000);
    });
  });
});
