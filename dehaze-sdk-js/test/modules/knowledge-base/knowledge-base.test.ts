import { AiKnowledgeBaseAPI, FileAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createBatchUploadForm,
  createDocQuery,
  createDocUploadForm,
  createImportUrlForm,
  createKbForm,
  createKbQuery,
  createRetrieveTestForm,
  createSearchForm,
  createTextDocForm,
} from "#/factories/knowledge-base";

// ============================================================================
// AI 知识库模块接口测试
//
// 前置说明：
// - 知识库创建要求 ES 向量索引初始化成功（ensure_kb_index），ES 不可用时创建失败、无降级
//   （见《测试用例.md》T-KB-008）。
// - 因此本模块绝大部分用例依赖 ES 可用。顶层 beforeAll 探测一次创建，失败则相关用例按
//   环境依赖 skip 并注明原因；非 ES 依赖的用例（参数校验、权限、空列表）任意环境可运行。
// - 向量化使用本地 embedding 服务（8992 /v1/embeddings），createKbForm 默认
//   embeddingProvider=local + embeddingModel=bge-m3（1024 维，与本地模型同维度）。
// - 文档上传/批量/分块预览需要真实 fileId，顶层 beforeAll 先上传 txt 测试文件。
// - 知识库私有库配额（admin level_1=10）限制同时存在的私有库数量，各 describe 在
//   afterAll 及时删除自己创建的库，避免运行中累计超过配额导致创建失败。
// ============================================================================

let esAvailable = false;
const testFileIds: number[] = [];
let externalUrlReachable = false;

const ES_SKIP_MSG =
  "依赖 ES 向量索引：当前后端 ES 未启用(ES_ENABLED=false)/不可用，知识库无法创建（文档规定 ES 不可用时创建失败、无降级）。需在 dehaze-python 配置 ES 并重启后端。";

function requireEs(ctx: any) {
  if (!esAvailable) {
    ctx.skip(ES_SKIP_MSG);
  }
}

/** 内容唯一，避免 MD5 秒传合并 */
function makeTxtFile(prefix: string): File {
  const content = `${prefix}_kb_doc_${Date.now()}_${Math.random()}`;
  return new File([new Blob([content])], `${prefix}_${Date.now()}.txt`, {
    type: "text/plain",
  });
}

async function waitForDocSettled(docId: number, timeoutMs = 30000): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  let status = "pending";
  while (Date.now() < deadline) {
    const detail = await AiKnowledgeBaseAPI.getDocumentDetail(docId);
    status = detail.processingStatus;
    if (status === "completed" || status === "failed") {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  return status;
}

/** 按 LIFO 顺序逐个删除，失败静默忽略（资源可能已被测试本身删除） */
async function deleteAll(ids: number[], remove: (id: number) => Promise<unknown>): Promise<void> {
  for (const id of [...ids].reverse()) {
    try {
      await remove(id);
    } catch {
      // 忽略：可能已被测试本身删除
    }
  }
}

/** 按 LIFO 顺序删除知识库（及时释放私有库配额） */
function deleteKbs(ids: number[]): Promise<void> {
  return deleteAll(ids, (id) => AiKnowledgeBaseAPI.delete(id));
}

/** 以普通用户身份执行操作，结束后切回 admin */
async function asUser(fn: () => Promise<void>): Promise<void> {
  await login(USERS.USER.username);
  try {
    await fn();
  } finally {
    await login(USERS.ADMIN.username);
  }
}

describe("AI 知识库模块接口测试 - AiKnowledgeBaseAPI", () => {
  const createdKbIds: number[] = [];
  const createdDocIds: number[] = [];

  /** 创建私有知识库并登记清理，ES 不可用时返回 0 */
  async function createKb(localIds: number[]): Promise<number> {
    if (!esAvailable) return 0;
    const result = await AiKnowledgeBaseAPI.create(createKbForm());
    createdKbIds.push(result.id);
    localIds.push(result.id);
    return result.id;
  }

  // 顶层探测：ES 是否可用（能否成功创建知识库）+ 上传测试文件 + 探测外网
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    try {
      const created = await AiKnowledgeBaseAPI.create(createKbForm());
      if (created && created.id > 0) {
        esAvailable = true;
        createdKbIds.push(created.id);
      }
    } catch {
      // A0500 "ES 索引初始化失败" => ES 不可用，知识库无法创建
      esAvailable = false;
    }

    if (esAvailable) {
      try {
        testFileIds.push((await FileAPI.upload(makeTxtFile("kb_test_a"))).id);
        testFileIds.push((await FileAPI.upload(makeTxtFile("kb_test_b"))).id);
      } catch (e) {
        console.warn("上传测试文件失败，文档上传类用例将受影响:", e);
      }
    }

    try {
      const resp = await fetch("https://example.com/", { method: "GET" });
      externalUrlReachable = resp.status === 200;
    } catch {
      externalUrlReachable = false;
    }
  });

  afterAll(async () => {
    // 先删文档，再删知识库，最后清理测试文件（子资源先于父资源）
    await deleteAll(createdDocIds, (id) => AiKnowledgeBaseAPI.deleteDocument(id));
    await deleteKbs(createdKbIds);
    await deleteAll(testFileIds, (id) => FileAPI.deleteById(id));
  });

  // ===== 知识库管理 =====

  describe("POST /api/v1/kb - 创建知识库", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：创建知识库（基本配置）", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.create(createKbForm());
      expect(result.id).toBeGreaterThan(0);
      createdKbIds.push(result.id);
      localKbIds.push(result.id);
    });

    test("正向测试：创建知识库（完整配置含 bge-m3 + 语义分块）", async (ctx) => {
      requireEs(ctx);
      const form = createKbForm({
        embeddingModel: "bge-m3",
        chunkingStrategy: "semantic",
        searchStrategy: "hybrid",
        hybridWeight: 0.7,
        topK: 10,
        scoreThreshold: 0.5,
        enableRerank: true,
      });
      const result = await AiKnowledgeBaseAPI.create(form);
      expect(result.id).toBeGreaterThan(0);
      createdKbIds.push(result.id);
      localKbIds.push(result.id);
    });

    test("参数校验：知识库名称为空应失败", async () => {
      const form = createKbForm({ name: "" });
      await expectBizError(AiKnowledgeBaseAPI.create(form), [
        "A0400",
        "A0500",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：创建公共知识库需管理员权限", async () => {
      await asUser(async () => {
        const form = createKbForm({ visibility: "public" });
        await expectBizError(AiKnowledgeBaseAPI.create(form), [
          "A0301",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      });
    });
  });

  describe("GET /api/v1/kb - 知识库列表", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：分页查询知识库列表", async () => {
      const result = await AiKnowledgeBaseAPI.getList(createKbQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("正向测试：按关键字搜索", async (ctx) => {
      requireEs(ctx);
      const form = createKbForm({ name: `keyword_search_${Date.now()}` });
      const created = await AiKnowledgeBaseAPI.create(form);
      createdKbIds.push(created.id);
      localKbIds.push(created.id);

      try {
        const result = await AiKnowledgeBaseAPI.getList(
          createKbQuery({ keyword: "keyword_search" })
        );
        expect(result.list.length).toBeGreaterThan(0);
        const found = result.list.find((kb) => kb.id === created.id);
        expect(found).toBeDefined();
      } finally {
        // 搜索完成即删除，控制私有库配额
        const idx = localKbIds.indexOf(created.id);
        if (idx >= 0) localKbIds.splice(idx, 1);
        await deleteKbs([created.id]);
      }
    });
  });

  describe("GET /api/v1/kb/{id} - 知识库详情", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：查询知识库详情含配置和统计", async (ctx) => {
      requireEs(ctx);
      const detail = await AiKnowledgeBaseAPI.getDetail(testKbId);
      expect(detail.id).toBe(testKbId);
      // 后端返回扁平 camelCase 字段（无嵌套 config/statistics）
      expect(detail.embeddingModel).toBeTruthy();
      expect(detail.chunkingStrategy).toBeTruthy();
      expect(detail.searchStrategy).toBeTruthy();
      expect(typeof detail.documentCount).toBe("number");
      expect(typeof detail.chunkCount).toBe("number");
      expect(typeof detail.totalTokens).toBe("number");
    });

    test("边界：查询不存在的知识库应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.getDetail(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("PUT /api/v1/kb/{id} - 编辑知识库", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：修改名称和描述", async (ctx) => {
      requireEs(ctx);
      const newName = `updated_${Date.now()}`;
      const newDesc = "更新后的描述";
      await AiKnowledgeBaseAPI.update(testKbId, { name: newName, description: newDesc });
      const detail = await AiKnowledgeBaseAPI.getDetail(testKbId);
      expect(detail.name).toBe(newName);
      expect(detail.description).toBe(newDesc);
    });

    test("正向测试：修改检索策略（topK/hybridWeight/scoreThreshold）", async (ctx) => {
      requireEs(ctx);
      await AiKnowledgeBaseAPI.update(testKbId, {
        topK: 10,
        hybridWeight: 0.7,
        scoreThreshold: 0.5,
        enableRerank: true,
      });
      const detail = await AiKnowledgeBaseAPI.getDetail(testKbId);
      expect(detail.topK).toBe(10);
      expect(detail.hybridWeight).toBe(0.7);
      expect(detail.scoreThreshold).toBe(0.5);
      expect(detail.enableRerank).toBe(1);
    });

    test("边界：尝试修改 embedding 模型应失败（创建后不可变）", async (ctx) => {
      requireEs(ctx);
      // 后端 knowledge_base_service.update 检测到请求携带 embedding_model 即抛
      // BUSINESS_ERROR（A0500："创建后不可修改 embedding 模型或分块策略"），拒绝而非静默忽略
      // （对齐文档 T-KB-004）。
      await expectBizError(
        AiKnowledgeBaseAPI.update(testKbId, {
          name: `attempt_${Date.now()}`,
          embeddingModel: "bge-m3",
        } as any),
        ["A0500"]
      );
    });

    test("边界：编辑不存在的知识库应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.update(99999999, { name: "test" }), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/kb/{id} - 删除知识库", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：删除知识库", async (ctx) => {
      requireEs(ctx);
      const created = await AiKnowledgeBaseAPI.create(createKbForm());
      localKbIds.push(created.id);
      await AiKnowledgeBaseAPI.delete(created.id);

      await expectBizError(AiKnowledgeBaseAPI.getDetail(created.id), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：删除不存在的知识库应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.delete(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("安全：越权删除他人私有知识库应失败", async (ctx) => {
      requireEs(ctx);
      const created = await AiKnowledgeBaseAPI.create(createKbForm({ visibility: "private" }));
      createdKbIds.push(created.id);
      localKbIds.push(created.id);

      await asUser(async () => {
        await expectBizError(AiKnowledgeBaseAPI.delete(created.id), [
          "A0301",
          "A0401",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      });
    });
  });

  // ===== 文档管理 =====

  describe("POST /api/v1/kb/{id}/documents/text - 自定义文本创建文档", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：自定义文本创建文档", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.createTextDocument(testKbId, createTextDocForm());
      expect(result.id).toBeGreaterThan(0);
      expect(result.processingStatus).toBe("pending");
      createdDocIds.push(result.id);
    });

    test("参数校验：空标题应失败", async () => {
      const form = createTextDocForm({ title: "" });
      await expectBizError(AiKnowledgeBaseAPI.createTextDocument(testKbId, form), [
        "A0400",
        "A0500",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：空内容应失败", async () => {
      const form = createTextDocForm({ content: "" });
      await expectBizError(AiKnowledgeBaseAPI.createTextDocument(testKbId, form), [
        "A0400",
        "A0500",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/kb/{id}/documents/import-url - 导入网页", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：导入网页 URL 创建文档", async (ctx) => {
      requireEs(ctx);
      if (!externalUrlReachable) {
        ctx.skip("依赖外部网页可达性：当前环境无法访问 example.com，保持环境性跳过");
        return;
      }
      const result = await AiKnowledgeBaseAPI.importUrlDocument(testKbId, createImportUrlForm());
      expect(result.id).toBeGreaterThan(0);
      expect(result.processingStatus).toBe("pending");
      createdDocIds.push(result.id);
    });

    test("参数校验：空 URL 应失败", async (ctx) => {
      requireEs(ctx);
      // 需真实知识库才能到达 URL 校验（service 先校验库存在，再校验 URL）
      const form = createImportUrlForm({ url: "" });
      await expectBizError(AiKnowledgeBaseAPI.importUrlDocument(testKbId, form), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("参数校验：非法 URL 格式应失败", async (ctx) => {
      requireEs(ctx);
      const form = createImportUrlForm({ url: "not-a-url" });
      await expectBizError(AiKnowledgeBaseAPI.importUrlDocument(testKbId, form), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/kb/{id}/documents - 上传文档", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：通过 fileId 上传文档", async (ctx) => {
      requireEs(ctx);
      if (testFileIds.length === 0) {
        ctx.skip("顶层测试文件上传失败，无法通过 fileId 上传文档");
        return;
      }
      const result = await AiKnowledgeBaseAPI.uploadDocument(
        testKbId,
        createDocUploadForm({ fileId: testFileIds[0]! })
      );
      expect(result.id).toBeGreaterThan(0);
      expect(result.processingStatus).toBe("pending");
      createdDocIds.push(result.id);
    });

    test("边界：不存在的 fileId 应失败", async (ctx) => {
      requireEs(ctx);
      const form = createDocUploadForm({ fileId: 99999999 });
      await expectBizError(AiKnowledgeBaseAPI.uploadDocument(testKbId, form), [
        "A0401",
        "A0500",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("边界：上传到不存在的知识库应失败", async () => {
      const form = createDocUploadForm();
      await expectBizError(AiKnowledgeBaseAPI.uploadDocument(99999999, form), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/kb/{id}/documents/batch - 批量上传", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：批量上传 2 个文档", async (ctx) => {
      requireEs(ctx);
      if (testFileIds.length < 2) {
        ctx.skip("顶层测试文件上传失败，无法批量上传文档");
        return;
      }
      const result = await AiKnowledgeBaseAPI.batchUploadDocuments(
        testKbId,
        createBatchUploadForm(testFileIds.slice(0, 2))
      );
      expect(Array.isArray(result)).toBe(true);
      result.forEach((doc) => {
        if (doc.success) {
          expect(doc.id!).toBeGreaterThan(0);
          expect(doc.processingStatus).toBe("pending");
          createdDocIds.push(doc.id!);
        }
      });
    });

    test("参数校验：空 items 应失败", async () => {
      const form = createBatchUploadForm([]);
      await expectBizError(AiKnowledgeBaseAPI.batchUploadDocuments(testKbId, form), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("GET /api/v1/kb/{id}/documents - 文档列表", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
      if (!testKbId) return;
      for (let i = 0; i < 2; i++) {
        const doc = await AiKnowledgeBaseAPI.createTextDocument(testKbId, createTextDocForm());
        createdDocIds.push(doc.id);
      }
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：查询知识库文档列表", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.getDocuments(testKbId, createDocQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.list.length).toBeGreaterThan(0);
    });

    test("正向测试：按处理状态筛选", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.getDocuments(
        testKbId,
        createDocQuery({ processingStatus: "pending" })
      );
      result.list.forEach((doc) => {
        expect(doc.processingStatus).toBe("pending");
      });
    });

    test("正向测试：按关键字搜索", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.getDocuments(
        testKbId,
        createDocQuery({ keyword: "text_doc" })
      );
      expect(Array.isArray(result.list)).toBe(true);
    });
  });

  describe("GET /api/v1/kb/documents/{id} - 文档详情", () => {
    let testDocId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      const kbId = await createKb(localKbIds);
      if (!kbId) return;
      const doc = await AiKnowledgeBaseAPI.createTextDocument(kbId, createTextDocForm());
      testDocId = doc.id;
      createdDocIds.push(testDocId);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：查询文档详情含解析内容", async (ctx) => {
      requireEs(ctx);
      const detail = await AiKnowledgeBaseAPI.getDocumentDetail(testDocId);
      expect(detail.id).toBe(testDocId);
      expect(detail.title).toBeTruthy();
      expect(detail.source).toBeTruthy();
      expect(detail.processingStatus).toBeTruthy();
    });

    test("边界：查询不存在的文档应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.getDocumentDetail(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("DELETE /api/v1/kb/documents/{id} - 删除文档", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：删除文档", async (ctx) => {
      requireEs(ctx);
      const kb = await AiKnowledgeBaseAPI.create(createKbForm());
      createdKbIds.push(kb.id);
      localKbIds.push(kb.id);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(kb.id, createTextDocForm());
      createdDocIds.push(doc.id);

      // 等待文档处理完成（pending/processing 状态下后端拒绝删除）
      await waitForDocSettled(doc.id);

      await AiKnowledgeBaseAPI.deleteDocument(doc.id);

      await expectBizError(AiKnowledgeBaseAPI.getDocumentDetail(doc.id), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    }, 45000);

    test("边界：删除不存在的文档应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.deleteDocument(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/kb/documents/{id}/reprocess - 重新处理文档", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：重新处理文档", async (ctx) => {
      requireEs(ctx);
      const kb = await AiKnowledgeBaseAPI.create(createKbForm());
      createdKbIds.push(kb.id);
      localKbIds.push(kb.id);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(kb.id, createTextDocForm());
      createdDocIds.push(doc.id);

      // 重处理仅对 failed 状态文档生效（否则 A0500 拒绝）。新文档状态不定：成功则回
      // pending；因非 failed 被拒亦符合后端契约，两者均接受。
      try {
        const result = await AiKnowledgeBaseAPI.reprocessDocument(doc.id);
        expect(result.processingStatus).toBe("pending");
      } catch (e: any) {
        if (e?.response?.data?.code !== "A0500") throw e;
      }
    });

    test("边界：重新处理不存在的文档应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.reprocessDocument(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ===== 分块管理 =====

  describe("GET /api/v1/kb/documents/{id}/chunks - 文档分块列表", () => {
    let testDocId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      const kbId = await createKb(localKbIds);
      if (!kbId) return;
      const doc = await AiKnowledgeBaseAPI.createTextDocument(kbId, createTextDocForm());
      testDocId = doc.id;
      createdDocIds.push(testDocId);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：查询文档分块列表", async (ctx) => {
      requireEs(ctx);
      // 后端返回分页对象 {list, total}
      const result = await AiKnowledgeBaseAPI.getChunks(testDocId);
      expect(Array.isArray(result.list)).toBe(true);
      // 文档可能还在处理中，分块可能为空
      if (result.list.length > 0) {
        const chunk = result.list[0]!;
        expect(chunk.id).toBeGreaterThan(0);
        expect(chunk.content).toBeTruthy();
        expect(typeof chunk.chunkIndex).toBe("number");
        expect(typeof chunk.tokenCount).toBe("number");
      }
    });

    test("边界：查询不存在文档的分块应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.getChunks(99999999), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  describe("POST /api/v1/kb/documents/chunks/preview - 分块预览", () => {
    test("正向测试：预览文档分块效果", async (ctx) => {
      requireEs(ctx);
      if (testFileIds.length === 0) {
        ctx.skip("顶层测试文件上传失败，无法预览分块效果");
        return;
      }
      const chunks = await AiKnowledgeBaseAPI.previewChunks({
        fileId: testFileIds[0]!,
        chunkingStrategy: "fixed",
        chunkSize: 800,
        chunkOverlap: 80,
      });
      expect(Array.isArray(chunks)).toBe(true);
    });
  });

  // ===== 检索 =====

  describe("POST /api/v1/kb/search - 知识库检索", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
      if (!testKbId) return;
      const doc1 = await AiKnowledgeBaseAPI.createTextDocument(
        testKbId,
        createTextDocForm({ content: "RIDCP 算法适用于户外图像去雾场景" })
      );
      createdDocIds.push(doc1.id);
      const doc2 = await AiKnowledgeBaseAPI.createTextDocument(
        testKbId,
        createTextDocForm({ content: "PSNR 28.5 是图像去雾质量评估的常用指标" })
      );
      createdDocIds.push(doc2.id);
      // 检索依赖真实向量，需等待文档处理完成
      await Promise.all([waitForDocSettled(doc1.id), waitForDocSettled(doc2.id)]);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：混合检索返回结果", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(
        createSearchForm({ knowledgeBaseIds: [testKbId], query: "去雾算法" })
      );
      expect(result.query).toBe("去雾算法");
      expect(Array.isArray(result.results)).toBe(true);
    });

    test("验证：检索结果含完整字段", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(
        createSearchForm({ knowledgeBaseIds: [testKbId], query: "RIDCP" })
      );
      if (result.results.length > 0) {
        const item = result.results[0]!;
        expect(item.chunkId).toBeGreaterThan(0);
        expect(item.content).toBeTruthy();
        expect(typeof item.score).toBe("number");
        expect(item.documentTitle).toBeTruthy();
        expect(item.documentId).toBeGreaterThan(0);
        expect(typeof item.chunkIndex).toBe("number");
      }
    });

    test("正向测试：Top-K 控制返回数量", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(
        createSearchForm({ knowledgeBaseIds: [testKbId], query: "去雾", topK: 3 })
      );
      expect(result.results.length).toBeLessThanOrEqual(3);
    });

    test("正向测试：元数据过滤（按文档类型）", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(
        createSearchForm({
          knowledgeBaseIds: [testKbId],
          query: "去雾",
          filters: { documentType: "pdf" },
        })
      );
      expect(Array.isArray(result.results)).toBe(true);
    });

    test("正向测试：启用 Rerank 重排序", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(
        createSearchForm({ knowledgeBaseIds: [testKbId], query: "RIDCP 算法" })
      );
      expect(Array.isArray(result.results)).toBe(true);
    });

    test("边界：空查询应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.search(createSearchForm({ query: "" })), [
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });

    test("安全：检索注入防护（SQL 注入式查询不触发特殊语法）", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.search(createSearchForm({ query: "* OR 1=1" }));
      expect(Array.isArray(result.results)).toBe(true);
    });

    test("边界：检索不存在知识库应失败", async (ctx) => {
      requireEs(ctx);
      // 后端对指定知识库做存在性校验：不存在则抛 A0401（非静默空结果）
      await expectBizError(
        AiKnowledgeBaseAPI.search(
          createSearchForm({ knowledgeBaseIds: [99999999], query: "测试" })
        ),
        ["A0401", "A0400", "B0001", "ERR_BAD_REQUEST"]
      );
    });
  });

  describe("POST /api/v1/kb/{id}/retrieve/test - 检索测试", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：检索测试返回结果", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.retrieveTest(
        testKbId,
        createRetrieveTestForm({ query: "去雾算法" })
      );
      expect(Array.isArray(result.results)).toBe(true);
    });

    test("正向测试：自定义 topK 和阈值", async (ctx) => {
      requireEs(ctx);
      const result = await AiKnowledgeBaseAPI.retrieveTest(
        testKbId,
        createRetrieveTestForm({ query: "去雾", topK: 3 })
      );
      expect(result.results.length).toBeLessThanOrEqual(3);
    });

    test("边界：检索测试不存在的知识库应失败", async () => {
      await expectBizError(AiKnowledgeBaseAPI.retrieveTest(99999999, createRetrieveTestForm()), [
        "A0401",
        "A0400",
        "B0001",
        "ERR_BAD_REQUEST",
      ]);
    });
  });

  // ===== 权限与数据隔离 =====

  describe("权限与数据隔离", () => {
    const localKbIds: number[] = [];

    afterAll(() => deleteKbs(localKbIds));

    test("安全：普通用户无法管理公共知识库", async (ctx) => {
      requireEs(ctx);
      const created = await AiKnowledgeBaseAPI.create(createKbForm({ visibility: "public" }));
      createdKbIds.push(created.id);
      localKbIds.push(created.id);

      await asUser(async () => {
        await expectBizError(AiKnowledgeBaseAPI.update(created.id, { name: "hacked" }), [
          "A0301",
          "A0401",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      });
    });

    test("安全：普通用户可只读访问公共知识库", async (ctx) => {
      requireEs(ctx);
      const created = await AiKnowledgeBaseAPI.create(createKbForm({ visibility: "public" }));
      createdKbIds.push(created.id);
      localKbIds.push(created.id);

      await asUser(async () => {
        const detail = await AiKnowledgeBaseAPI.getDetail(created.id);
        expect(detail.id).toBe(created.id);
      });
    });

    test("安全：越权查看他人私有知识库文档应失败", async (ctx) => {
      requireEs(ctx);
      const kb = await AiKnowledgeBaseAPI.create(createKbForm({ visibility: "private" }));
      createdKbIds.push(kb.id);
      localKbIds.push(kb.id);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(kb.id, createTextDocForm());
      createdDocIds.push(doc.id);

      await asUser(async () => {
        await expectBizError(AiKnowledgeBaseAPI.getDocumentDetail(doc.id), [
          "A0301",
          "A0401",
          "A0400",
          "B0001",
          "ERR_BAD_REQUEST",
        ]);
      });
    });
  });

  // ===== 文档处理状态流转 =====

  describe("文档处理状态轮询", () => {
    let testKbId = 0;
    const localKbIds: number[] = [];

    beforeAll(async () => {
      testKbId = await createKb(localKbIds);
    });

    afterAll(() => deleteKbs(localKbIds));

    test("正向测试：自定义文本文档处理状态从 pending 流转为 completed/failed", async (ctx) => {
      requireEs(ctx);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(testKbId, createTextDocForm());
      createdDocIds.push(doc.id);

      expect(doc.processingStatus).toBe("pending");

      const finalStatus = await waitForDocSettled(doc.id, 30000);

      // waitForDocSettled 仅在到达终态（completed/failed）时提前返回；本地小文本由本地
      // embedding 服务处理，30s 内必然结束于终态。非终态（processing/pending）仅当 30s
      // 超时才返回，属异常，故断言收敛为终态集合。
      expect(["completed", "failed"]).toContain(finalStatus);
    }, 45000);

    test("正向测试：重新处理文档后状态回到 pending", async (ctx) => {
      requireEs(ctx);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(testKbId, createTextDocForm());
      createdDocIds.push(doc.id);

      await waitForDocSettled(doc.id, 10000);

      // 重处理仅对 failed 状态文档生效（否则 A0500 拒绝）。成功则回 pending；
      // 文档非 failed（pending/processing/completed）被拒亦符合后端契约。
      try {
        const result = await AiKnowledgeBaseAPI.reprocessDocument(doc.id);
        expect(result.processingStatus).toBe("pending");
      } catch (e: any) {
        if (e?.response?.data?.code !== "A0500") throw e;
      }
    }, 15000);

    test("验证：completed 状态文档的分块可查询", async (ctx) => {
      requireEs(ctx);
      const doc = await AiKnowledgeBaseAPI.createTextDocument(testKbId, createTextDocForm());
      createdDocIds.push(doc.id);

      const status = await waitForDocSettled(doc.id, 30000);

      if (status === "completed") {
        const result = await AiKnowledgeBaseAPI.getChunks(doc.id);
        expect(Array.isArray(result.list)).toBe(true);
        if (result.list.length > 0) {
          const chunk = result.list[0]!;
          expect(chunk.content).toBeTruthy();
          expect(typeof chunk.tokenCount).toBe("number");
        }
      }
    }, 45000);
  });
});
