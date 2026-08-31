import type {
  ChunkVO,
  DocumentProcessingStatus,
  DocumentSource,
  DocumentVO,
  KnowledgeBaseVO,
  LowQualityChunkVO,
  TestSetVO,
} from "dehaze-sdk-js";
import { defineMock } from "./base";

/** 与 auth.mock.ts 中登录的 admin 保持一致，用于私有库可见性过滤 */
const CURRENT_USER_ID = 1;

let nextKbId = 7;
let nextDocumentId = 18;
let nextChunkId = 1007;
let nextTestSetId = 4;

const knowledgeBases: KnowledgeBaseVO[] = [
  {
    id: 1,
    name: "图像增强算法文档库",
    description:
      "暗通道先验、Retinex、直方图均衡等图像增强算法的原理说明与工程实现文档",
    visibility: "public",
    status: 1,
    embeddingProvider: "openai",
    embeddingModel: "text-embedding-3-small",
    chunkingStrategy: "recursive",
    chunkSize: 512,
    chunkOverlap: 64,
    searchStrategy: "hybrid",
    hybridWeight: 0.7,
    topK: 5,
    scoreThreshold: 0.35,
    enableRerank: 1,
    rerankModel: "bge-reranker-v2-m3",
    documentCount: 5,
    chunkCount: 58,
    totalTokens: 28000,
    createBy: 1,
    createTime: "2026-07-02 09:30:00",
    updateTime: "2026-08-20 14:12:07",
  },
  {
    id: 2,
    name: "去雾算法最佳实践",
    description: "去雾算法在交通监控、航拍、车载等场景的落地方案与调参经验",
    visibility: "public",
    status: 1,
    embeddingProvider: "openai",
    embeddingModel: "text-embedding-3-small",
    chunkingStrategy: "semantic",
    chunkSize: 768,
    chunkOverlap: 96,
    searchStrategy: "hybrid",
    hybridWeight: 0.6,
    topK: 8,
    scoreThreshold: 0.3,
    enableRerank: 1,
    rerankModel: "bge-reranker-v2-m3",
    documentCount: 4,
    chunkCount: 63,
    totalTokens: 31560,
    createBy: 1,
    createTime: "2026-07-11 15:04:22",
    updateTime: "2026-08-24 09:41:35",
  },
  {
    id: 3,
    name: "低照度图像增强技术手册",
    description: "夜间与低照度场景的增强方法综述、数据集构建与效果评测规范",
    visibility: "public",
    status: 1,
    embeddingProvider: "openai",
    embeddingModel: "text-embedding-3-small",
    chunkingStrategy: "fixed",
    chunkSize: 512,
    chunkOverlap: 50,
    searchStrategy: "vector",
    hybridWeight: 0.5,
    topK: 5,
    scoreThreshold: 0.4,
    enableRerank: 0,
    documentCount: 3,
    chunkCount: 41,
    totalTokens: 20340,
    createBy: 1,
    createTime: "2026-07-19 11:20:48",
    updateTime: "2026-08-12 17:05:11",
  },
  {
    id: 4,
    name: "我的去雾实验记录",
    description: "个人整理的雾浓度分级测试与窗口半径对比实验数据",
    visibility: "private",
    status: 1,
    embeddingProvider: "openai",
    embeddingModel: "text-embedding-3-small",
    chunkingStrategy: "recursive",
    chunkSize: 512,
    chunkOverlap: 64,
    searchStrategy: "hybrid",
    hybridWeight: 0.65,
    topK: 6,
    scoreThreshold: 0.32,
    enableRerank: 0,
    documentCount: 2,
    chunkCount: 12,
    totalTokens: 5980,
    createBy: 1,
    createTime: "2026-08-01 20:15:03",
    updateTime: "2026-08-26 10:33:29",
  },
  {
    id: 5,
    name: "客户项目交付资料（A 市交通监控）",
    description: "A 市交通监控去雾增强项目的交付说明书、验收指标与测试报告",
    visibility: "private",
    status: 1,
    embeddingProvider: "openai",
    embeddingModel: "bge-m3",
    chunkingStrategy: "fixed",
    chunkSize: 1024,
    chunkOverlap: 128,
    searchStrategy: "hybrid",
    hybridWeight: 0.75,
    topK: 5,
    scoreThreshold: 0.35,
    enableRerank: 1,
    rerankModel: "bge-reranker-v2-m3",
    documentCount: 2,
    chunkCount: 27,
    totalTokens: 13550,
    createBy: 1,
    createTime: "2026-08-14 08:47:56",
    updateTime: "2026-08-27 16:22:04",
  },
  {
    id: 6,
    name: "算法组内部调参笔记",
    description: "算法组共享的去雾/增强模型调参结论（仅组内可见）",
    visibility: "private",
    status: 2,
    embeddingProvider: "openai",
    embeddingModel: "bge-m3",
    chunkingStrategy: "recursive",
    chunkSize: 512,
    chunkOverlap: 64,
    searchStrategy: "vector",
    hybridWeight: 0.5,
    topK: 5,
    scoreThreshold: 0.35,
    enableRerank: 0,
    documentCount: 1,
    chunkCount: 9,
    totalTokens: 4310,
    createBy: 2,
    createTime: "2026-08-21 13:09:41",
    updateTime: "2026-08-21 13:09:41",
  },
];

const documents: DocumentVO[] = [
  {
    id: 1,
    knowledgeBaseId: 1,
    fileId: 1001,
    title: "暗通道先验去雾算法原理与实现.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 32,
    totalTokens: 15600,
    version: 1,
    createTime: "2026-07-02 09:41:12",
    updateTime: "2026-07-02 09:52:40",
  },
  {
    id: 2,
    knowledgeBaseId: 1,
    title: "基于 Retinex 的低照度增强方法.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 26,
    totalTokens: 12400,
    version: 2,
    createTime: "2026-07-05 14:22:09",
    updateTime: "2026-08-19 10:07:52",
  },
  {
    id: 3,
    knowledgeBaseId: 1,
    fileId: 1003,
    title: "去雾效果评价指标（PSNR/SSIM/NIQE）.docx",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "processing",
    chunkCount: 0,
    totalTokens: 0,
    version: 1,
    createTime: "2026-08-20 14:10:33",
  },
  {
    id: 4,
    knowledgeBaseId: 1,
    fileId: 1004,
    title: "2026 图像增强算法选型报告.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "pending",
    chunkCount: 0,
    totalTokens: 0,
    version: 1,
    createTime: "2026-08-20 14:12:07",
  },
  {
    id: 5,
    knowledgeBaseId: 1,
    fileId: 1005,
    title: "扫描版去雾论文（1998）.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "failed",
    chunkCount: 0,
    totalTokens: 0,
    version: 1,
    error: "PDF 解析失败：文件为纯扫描件，未提取到文本层，请先 OCR 后重新上传",
    createTime: "2026-08-18 09:15:27",
  },
  {
    id: 6,
    knowledgeBaseId: 2,
    fileId: 1006,
    title: "雾天图像增强工程实践指南.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 21,
    totalTokens: 10240,
    version: 1,
    createTime: "2026-07-11 15:20:14",
    updateTime: "2026-07-11 15:31:02",
  },
  {
    id: 7,
    knowledgeBaseId: 2,
    title: "去雾算法在交通监控场景的调参经验.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 15,
    totalTokens: 7310,
    version: 3,
    createTime: "2026-07-25 10:05:44",
    updateTime: "2026-08-24 09:41:35",
  },
  {
    id: 8,
    knowledgeBaseId: 2,
    fileId: 1008,
    title: "实时去雾算法性能优化 checklist.xlsx",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 9,
    totalTokens: 3480,
    version: 1,
    createTime: "2026-08-02 16:48:19",
    updateTime: "2026-08-02 16:53:07",
  },
  {
    id: 9,
    knowledgeBaseId: 2,
    title: "CLAHE 参数调优踩坑记录.txt",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 18,
    totalTokens: 6530,
    version: 1,
    createTime: "2026-08-09 11:32:55",
    updateTime: "2026-08-09 11:38:20",
  },
  {
    id: 10,
    knowledgeBaseId: 3,
    fileId: 1010,
    title: "低照度图像增强综述.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 20,
    totalTokens: 9800,
    version: 1,
    createTime: "2026-07-19 11:26:31",
    updateTime: "2026-07-19 11:35:48",
  },
  {
    id: 11,
    knowledgeBaseId: 3,
    title: "基于深度学习的夜间图像增强（Zero-DCE）.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 14,
    totalTokens: 6900,
    version: 2,
    createTime: "2026-07-28 09:14:03",
    updateTime: "2026-08-12 17:05:11",
  },
  {
    id: 12,
    knowledgeBaseId: 3,
    fileId: 1012,
    title: "低照度数据集合成方法.docx",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 7,
    totalTokens: 3640,
    version: 1,
    createTime: "2026-08-05 15:02:47",
    updateTime: "2026-08-05 15:09:13",
  },
  {
    id: 13,
    knowledgeBaseId: 4,
    title: "去雾实验记录：雾浓度分级测试.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 8,
    totalTokens: 3260,
    version: 1,
    createTime: "2026-08-01 20:22:41",
    updateTime: "2026-08-01 20:28:09",
  },
  {
    id: 14,
    knowledgeBaseId: 4,
    title: "暗通道窗口半径对去雾效果的影响.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 4,
    totalTokens: 2720,
    version: 1,
    createTime: "2026-08-26 10:31:12",
    updateTime: "2026-08-26 10:33:29",
  },
  {
    id: 15,
    knowledgeBaseId: 5,
    fileId: 1015,
    title: "A 市交通监控项目交付说明书.pdf",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 16,
    totalTokens: 8240,
    version: 1,
    createTime: "2026-08-14 08:52:30",
    updateTime: "2026-08-14 09:01:44",
  },
  {
    id: 16,
    knowledgeBaseId: 5,
    fileId: 1016,
    title: "项目验收指标与测试报告.docx",
    source: "upload",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 11,
    totalTokens: 5310,
    version: 1,
    createTime: "2026-08-27 16:18:02",
    updateTime: "2026-08-27 16:22:04",
  },
  {
    id: 17,
    knowledgeBaseId: 6,
    title: "算法组内部调参笔记.md",
    source: "manual",
    parsingStrategy: "auto",
    processingStatus: "completed",
    chunkCount: 9,
    totalTokens: 4310,
    version: 1,
    createTime: "2026-08-21 13:12:06",
    updateTime: "2026-08-21 13:18:52",
  },
];

/** 去雾/增强领域语料片段，用于生成分块、检索与分块预览 */
const CHUNK_CORPUS = [
  "暗通道先验（Dark Channel Prior）基于这样一个统计规律：在绝大多数无雾图像的局部区域中，至少有一个颜色通道存在亮度极低的像素，其暗通道值趋近于零。含雾图像因大气光散射导致暗通道整体亮度被抬高，据此即可估计雾的浓度。",
  "大气散射模型 I(x) = J(x)t(x) + A(1 - t(x)) 中，I 为观测到的有雾图像，J 为待恢复的场景辐射，A 为全局大气光，t(x) 为透射率。去雾的目标是在已知 I 的前提下求解 J，关键在于准确估计 A 与 t(x)。",
  "透射率估计公式为 t(x) = 1 - ω·min_c(min_{y∈Ω(x)}(I^c(y)/A^c))，其中 ω 为保留少量雾的常数（工程上常取 0.95），避免去雾过度造成图像失真，同时保留一定的景深信息。",
  "导向滤波（Guided Filter）用于细化粗透射率图，相比软抠图（Soft Matting）将复杂度由 O(N²) 降至 O(N)，在保持边缘的同时显著提升速度，是工程落地的首选方案。",
  "去雾效果评价分两类指标：PSNR 与 SSIM 属于有参考指标，需要无雾真值图像；NIQE、BRISQUE 属于无参考指标，可直接对真实雾图评分。工程上通常组合使用，避免单一指标给出误导性结论。",
  "Retinex 理论将图像分解为照度分量与反射分量 S(x,y) = L(x,y)·R(x,y)，通过高斯估计照度分量并压缩其动态范围，在提升亮度的同时保持色彩恒常性，是低照度增强的经典思路。",
];

const chunkMap: Record<number, ChunkVO[]> = {
  1: CHUNK_CORPUS.map((content, index) => ({
    id: 1001 + index,
    documentId: 1,
    chunkIndex: index,
    content,
    tokenCount: 180 + index * 26,
    metadata: { page: Math.floor(index / 2) + 1, type: "paragraph" },
  })),
};

const testSets: TestSetVO[] = [
  {
    id: 1,
    knowledgeBaseId: 1,
    question: "暗通道先验去雾的核心假设是什么？",
    expectedChunkIds: [1001],
    createTime: "2026-08-06 10:20:00",
  },
  {
    id: 2,
    knowledgeBaseId: 1,
    question: "如何估计透射率并对其进行细化？",
    expectedChunkIds: [1003, 1004],
    createTime: "2026-08-06 10:24:31",
  },
  {
    id: 3,
    knowledgeBaseId: 2,
    question: "工程上如何评价去雾算法的效果？",
    expectedChunkIds: [1001, 1005],
    createTime: "2026-08-13 15:47:08",
  },
];

const lowQualityChunks: LowQualityChunkVO[] = [
  {
    chunkId: 1004,
    content: CHUNK_CORPUS[3],
    documentId: 1,
    documentTitle: "暗通道先验去雾算法原理与实现.pdf",
    chunkIndex: 3,
    thumbsDownCount: 4,
    metadata: {
      feedbackType: "thumbs_down",
      reason: "答案与提问无关",
      lastFeedbackAt: "2026-08-22 16:02:11",
    },
  },
  {
    chunkId: 1006,
    content: CHUNK_CORPUS[5],
    documentId: 1,
    documentTitle: "暗通道先验去雾算法原理与实现.pdf",
    chunkIndex: 5,
    thumbsDownCount: 2,
    metadata: {
      feedbackType: "thumbs_down",
      reason: "内容过时，结论与新版实验不符",
      lastFeedbackAt: "2026-08-25 09:18:44",
    },
  },
  {
    chunkId: 1002,
    content: CHUNK_CORPUS[1],
    documentId: 1,
    documentTitle: "暗通道先验去雾算法原理与实现.pdf",
    chunkIndex: 1,
    thumbsDownCount: 1,
    metadata: {
      feedbackType: "thumbs_down",
      reason: "公式排版错乱",
      lastFeedbackAt: "2026-08-27 14:36:52",
    },
  },
];

function formatNow() {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
    d.getMinutes()
  )}:${pad(d.getSeconds())}`;
}

function paginate<T>(list: T[], query: Record<string, any>) {
  const pageNum = Number(query.pageNum) || 1;
  const pageSize = Number(query.pageSize) || 10;
  const start = (pageNum - 1) * pageSize;
  return { list: list.slice(start, start + pageSize), total: list.length };
}

function findKb(id: number) {
  return knowledgeBases.find((kb) => kb.id === id);
}

function findDocument(id: number) {
  return documents.find((doc) => doc.id === id);
}

function kbDocuments(knowledgeBaseId: number) {
  return documents.filter((doc) => doc.knowledgeBaseId === knowledgeBaseId);
}

function chunksOf(documentId: number): ChunkVO[] {
  if (!chunkMap[documentId]) {
    const doc = findDocument(documentId);
    const count = doc && doc.chunkCount > 0 ? Math.min(doc.chunkCount, 6) : 0;
    chunkMap[documentId] = Array.from({ length: count }, (_, index) => ({
      id: nextChunkId++,
      documentId,
      chunkIndex: index,
      content: CHUNK_CORPUS[(index + documentId) % CHUNK_CORPUS.length],
      tokenCount: 180 + ((index * 37 + documentId * 11) % 120),
      metadata: { page: Math.floor(index / 2) + 1, type: "paragraph" },
    }));
  }
  return chunkMap[documentId];
}

function completedChunksOf(knowledgeBaseId: number) {
  return kbDocuments(knowledgeBaseId)
    .filter((doc) => doc.processingStatus === "completed")
    .flatMap((doc) => chunksOf(doc.id).map((chunk) => ({ chunk, doc })));
}

function lowQualityOf(knowledgeBaseId: number): LowQualityChunkVO[] {
  const seeded = lowQualityChunks.filter((item) =>
    kbDocuments(knowledgeBaseId).some((doc) => doc.id === item.documentId)
  );
  if (seeded.length > 0) {
    return seeded;
  }
  return kbDocuments(knowledgeBaseId)
    .filter((doc) => doc.processingStatus === "completed")
    .flatMap((doc) =>
      chunksOf(doc.id)
        .filter((chunk) => chunk.chunkIndex % 3 === 2)
        .map((chunk) => ({
          chunkId: chunk.id,
          content: chunk.content,
          documentId: doc.id,
          documentTitle: doc.title,
          chunkIndex: chunk.chunkIndex,
          thumbsDownCount: 1 + (chunk.chunkIndex % 3),
          metadata: {
            feedbackType: "thumbs_down",
            reason: "检索结果与问题不匹配",
          },
        }))
    );
}

function createDocument(
  knowledgeBaseId: number,
  doc: Pick<DocumentVO, "title" | "source"> & Partial<DocumentVO>
) {
  const kb = findKb(knowledgeBaseId);
  const document: DocumentVO = {
    id: nextDocumentId++,
    knowledgeBaseId,
    title: doc.title,
    source: doc.source,
    parsingStrategy: "auto",
    processingStatus: "processing",
    content: doc.content,
    chunkCount: 0,
    totalTokens: 0,
    version: 1,
    createTime: formatNow(),
    ...(doc.fileId === undefined ? {} : { fileId: doc.fileId }),
  };
  documents.push(document);
  kb!.documentCount += 1;
  // 真实链路为异步流水线，mock 中延时置为完成，便于直接查看分块与检索效果
  setTimeout(() => {
    if (findDocument(document.id)?.processingStatus !== "processing") {
      return;
    }
    const chunkCount = 3 + (document.id % 4);
    const totalTokens = chunkCount * 480;
    document.processingStatus = "completed";
    document.chunkCount = chunkCount;
    document.totalTokens = totalTokens;
    document.updateTime = formatNow();
    kb!.chunkCount += chunkCount;
    kb!.totalTokens += totalTokens;
  }, 1500);
  return document;
}

function removeDocument(document: DocumentVO) {
  const kb = findKb(document.knowledgeBaseId);
  documents.splice(documents.indexOf(document), 1);
  delete chunkMap[document.id];
  if (kb) {
    kb.documentCount -= 1;
    kb.chunkCount -= document.chunkCount;
    kb.totalTokens -= document.totalTokens;
  }
}

export default defineMock([
  // 知识库列表
  {
    url: "kb",
    method: ["GET"],
    body({ query }) {
      const keyword = (query.keyword as string)?.trim();
      const visible =
        query.view === "admin"
          ? knowledgeBases
          : knowledgeBases.filter(
              (kb) =>
                kb.visibility === "public" || kb.createBy === CURRENT_USER_ID
            );
      const matched = keyword
        ? visible.filter(
            (kb) =>
              kb.name.includes(keyword) || kb.description?.includes(keyword)
          )
        : visible;
      return {
        code: "00000",
        data: paginate(matched, query),
        msg: "一切ok",
      };
    },
  },

  // 新增知识库
  {
    url: "kb",
    method: ["POST"],
    body({ body }) {
      const kb: KnowledgeBaseVO = {
        id: nextKbId++,
        name: body.name,
        description: body.description,
        visibility: body.visibility ?? "private",
        status: 1,
        embeddingProvider: body.embeddingProvider ?? "openai",
        embeddingModel: body.embeddingModel,
        chunkingStrategy: body.chunkingStrategy ?? "recursive",
        chunkSize: body.chunkSize ?? 512,
        chunkOverlap: body.chunkOverlap ?? 64,
        searchStrategy: body.searchStrategy ?? "hybrid",
        hybridWeight: body.hybridWeight ?? 0.7,
        topK: body.topK ?? 5,
        scoreThreshold: body.scoreThreshold ?? 0.35,
        enableRerank: body.enableRerank ? 1 : 0,
        rerankModel: body.rerankModel,
        documentCount: 0,
        chunkCount: 0,
        totalTokens: 0,
        createBy: CURRENT_USER_ID,
        createTime: formatNow(),
        updateTime: formatNow(),
      };
      knowledgeBases.push(kb);
      return {
        code: "00000",
        data: kb,
        msg: "新增知识库" + kb.name + "成功",
      };
    },
  },

  // 知识库详情
  {
    url: "kb/:id",
    method: ["GET"],
    body({ params }) {
      const kb = findKb(Number(params.id));
      if (!kb) {
        return { code: "A0401", data: null, msg: "知识库不存在" };
      }
      return {
        code: "00000",
        data: kb,
        msg: "一切ok",
      };
    },
  },

  // 编辑知识库（分块策略与 embedding 模型创建后不可修改）
  {
    url: "kb/:id",
    method: ["PUT"],
    body({ body, params }) {
      const kb = findKb(Number(params.id));
      if (!kb) {
        return { code: "A0401", data: null, msg: "知识库不存在" };
      }
      Object.assign(kb, {
        ...(body.name === undefined ? {} : { name: body.name }),
        ...(body.description === undefined
          ? {}
          : { description: body.description }),
        ...(body.searchStrategy === undefined
          ? {}
          : { searchStrategy: body.searchStrategy }),
        ...(body.hybridWeight === undefined
          ? {}
          : { hybridWeight: Number(body.hybridWeight) }),
        ...(body.topK === undefined ? {} : { topK: Number(body.topK) }),
        ...(body.scoreThreshold === undefined
          ? {}
          : { scoreThreshold: Number(body.scoreThreshold) }),
        ...(body.enableRerank === undefined
          ? {}
          : { enableRerank: body.enableRerank ? 1 : 0 }),
        ...(body.rerankModel === undefined
          ? {}
          : { rerankModel: body.rerankModel }),
        updateTime: formatNow(),
      });
      return {
        code: "00000",
        data: kb,
        msg: "修改知识库" + kb.name + "成功",
      };
    },
  },

  // 删除知识库
  {
    url: "kb/:id",
    method: ["DELETE"],
    body({ params }) {
      const kb = findKb(Number(params.id));
      if (!kb) {
        return { code: "A0401", data: null, msg: "知识库不存在" };
      }
      kbDocuments(kb.id).forEach((doc) => removeDocument(doc));
      knowledgeBases.splice(knowledgeBases.indexOf(kb), 1);
      return {
        code: "00000",
        data: null,
        msg: "删除知识库" + kb.name + "成功",
      };
    },
  },

  // 文档列表
  {
    url: "kb/:id/documents",
    method: ["GET"],
    body({ params, query }) {
      let list = kbDocuments(Number(params.id));
      if (query.processingStatus) {
        list = list.filter(
          (doc) => doc.processingStatus === query.processingStatus
        );
      }
      if (query.keyword) {
        list = list.filter((doc) =>
          doc.title.includes(query.keyword as string)
        );
      }
      return {
        code: "00000",
        data: paginate(list, query),
        msg: "一切ok",
      };
    },
  },

  // 上传文档
  {
    url: "kb/:id/documents",
    method: ["POST"],
    body({ body, params }) {
      const document = createDocument(Number(params.id), {
        title: body.title ?? "未命名文档.pdf",
        source: "upload",
        fileId: body.fileId,
      });
      return {
        code: "00000",
        data: document,
        msg: "上传文档" + document.title + "成功",
      };
    },
  },

  // 批量上传文档
  {
    url: "kb/:id/documents/batch",
    method: ["POST"],
    body({ body, params }) {
      const fileIds: number[] = body.fileIds ?? [];
      const results = fileIds.map((fileId) => {
        if (!Number(fileId)) {
          return {
            fileId,
            success: false,
            code: "B0401",
            message: "文件不存在",
          };
        }
        const document = createDocument(Number(params.id), {
          title: `批量上传文档-${fileId}.pdf`,
          source: "upload",
          fileId,
        });
        return {
          fileId,
          success: true,
          id: document.id,
          processingStatus: document.processingStatus,
        };
      });
      return {
        code: "00000",
        data: results,
        msg: "一切ok",
      };
    },
  },

  // 导入网页为文档
  {
    url: "kb/:id/documents/import-url",
    method: ["POST"],
    body({ body, params }) {
      const document = createDocument(Number(params.id), {
        title: body.title ?? new URL(body.url).hostname,
        source: "url",
      });
      return {
        code: "00000",
        data: document,
        msg: "导入网页" + document.title + "成功",
      };
    },
  },

  // 自定义文本创建文档
  {
    url: "kb/:id/documents/text",
    method: ["POST"],
    body({ body, params }) {
      const document = createDocument(Number(params.id), {
        title: body.title,
        source: "manual",
        content: body.content,
      });
      return {
        code: "00000",
        data: document,
        msg: "新增文档" + document.title + "成功",
      };
    },
  },

  // 文档详情
  {
    url: "kb/documents/:id",
    method: ["GET"],
    body({ params }) {
      const document = findDocument(Number(params.id));
      if (!document) {
        return { code: "A0401", data: null, msg: "文档不存在" };
      }
      return {
        code: "00000",
        data: {
          ...document,
          content:
            document.content ??
            (document.processingStatus === "completed"
              ? chunksOf(document.id)
                  .map((chunk) => chunk.content)
                  .join("\n\n")
              : undefined),
        },
        msg: "一切ok",
      };
    },
  },

  // 删除文档
  {
    url: "kb/documents/:id",
    method: ["DELETE"],
    body({ params }) {
      const document = findDocument(Number(params.id));
      if (!document) {
        return { code: "A0401", data: null, msg: "文档不存在" };
      }
      removeDocument(document);
      return {
        code: "00000",
        data: null,
        msg: "删除文档" + document.title + "成功",
      };
    },
  },

  // 重新处理文档
  {
    url: "kb/documents/:id/reprocess",
    method: ["POST"],
    body({ params }) {
      const document = findDocument(Number(params.id));
      if (!document) {
        return { code: "A0401", data: null, msg: "文档不存在" };
      }
      const kb = findKb(document.knowledgeBaseId);
      if (kb) {
        kb.chunkCount -= document.chunkCount;
        kb.totalTokens -= document.totalTokens;
      }
      delete chunkMap[document.id];
      Object.assign(document, {
        processingStatus: "pending" as DocumentProcessingStatus,
        chunkCount: 0,
        totalTokens: 0,
        version: document.version + 1,
        error: undefined,
        updateTime: formatNow(),
      });
      return {
        code: "00000",
        data: document,
        msg: "文档" + document.title + "已重新提交处理",
      };
    },
  },

  // 分块预览
  {
    url: "kb/documents/chunks/preview",
    method: ["POST"],
    body({ body }) {
      const chunkSize = Number(body.chunkSize) || 512;
      const chunkOverlap = Number(body.chunkOverlap) || 64;
      const step = Math.max(chunkSize - chunkOverlap, 1);
      const text = CHUNK_CORPUS.join("");
      const chunks: { index: number; content: string; tokenCount: number }[] =
        [];
      for (
        let start = 0, index = 0;
        start < text.length;
        start += step, index++
      ) {
        const content = text.slice(start, start + chunkSize);
        chunks.push({
          index,
          content,
          tokenCount: Math.ceil(content.length / 1.6),
        });
      }
      return {
        code: "00000",
        data: chunks,
        msg: "一切ok",
      };
    },
  },

  // 文档分块列表
  {
    url: "kb/documents/:id/chunks",
    method: ["GET"],
    body({ params, query }) {
      return {
        code: "00000",
        data: paginate(chunksOf(Number(params.id)), query),
        msg: "一切ok",
      };
    },
  },

  // 检索测试
  {
    url: "kb/:id/retrieve/test",
    method: ["POST"],
    body({ body, params }) {
      const knowledgeBaseId = Number(params.id);
      const topK = Number(body.topK) || 5;
      const results = completedChunksOf(knowledgeBaseId)
        .map(({ chunk, doc }, index) => ({
          chunkId: chunk.id,
          content: chunk.content,
          metadata: chunk.metadata,
          score: Number(
            (
              0.93 -
              ((index * 7 + String(body.query).length) % 40) / 100
            ).toFixed(2)
          ),
          documentTitle: doc.title,
          documentId: doc.id,
          chunkIndex: chunk.chunkIndex,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
      return {
        code: "00000",
        data: {
          query: body.query,
          knowledgeBaseIds: [knowledgeBaseId],
          results,
        },
        msg: "一切ok",
      };
    },
  },

  // 索引状态
  {
    url: "kb/:id/index-stats",
    method: ["GET"],
    body({ params }) {
      const kb = findKb(Number(params.id));
      if (!kb) {
        return { code: "A0401", data: null, msg: "知识库不存在" };
      }
      const indexSize = kb.chunkCount * 4096;
      return {
        code: "00000",
        data: {
          indexSize,
          indexDocCount: kb.chunkCount,
          thresholdWarning: indexSize > 1024 * 1024 * 1024,
        },
        msg: "一切ok",
      };
    },
  },

  // 召回测试集列表
  {
    url: "kb/:id/retrieve/test-sets",
    method: ["GET"],
    body({ params, query }) {
      const list = testSets.filter(
        (item) => item.knowledgeBaseId === Number(params.id)
      );
      return {
        code: "00000",
        data: paginate(list, query),
        msg: "一切ok",
      };
    },
  },

  // 新增召回测试集
  {
    url: "kb/:id/retrieve/test-sets",
    method: ["POST"],
    body({ body, params }) {
      const testSet: TestSetVO = {
        id: nextTestSetId++,
        knowledgeBaseId: Number(params.id),
        question: body.question,
        expectedChunkIds: body.expectedChunkIds ?? [],
        createTime: formatNow(),
      };
      testSets.push(testSet);
      return {
        code: "00000",
        data: testSet,
        msg: "新增测试集成功",
      };
    },
  },

  // 执行召回测试集
  {
    url: "kb/:id/retrieve/test-sets/:testSetId/run",
    method: ["POST"],
    body({ params }) {
      const knowledgeBaseId = Number(params.id);
      const testSet = testSets.find(
        (item) =>
          item.id === Number(params.testSetId) &&
          item.knowledgeBaseId === knowledgeBaseId
      );
      if (!testSet) {
        return { code: "A0401", data: null, msg: "测试集不存在" };
      }
      const retrievableChunkIds = new Set(
        completedChunksOf(knowledgeBaseId).map((item) => item.chunk.id)
      );
      const totalCases = testSet.expectedChunkIds.length;
      const hitCases = testSet.expectedChunkIds.filter((id) =>
        retrievableChunkIds.has(id)
      ).length;
      return {
        code: "00000",
        data: {
          testSetId: testSet.id,
          recallAtK:
            totalCases === 0 ? 0 : Number((hitCases / totalCases).toFixed(2)),
          hitRate: hitCases > 0 ? 1 : 0,
          totalCases,
          hitCases,
        },
        msg: "一切ok",
      };
    },
  },

  // 低质量片段列表
  {
    url: "kb/:id/chunks/low-quality",
    method: ["GET"],
    body({ params, query }) {
      let list = lowQualityOf(Number(params.id));
      if (query.feedbackType) {
        list = list.filter(
          (item) => item.metadata?.feedbackType === query.feedbackType
        );
      }
      if (query.keyword) {
        list = list.filter((item) =>
          item.content.includes(query.keyword as string)
        );
      }
      return {
        code: "00000",
        data: paginate(list, query),
        msg: "一切ok",
      };
    },
  },
]);
