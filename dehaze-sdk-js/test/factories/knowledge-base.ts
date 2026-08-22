import { pageQuery, uniqueName } from "./common";
import type {
  DocumentBatchUploadForm,
  DocumentImportUrlForm,
  DocumentQuery,
  DocumentTextForm,
  DocumentUploadForm,
  KnowledgeBaseCreateForm,
  KnowledgeBaseQuery,
  KnowledgeBaseSearchForm,
  RetrieveTestForm,
} from "../../src/api/ai-knowledge-base/model";

/** 知识库创建表单工厂（扁平字段，对齐后端 KnowledgeBaseCreateForm） */
export const createKbForm = (
  overrides?: Partial<KnowledgeBaseCreateForm>
): KnowledgeBaseCreateForm => ({
  name: uniqueName("test_kb"),
  description: "测试知识库",
  visibility: "private",
  // 测试环境使用本地 embedding 服务（8992 /v1/embeddings，Qwen3-Embedding-0.6B，1024 维）
  // bge-m3 与本地模型同维度且为后端维度表已知模型，保证 create 时 ES 索引维度正确
  embeddingProvider: "local",
  embeddingModel: "bge-m3",
  chunkingStrategy: "fixed",
  chunkSize: 800,
  chunkOverlap: 80,
  searchStrategy: "hybrid",
  hybridWeight: 0.5,
  topK: 5,
  scoreThreshold: 0.3,
  enableRerank: false,
  ...overrides,
});

export const createKbQuery = (overrides?: Partial<KnowledgeBaseQuery>) =>
  pageQuery<KnowledgeBaseQuery>({ ...overrides });

export const createDocQuery = (overrides?: Partial<DocumentQuery>) =>
  pageQuery<DocumentQuery>({ ...overrides });

export const createDocUploadForm = (
  overrides?: Partial<DocumentUploadForm>
): DocumentUploadForm => ({
  fileId: 1,
  title: uniqueName("test_doc"),
  ...overrides,
});

/** 批量上传表单（后端使用 fileIds 列表） */
export const createBatchUploadForm = (fileIds?: number[]): DocumentBatchUploadForm => ({
  fileIds: fileIds || [1, 2],
});

export const createImportUrlForm = (
  overrides?: Partial<DocumentImportUrlForm>
): DocumentImportUrlForm => ({
  url: "https://www.baidu.com/",
  title: uniqueName("imported_doc"),
  ...overrides,
});

export const createTextDocForm = (overrides?: Partial<DocumentTextForm>): DocumentTextForm => ({
  title: uniqueName("text_doc"),
  content: "RIDCP 算法适用于户外图像去雾场景，在 PSNR 28.5 以上的测试集上表现优异。",
  ...overrides,
});

export const createSearchForm = (
  overrides?: Partial<KnowledgeBaseSearchForm>
): KnowledgeBaseSearchForm => ({
  query: "去雾算法",
  topK: 5,
  ...overrides,
});

export const createRetrieveTestForm = (
  overrides?: Partial<RetrieveTestForm>
): RetrieveTestForm => ({
  query: "去雾算法",
  topK: 5,
  ...overrides,
});
