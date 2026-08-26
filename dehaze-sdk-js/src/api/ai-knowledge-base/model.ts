import { PageQuery } from "@/types";

// ==================== 枚举类型 ====================

/** 知识库可见性 */
export type KnowledgeBaseVisibility = "public" | "private";

/** 知识库状态：1-启用，2-处理中，0-禁用 */
export type KnowledgeBaseStatus = 0 | 1 | 2;

/** 文档处理状态 */
export type DocumentProcessingStatus = "pending" | "processing" | "completed" | "failed";

/** 文档来源 */
export type DocumentSource = "upload" | "url" | "manual" | "algorithm_sync" | "experience";

/** 分块策略 */
export type ChunkingStrategy = "fixed" | "semantic" | "recursive" | "qa" | "table";

/** 检索策略 */
export type SearchStrategy = "vector" | "keyword" | "hybrid";

// ==================== 知识库 ====================

/** 知识库创建表单（扁平字段，与后端 KnowledgeBaseCreateForm 对齐） */
export interface KnowledgeBaseCreateForm {
  /** 知识库名称 */
  name: string;
  /** 知识库描述 */
  description?: string;
  /** 公开/私有 */
  visibility: KnowledgeBaseVisibility;
  /** Embedding 提供商（默认 openai） */
  embeddingProvider?: string;
  /** Embedding 模型标识（text-embedding-3-small / bge-m3 等） */
  embeddingModel: string;
  /** 分块策略 */
  chunkingStrategy: ChunkingStrategy;
  /** 分块大小（token） */
  chunkSize?: number;
  /** 分块重叠（token） */
  chunkOverlap?: number;
  /** 检索策略 */
  searchStrategy?: SearchStrategy;
  /** 混合检索向量权重（0-1，仅 hybrid 生效） */
  hybridWeight?: number;
  /** 检索 Top-K */
  topK?: number;
  /** 相似度阈值 */
  scoreThreshold?: number;
  /** 是否启用 Rerank 重排序 */
  enableRerank?: boolean;
  /** 重排序模型标识 */
  rerankModel?: string;
}

/** 知识库更新表单（仅可编辑项：名称/描述/检索策略，分块策略和 embedding 模型创建后不可修改） */
export interface KnowledgeBaseUpdateForm {
  name?: string;
  description?: string;
  searchStrategy?: SearchStrategy;
  hybridWeight?: number;
  topK?: number;
  scoreThreshold?: number;
  enableRerank?: boolean;
  rerankModel?: string;
}

/** 知识库视图对象（后端 KnowledgeBaseVO 扁平 camelCase 输出） */
export interface KnowledgeBaseVO {
  id: number;
  name: string;
  description?: string;
  visibility: KnowledgeBaseVisibility;
  /** 知识库状态 */
  status: KnowledgeBaseStatus;
  embeddingProvider: string;
  embeddingModel: string;
  chunkingStrategy: ChunkingStrategy;
  chunkSize: number;
  chunkOverlap: number;
  searchStrategy: SearchStrategy;
  hybridWeight: number;
  topK: number;
  scoreThreshold: number;
  /** 是否启用 Rerank（0/1） */
  enableRerank: number;
  rerankModel?: string;
  /** 文档数（冗余统计） */
  documentCount: number;
  /** 分块总数（冗余统计） */
  chunkCount: number;
  /** 编码 Token 总数（冗余统计） */
  totalTokens: number;
  createBy?: number;
  createTime?: string;
  updateTime?: string;
}

/** 知识库列表查询参数 */
export interface KnowledgeBaseQuery extends PageQuery {
  keyword?: string;
  /** 管理端视角：返回全部知识库含私有库（仅 kb:manage 可用，普通用户 403） */
  view?: "admin";
}

// ==================== 管理端 ====================

/** 知识库索引状态（管理端索引状态区） */
export interface IndexStatsVO {
  /** ES 索引大小（字节） */
  indexSize: number;
  /** 索引文档数 */
  indexDocCount: number;
  /** 是否触发索引大小阈值告警（默认 1GB） */
  thresholdWarning: boolean;
}

/** 召回测试集（问题 + 期望命中段落） */
export interface TestSetVO {
  id: number;
  knowledgeBaseId: number;
  /** 测试问题 */
  question: string;
  /** 期望命中分块 ID 列表（must_include） */
  expectedChunkIds: number[];
  createTime?: string;
}

/** 召回测试集创建表单 */
export interface TestSetCreateForm {
  /** 测试问题 */
  question: string;
  /** 期望命中分块 ID 列表（must_include） */
  expectedChunkIds: number[];
}

/** 召回测试集执行结果（Recall@K 与命中率） */
export interface RecallTestResultVO {
  testSetId: number;
  /** Recall@K：期望命中分块出现在 Top-K 结果中的比例（0-1） */
  recallAtK: number;
  /** 命中率（0-1） */
  hitRate: number;
  /** 测试问题总数 */
  totalCases: number;
  /** 命中问题数 */
  hitCases: number;
}

/** 低质量片段（被点踩片段，用于反馈闭环） */
export interface LowQualityChunkVO {
  /** 分块 ID */
  chunkId: number;
  /** 分块内容 */
  content: string;
  /** 来源文档 ID */
  documentId: number;
  /** 来源文档标题 */
  documentTitle?: string;
  /** 分块序号 */
  chunkIndex?: number;
  /** 被点踩次数 */
  thumbsDownCount: number;
  /** 分块元数据 */
  metadata?: Record<string, unknown>;
}

/** 低质量片段查询参数（类型筛选 + 分页） */
export interface LowQualityChunkQuery extends PageQuery {
  /** 按片段类型筛选 */
  feedbackType?: string;
  /** 关键字搜索 */
  keyword?: string;
}

// ==================== 文档 ====================

/** 文档上传表单（基于已上传文件的 fileId 关联） */
export interface DocumentUploadForm {
  fileId: number;
  title?: string;
}

/** 批量上传表单（fileIds 列表，与后端 DocumentBatchUploadForm 对齐） */
export interface DocumentBatchUploadForm {
  fileIds: number[];
}

/** 导入网页为文档 */
export interface DocumentImportUrlForm {
  url: string;
  title?: string;
}

/** 自定义文本创建文档 */
export interface DocumentTextForm {
  title: string;
  content: string;
}

/** 文档视图对象 */
export interface DocumentVO {
  id: number;
  knowledgeBaseId: number;
  fileId?: number;
  title: string;
  source: DocumentSource;
  /** 解析策略（auto 自动根据文件类型选择） */
  parsingStrategy: string;
  processingStatus: DocumentProcessingStatus;
  /** 解析后的文本内容（文档详情接口返回） */
  content?: string;
  /** 结构化原始内容（含表格/代码块） */
  rawContent?: string;
  chunkCount: number;
  totalTokens: number;
  /** 版本号 */
  version: number;
  /** 处理失败时的错误信息 */
  error?: string;
  createTime: string;
  updateTime?: string;
}

/** 文档列表查询参数 */
export interface DocumentQuery extends PageQuery {
  /** 处理状态过滤 */
  processingStatus?: DocumentProcessingStatus;
  /** 关键字搜索 */
  keyword?: string;
}

// ==================== 分块 ====================

/** 分块视图对象 */
export interface ChunkVO {
  id: number;
  documentId: number;
  /** 分块序号 */
  chunkIndex: number;
  content: string;
  /** 元数据（页码/段落/类型等） */
  metadata?: Record<string, unknown>;
  tokenCount: number;
}

// ==================== 检索 ====================

/** 元数据过滤条件 */
export interface SearchFilters {
  /** 按文档类型过滤 */
  documentType?: string;
  /** 按标签过滤 */
  tags?: string[];
  /** 按时间范围过滤 */
  dateStart?: string;
  dateEnd?: string;
  /** 关联算法 ID */
  algorithmId?: number;
}

/** 知识库检索请求 */
export interface KnowledgeBaseSearchForm {
  /** 查询文本 */
  query: string;
  /** 指定知识库 ID 列表（为空检索所有可见知识库） */
  knowledgeBaseIds?: number[];
  /** 返回 Top-K 数量 */
  topK?: number;
  /** 元数据过滤 */
  filters?: SearchFilters;
  /** 是否启用 MMR 多样性去重 */
  enableMMR?: boolean;
}

/** 检索结果项 */
export interface SearchResultItem {
  /** 分块 ID */
  chunkId: number;
  /** 分块内容 */
  content: string;
  /** 分块元数据 */
  metadata?: Record<string, unknown>;
  /** 匹配分数 */
  score: number;
  /** 来源文档标题 */
  documentTitle: string;
  /** 来源文档 ID */
  documentId: number;
  /** 分块序号 */
  chunkIndex: number;
}

/** 检索结果 */
export interface SearchResultVO {
  query: string;
  /** 命中的知识库 ID 列表 */
  knowledgeBaseIds: number[];
  results: SearchResultItem[];
}

/** 检索测试请求（知识库管理页面的调试工具） */
export interface RetrieveTestForm {
  query: string;
  topK?: number;
  /** 是否启用 MMR 多样性去重 */
  enableMMR?: boolean;
}
