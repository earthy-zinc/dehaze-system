import { PageResult } from "@/types";
import request from "@/utils/request";
import {
  ChunkVO,
  ChunkingStrategy,
  DocumentBatchUploadForm,
  DocumentImportUrlForm,
  DocumentQuery,
  DocumentTextForm,
  DocumentUploadForm,
  DocumentVO,
  KnowledgeBaseCreateForm,
  KnowledgeBaseQuery,
  KnowledgeBaseSearchForm,
  KnowledgeBaseUpdateForm,
  KnowledgeBaseVO,
  RetrieveTestForm,
  SearchResultVO,
} from "./model";

/**
 * AI 知识库 API
 *
 * 知识库 CRUD、文档管理（上传/批量/导入网页/自定义文本）、分块预览、检索。
 */
class AiKnowledgeBaseAPI {
  // ==================== 知识库管理 ====================

  /** 创建知识库 */
  static create(data: KnowledgeBaseCreateForm) {
    return request<KnowledgeBaseVO>({
      url: "/api/v1/kb",
      method: "post",
      data,
    });
  }

  /** 知识库列表（按可见性过滤：私有库仅创建者可见，公共库全员只读） */
  static getList(query?: KnowledgeBaseQuery) {
    return request<PageResult<KnowledgeBaseVO[]>>({
      url: "/api/v1/kb",
      method: "get",
      params: query,
    });
  }

  /** 知识库详情（含配置、统计信息） */
  static getDetail(id: number) {
    return request<KnowledgeBaseVO>({
      url: `/api/v1/kb/${id}`,
      method: "get",
    });
  }

  /** 编辑知识库（名称/描述/检索策略，分块策略和 embedding 模型不可修改） */
  static update(id: number, data: KnowledgeBaseUpdateForm) {
    return request<KnowledgeBaseVO>({
      url: `/api/v1/kb/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除知识库（软删除，同步删除 ES 索引） */
  static delete(id: number) {
    return request({
      url: `/api/v1/kb/${id}`,
      method: "delete",
    });
  }

  // ==================== 文档管理 ====================

  /** 上传文档（关联已上传文件的 fileId） */
  static uploadDocument(knowledgeBaseId: number, data: DocumentUploadForm) {
    return request<DocumentVO>({
      url: `/api/v1/kb/${knowledgeBaseId}/documents`,
      method: "post",
      data,
    });
  }

  /** 批量上传文档（逐条返回成功/失败结果） */
  static batchUploadDocuments(knowledgeBaseId: number, data: DocumentBatchUploadForm) {
    return request<
      {
        fileId: number;
        success: boolean;
        id?: number;
        processingStatus?: string;
        code?: string;
        message?: string;
      }[]
    >({
      url: `/api/v1/kb/${knowledgeBaseId}/documents/batch`,
      method: "post",
      data,
    });
  }

  /** 导入网页为文档 */
  static importUrlDocument(knowledgeBaseId: number, data: DocumentImportUrlForm) {
    return request<DocumentVO>({
      url: `/api/v1/kb/${knowledgeBaseId}/documents/import-url`,
      method: "post",
      data,
    });
  }

  /** 自定义文本创建文档 */
  static createTextDocument(knowledgeBaseId: number, data: DocumentTextForm) {
    return request<DocumentVO>({
      url: `/api/v1/kb/${knowledgeBaseId}/documents/text`,
      method: "post",
      data,
    });
  }

  /** 知识库文档列表（含处理状态） */
  static getDocuments(knowledgeBaseId: number, query?: DocumentQuery) {
    return request<PageResult<DocumentVO[]>>({
      url: `/api/v1/kb/${knowledgeBaseId}/documents`,
      method: "get",
      params: query,
    });
  }

  /** 文档详情（含解析后内容） */
  static getDocumentDetail(id: number) {
    return request<DocumentVO>({
      url: `/api/v1/kb/documents/${id}`,
      method: "get",
    });
  }

  /** 删除文档及关联分块 */
  static deleteDocument(id: number) {
    return request({
      url: `/api/v1/kb/documents/${id}`,
      method: "delete",
    });
  }

  /** 重新处理文档（重置状态为 pending，删除旧分块重新走异步流水线） */
  static reprocessDocument(id: number) {
    return request<DocumentVO>({
      url: `/api/v1/kb/documents/${id}/reprocess`,
      method: "post",
    });
  }

  // ==================== 分块管理 ====================

  /** 文档分块列表（分页） */
  static getChunks(documentId: number) {
    return request<PageResult<ChunkVO[]>>({
      url: `/api/v1/kb/documents/${documentId}/chunks`,
      method: "get",
    });
  }

  /** 文档分块预览（基于 fileId + 分块配置，返回分块效果预览，不向量化不写索引） */
  static previewChunks(data: {
    fileId: number;
    chunkingStrategy: ChunkingStrategy;
    chunkSize?: number;
    chunkOverlap?: number;
  }) {
    return request<{ index: number; content: string; tokenCount: number }[]>({
      url: `/api/v1/kb/documents/chunks/preview`,
      method: "post",
      data,
    });
  }

  // ==================== 检索 ====================

  /** 知识库检索（支持多知识库、元数据过滤、Rerank） */
  static search(data: KnowledgeBaseSearchForm) {
    return request<SearchResultVO>({
      url: "/api/v1/kb/search",
      method: "post",
      data,
    });
  }

  /** 检索测试（知识库管理页面的调试工具） */
  static retrieveTest(knowledgeBaseId: number, data: RetrieveTestForm) {
    return request<SearchResultVO>({
      url: `/api/v1/kb/${knowledgeBaseId}/retrieve/test`,
      method: "post",
      data,
    });
  }
}

export default AiKnowledgeBaseAPI;
