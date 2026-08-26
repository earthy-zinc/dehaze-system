"""
AI 知识库模块 Schema 模型

包含知识库/文档/分块管理、检索相关的 Form 与 VO。
Form 若字段为 snake_case 则配置 camelCase alias 以兼容前端提交；VO 继承 OrmResult
（字段与实体一致 snake_case，序列化输出 camelCase）。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.models.schema.common import BasePageQuery, OrmResult


# 分块策略(fixed:固定长度;semantic:语义切分;recursive:递归切分;qa:问答对;table:表格感知)
CHUNKING_STRATEGY_VALUES = ("fixed", "semantic", "recursive", "qa", "table")
ChunkingStrategy = Literal["fixed", "semantic", "recursive", "qa", "table"]
# 检索策略(vector:纯向量;keyword:纯关键词;hybrid:混合检索)
SearchStrategy = Literal["vector", "keyword", "hybrid"]
# 支持的 Embedding 模型标识(与 embedding_client._KNOWN_DIMS 对齐)
EMBEDDING_MODEL_VALUES = ("text-embedding-3-small", "text-embedding-3-large", "bge-m3")


class KnowledgeBaseCreateForm(BaseModel):
    """创建知识库表单（snake_case 字段，alias 兼容前端 camelCase 提交）"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=255, description="知识库名称")
    description: str | None = Field(default=None, description="知识库描述")
    visibility: Literal["public", "private"] = Field(
        ..., description="可见性(public:平台公共库;private:私有库)"
    )
    embedding_provider: str = Field(
        default="openai", max_length=32, description="Embedding提供商(openai;qwen;cohere;local)"
    )
    embedding_model: str = Field(
        ..., min_length=1, max_length=64, description="Embedding模型标识"
    )
    chunking_strategy: ChunkingStrategy = Field(
        ..., description="分块策略(fixed/semantic/recursive/qa/table)"
    )
    chunk_size: int = Field(default=800, ge=50, le=2000, description="分块大小(token)")
    chunk_overlap: int = Field(default=80, ge=0, description="分块重叠数(token)")
    search_strategy: SearchStrategy = Field(
        default="hybrid", description="检索策略(vector/keyword/hybrid)"
    )
    hybrid_weight: float = Field(
        default=0.7, ge=0, le=1, description="混合检索中向量权重(0-1)"
    )
    top_k: int = Field(default=5, ge=1, le=100, description="默认检索Top-K数")
    score_threshold: float = Field(
        default=0.5, ge=0, lt=1, description="相似度阈值"
    )
    enable_rerank: bool = Field(
        default=False, description="是否启用重排序(需额外Rerank模型)"
    )
    rerank_model: str | None = Field(
        default=None, max_length=64, description="重排序模型标识"
    )


class KnowledgeBaseUpdateForm(BaseModel):
    """编辑知识库表单（仅可编辑项；分块策略与 embedding 模型创建后不可修改）"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str | None = Field(default=None, min_length=1, max_length=255, description="知识库名称")
    description: str | None = Field(default=None, description="知识库描述")
    search_strategy: SearchStrategy | None = Field(
        default=None, description="检索策略(vector/keyword/hybrid)"
    )
    hybrid_weight: float | None = Field(
        default=None, ge=0, le=1, description="混合检索中向量权重(0-1)"
    )
    top_k: int | None = Field(default=None, ge=1, le=100, description="默认检索Top-K数")
    score_threshold: float | None = Field(
        default=None, ge=0, lt=1, description="相似度阈值"
    )
    enable_rerank: bool | None = Field(
        default=None, description="是否启用重排序"
    )
    rerank_model: str | None = Field(
        default=None, max_length=64, description="重排序模型标识"
    )
    # 以下两项用于服务端拒绝校验（创建后不可修改），不允许真正更新
    embedding_model: str | None = Field(default=None, max_length=64, description="Embedding模型标识")
    chunking_strategy: ChunkingStrategy | None = Field(
        default=None, description="分块策略(fixed/semantic/recursive/qa/table)"
    )


class KnowledgeBasePageQuery(BasePageQuery):
    keyword: str | None = Field(default=None, description="关键字(按知识库名称模糊搜索)")
    view: Literal["admin"] | None = Field(
        default=None, description="管理端视角(admin:返回全部知识库含私有库,仅管理员)"
    )


class KnowledgeBaseVO(OrmResult):
    """知识库视图对象（model_validate(实体) 构造，序列化输出 camelCase）"""

    id: int = Field(description="主键")
    name: str = Field(description="知识库名称")
    description: str | None = Field(default=None, description="知识库描述")
    visibility: str = Field(description="可见性(public/private)")
    embedding_provider: str = Field(description="Embedding提供商")
    embedding_model: str = Field(description="Embedding模型标识")
    chunking_strategy: str = Field(description="分块策略")
    chunk_size: int = Field(description="分块大小(token)")
    chunk_overlap: int = Field(description="分块重叠数(token)")
    search_strategy: str = Field(description="检索策略")
    hybrid_weight: float = Field(description="混合检索中向量权重")
    top_k: int = Field(description="默认检索Top-K数")
    score_threshold: float = Field(description="相似度阈值")
    enable_rerank: int = Field(description="是否启用重排序")
    rerank_model: str | None = Field(default=None, description="重排序模型标识")
    document_count: int = Field(description="文档数(冗余统计)")
    chunk_count: int = Field(description="分块总数(冗余统计)")
    total_tokens: int = Field(description="编码Token总数(冗余统计)")
    status: int = Field(description="状态(1:启用;2:处理中;0:禁用)")
    create_by: int | None = Field(default=None, description="创建人ID")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class DocumentUploadForm(BaseModel):
    fileId: int = Field(..., description="已上传文件ID(关联sys_file)")
    title: str | None = Field(default=None, max_length=512, description="文档标题(缺省用文件名)")


class DocumentBatchUploadForm(BaseModel):
    fileIds: list[int] = Field(..., min_length=1, description="待上传文件ID列表")


class DocumentImportUrlForm(BaseModel):
    url: str = Field(..., description="待导入网页URL")
    title: str | None = Field(default=None, max_length=512, description="文档标题(缺省用URL)")


class DocumentTextCreateForm(BaseModel):
    title: str = Field(..., min_length=1, max_length=512, description="文档标题")
    content: str = Field(..., min_length=1, description="自定义文本内容")


class DocumentUpdateForm(BaseModel):
    fileId: int | None = Field(default=None, description="新文件ID(重新上传)")
    content: str | None = Field(default=None, description="新文本内容(纯文本更新)")


class KnowledgeDocumentPageQuery(BasePageQuery):
    processingStatus: str | None = Field(
        default=None,
        description="处理状态过滤(pending/processing/completed/failed)",
    )


class KnowledgeDocumentVO(OrmResult):
    """文档视图对象（列表接口通过 exclude 剔除 content 大字段）"""

    id: int = Field(description="主键")
    knowledge_base_id: int = Field(description="知识库ID")
    file_id: int | None = Field(default=None, description="文件ID")
    title: str = Field(description="文档标题")
    source: str = Field(description="文档来源")
    version: int = Field(description="文档版本号")
    parsing_strategy: str = Field(description="解析策略")
    content: str | None = Field(default=None, description="解析后的纯文本内容")
    raw_content: str | None = Field(default=None, description="原始富文本")
    chunk_count: int = Field(description="分块数(冗余统计)")
    total_tokens: int = Field(description="编码Token总数(冗余统计)")
    processing_status: str = Field(description="处理状态")
    error: str | None = Field(default=None, description="失败原因")
    create_time: datetime | None = Field(default=None, description="创建时间")
    update_time: datetime | None = Field(default=None, description="更新时间")


class ChunkPreviewForm(BaseModel):
    """分块预览表单（snake_case 字段，alias 兼容前端 camelCase 提交）"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    fileId: int = Field(..., description="已上传文件ID(关联sys_file)")
    chunking_strategy: ChunkingStrategy = Field(
        ..., description="分块策略(fixed/semantic/recursive/qa/table)"
    )
    chunk_size: int = Field(default=800, ge=50, le=2000, description="分块大小(token)")
    chunk_overlap: int = Field(default=80, ge=0, description="分块重叠数(token)")


class KnowledgeChunkVO(OrmResult):
    id: int = Field(description="主键")
    document_id: int = Field(description="文档ID")
    chunk_index: int = Field(description="分块序号(从0开始)")
    content: str = Field(description="分块后的文本片段")
    token_count: int = Field(description="分块Token数")
    metadata: dict[str, Any] | None = Field(
        default=None,
        validation_alias="metadata_",
        description="分块元数据(来源文档/页码/段落/表格行等)",
    )
    create_time: datetime | None = Field(default=None, description="创建时间")


class LowQualityChunkVO(OrmResult):
    """低质量片段视图对象（被点踩片段，thumbsDownCount 按片段点踩次数聚合）"""

    chunk_id: int = Field(description="分块ID")
    content: str = Field(description="分块文本内容")
    document_id: int = Field(description="所属文档ID")
    thumbs_down_count: int = Field(description="被点踩次数")


class SearchFilters(BaseModel):
    """元数据过滤条件（snake_case 映射见 router._filters_to_dict）"""

    docType: str | None = Field(default=None, description="按文档类型过滤")
    tags: list[str] | None = Field(default=None, description="按标签过滤")
    startTime: datetime | None = Field(default=None, description="按创建时间起始过滤")
    endTime: datetime | None = Field(default=None, description="按创建时间结束过滤")
    algorithmId: int | None = Field(default=None, description="关联算法ID")
    entities: list[str] | None = Field(default=None, description="关联实体名称列表")
    relations: list[str] | None = Field(default=None, description="关联关系类型列表")


class SearchForm(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本")
    knowledgeBaseIds: list[int] | None = Field(
        default=None, description="指定知识库ID列表(为空检索所有可见库)"
    )
    topK: int | None = Field(default=None, ge=1, le=50, description="返回Top-K数量")
    filters: SearchFilters | None = Field(default=None, description="元数据过滤条件")
    enableMMR: bool = Field(default=False, description="是否启用MMR多样性去重")


class RetrieveTestForm(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本")
    topK: int | None = Field(default=None, ge=1, le=50, description="返回Top-K数量")
    enableMMR: bool = Field(default=False, description="是否启用MMR多样性去重")


class TestSetCreateForm(BaseModel):
    """创建召回测试集表单（snake_case 字段，alias 兼容前端 camelCase 提交）"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=1000, description="测试问题")
    expected_chunk_ids: list[int] = Field(
        ..., min_length=1, description="期望命中分块ID列表(必须召回)"
    )


class TestSetVO(OrmResult):
    """召回测试集视图对象（model_validate(实体) 构造，序列化输出 camelCase）"""

    id: int = Field(description="主键")
    knowledge_base_id: int = Field(description="知识库ID")
    question: str = Field(description="测试问题")
    expected_chunk_ids: list[int] = Field(description="期望命中分块ID列表")
    create_time: datetime | None = Field(default=None, description="创建时间")


class TestSetRunForm(BaseModel):
    """执行召回测试集表单（可选，缺省用知识库默认 Top-K）"""

    topK: int = Field(default=5, ge=1, le=50, description="检索Top-K数")


class RecallTestResultVO(BaseModel):
    """召回测试集执行结果：Recall@K 与命中率（0-1）"""

    test_set_id: int = Field(description="测试集ID")
    recall_at_k: float = Field(description="Recall@K：期望命中分块出现在Top-K结果中的比例(0-1)")
    hit_rate: float = Field(description="命中率：至少命中一条期望分块的用例占比(0-1)")
    total_cases: int = Field(description="测试问题总数")
    hit_cases: int = Field(description="至少命中一条期望分块的用例数")

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
