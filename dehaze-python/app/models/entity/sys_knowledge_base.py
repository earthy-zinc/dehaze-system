from sqlalchemy import BigInteger, Integer, Numeric, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysKnowledgeBase(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_knowledge_base"
    __table_args__ = {"comment": "AI知识库主表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="知识库名称")
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="知识库描述")
    visibility: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="private",
        comment="可见性(public:平台公共库全员只读;private:私有库仅创建者可读写)",
    )
    embedding_provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="openai",
        comment="Embedding提供商(openai;qwen;cohere;local等)",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="text-embedding-3-small",
        comment="Embedding模型标识(如text-embedding-3-small;bge-m3等)",
    )
    chunking_strategy: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="semantic",
        comment="分块策略(fixed:固定长度;semantic:语义切分;recursive:递归切分;qa:问答对;table:表格感知)",
    )
    chunk_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=800, comment="分块大小(token数，范围的中间值)"
    )
    chunk_overlap: Mapped[int] = mapped_column(
        Integer, nullable=False, default=80, comment="分块重叠数(token)"
    )
    search_strategy: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="hybrid",
        comment="检索策略(vector:纯向量;keyword:纯关键词BM25;hybrid:混合检索)",
    )
    hybrid_weight: Mapped[float] = mapped_column(
        Numeric(3, 2), nullable=False, default=0.70, comment="混合检索中向量权重(0-1，剩余为关键词权重)"
    )
    top_k: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5, comment="默认检索Top-K数"
    )
    score_threshold: Mapped[float] = mapped_column(
        Numeric(4, 3), nullable=False, default=0.500, comment="相似度阈值(低于此分数的结果不返回)"
    )
    enable_rerank: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否启用重排序(0:否;1:是,需额外Rerank模型)"
    )
    rerank_model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="重排序模型标识(bge-reranker-v2-m3等)"
    )
    document_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="文档数(冗余统计)"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="分块总数(冗余统计)"
    )
    total_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="编码Token总数(冗余统计)"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="状态(1:启用;2:处理中;0:禁用)"
    )
