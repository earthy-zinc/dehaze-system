from typing import Any

from sqlalchemy import BigInteger, Integer, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysKnowledgeChunk(BaseModel):
    __tablename__ = "sys_knowledge_chunk"
    __table_args__ = {"comment": "AI知识库分块表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger, index=True, nullable=False, comment="文档ID(关联sys_knowledge_document.id)"
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger, index=True, nullable=False, comment="知识库ID(冗余，便于跨文档检索)"
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="分块序号(从0开始)"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="分块后的文本片段")
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="分块Token数"
    )
    metadata_: Mapped[Any | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
        comment="分块元数据(来源文档/页码/段落/表格行等，检索时用于引用展示)",
    )
