from typing import Any

from sqlalchemy import BigInteger, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysKnowledgeTestSet(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_knowledge_test_set"
    __table_args__ = {"comment": "AI知识库召回测试集"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger, index=True, nullable=False, comment="知识库ID(关联sys_knowledge_base.id)"
    )
    question: Mapped[str] = mapped_column(
        String(1000), nullable=False, comment="测试问题"
    )
    expected_chunk_ids: Mapped[Any] = mapped_column(
        JSON, nullable=False, comment="期望命中分块ID数组(JSON，关联sys_knowledge_chunk.id)"
    )
