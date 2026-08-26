from typing import Any

from sqlalchemy import BigInteger, SmallInteger, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysKnowledgeChunkFeedback(BaseModel):
    __tablename__ = "sys_knowledge_chunk_feedback"
    __table_args__ = (
        UniqueConstraint("chunk_id", "user_id", name="uk_chunk_user"),
        {"comment": "AI知识库分块反馈表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    chunk_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False, comment="分块ID(关联sys_knowledge_chunk.id)")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="评分(1:点赞;-1:点踩)")
    comment: Mapped[str | None] = mapped_column(Text, nullable=True, comment="反馈内容(可选,点踩原因)")
