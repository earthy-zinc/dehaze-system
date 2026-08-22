from typing import Any

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiMessage(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_message"
    __table_args__ = {"comment": "AI对话消息表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="会话ID(关联sys_ai_conversation.id)"
    )
    parent_message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="父消息ID(支持分支对话)"
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="消息角色(system;user;assistant;tool)"
    )
    content: Mapped[str | None] = mapped_column(LONGTEXT, nullable=True, comment="消息内容")
    tool_calls: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="工具调用列表(assistant消息触发)"
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="工具调用结果关联ID(role=tool时关联)"
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="本条消息使用的模型标识"
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="消息状态(1:流式输出中;2:已完成;3:失败;4:已取消)",
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="错误信息(status=3时填充)"
    )
    metadata_: Mapped[Any | None] = mapped_column(
        "metadata", JSON, nullable=True, comment="元数据(多模态读取次数;工具调用耗时;RAG检索命中等)"
    )
    used_memory_ids: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="本次注入引用的记忆ID列表(JSON数组,注入可见性)"
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输入Token数(含缓存命中部分)"
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="输出Token数"
    )
    cached_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="其中缓存命中的输入Token数"
    )
    credits: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment="消耗积分数(按模型计费比例换算后)"
    )
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="关联异步任务ID")
    edited: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否已编辑(0:否;1:是)"
    )
    original_content: Mapped[str | None] = mapped_column(
        LONGTEXT, nullable=True, comment="编辑前原文(edited=1时填充)"
    )
