from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAiConversation(BaseModel, SoftDeleteMixin):
    __tablename__ = "sys_ai_conversation"
    __table_args__ = {"comment": "AI对话会话表"}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="用户ID")
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="新对话",
        comment="会话标题(首条消息自动提取，支持手动修改)",
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="会话使用的模型标识(关联sys_ai_model.model_id)"
    )
    agent_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="会话使用的智能体编码(关联sys_ai_agent.agent_code,为空使用默认Agent)",
    )
    agent_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="会话锚定的Agent已发布版本号(创建/切换会话时写入,发布/回滚不影响进行中会话)",
    )
    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="会话摘要(超token阈值时自动摘要老消息)"
    )
    summary_upto_message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="摘要水位：已纳入摘要覆盖范围的最后一条消息ID(增量摘要推进依据)",
    )
    system_prompt: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="系统提示词(会话级)"
    )
    model_config: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="模型参数配置(temperature;top_p;max_tokens等)"
    )
    api_key_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="绑定的API Key ID(关联sys_api_key)"
    )
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="消息数(冗余计数)"
    )
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最后消息时间(会话列表按此排序)"
    )
    current_branch_message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="当前激活的分支末端消息ID"
    )
    last_read_message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="最后已读消息ID(多端已读未读状态同步)"
    )
    pinned: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, comment="是否置顶(0:否;1:是)"
    )
    pinned_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="置顶时间(置顶会话按此倒序)"
    )
    delete_time: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="软删时间(30天恢复窗口判定，超期由定时任务物理清理)"
    )
    title_source: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="auto",
        comment="标题来源(auto:LLM自动生成;manual:手动修改)",
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=1, comment="会话状态(1:活跃;2:已归档)"
    )
