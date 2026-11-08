from typing import Any
from uuid import uuid4

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin

# 块类型（契约 Block.kind）
BLOCK_KIND_TEXT = "text"
BLOCK_KIND_THINKING = "thinking"
BLOCK_KIND_TOOL_USE = "tool_use"


def text_block(text: str, *, seq: int = 1) -> dict[str, Any]:
    """构造一个 text 块（用户消息落库的唯一形状：一条消息一个 text 块）。"""
    return {"id": uuid4().hex, "kind": BLOCK_KIND_TEXT, "seq": seq, "text": text}


class SysAiMessage(BaseModel, SoftDeleteMixin):
    """AI 对话消息。

    消息只有一种形状：正文、思考、工具调用统一由 blocks 承载；plan / pending_interrupt /
    suggestions 是消息级附属状态，刷新后从同一行读回，与流式终态一致。
    """

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
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="running",
        comment="消息状态(running;completed;failed;canceled;awaiting_input)",
    )
    blocks: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="内容块列表(Block[]:id/kind/seq/text/durationMs/tool)",
    )
    plan: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="执行计划(Plan:phase/status/tasks/revisions/awaitingApproval)"
    )
    pending_interrupt: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="待恢复中断(Interrupt:含expiresAt/actionable)"
    )
    suggestions: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="推荐追问列表(string[])"
    )
    model: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="本条消息使用的模型标识"
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="错误信息(status=failed时填充)"
    )
    used_memory_ids: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="本次注入引用的记忆ID列表(JSON数组,注入可见性)"
    )
    sub_agent_usage: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="子智能体分项用量(SubAgentUsage[]，与 usage.subAgents 同形)"
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

    @property
    def text(self) -> str:
        """正文派生：同 kind=text 的块按 seq 拼接。

        读模型、上下文组装、导出、ES 全文索引统一经此取值，不再单独存 content 列，
        避免"正文"出现两份事实源。
        """
        return blocks_text(self.blocks)


def blocks_text(blocks: Any) -> str:
    """拼接 text 块正文（按 seq 正序）；非列表入参视为空。

    块间以空行分隔：同一消息的多个 text 块是分段产出（正文→工具→正文），
    直接首尾相接会把两段话粘成一句；前端与 Go 读模型同口径。
    """
    if not isinstance(blocks, list):
        return ""
    text_blocks = sorted(
        (b for b in blocks if isinstance(b, dict) and b.get("kind") == BLOCK_KIND_TEXT),
        key=lambda b: b.get("seq") or 0,
    )
    return "\n\n".join(text for b in text_blocks if (text := (b.get("text") or "")))


def blocks_context_text(blocks: Any, *, tool_max_len: int = 500) -> str:
    """上下文用正文：正文块 + 工具调用结果摘要。

    工具调用不再是独立的 role=tool 消息，结果挂在 assistant 的 tool_use 块上；
    组装历史上下文时折叠回本轮 assistant 文本，使后续轮次仍能看到工具产出。
    """
    if not isinstance(blocks, list):
        return ""
    text = blocks_text(blocks)
    tool_lines: list[str] = []
    for block in sorted(
        (b for b in blocks if isinstance(b, dict) and b.get("kind") == BLOCK_KIND_TOOL_USE),
        key=lambda b: b.get("seq") or 0,
    ):
        tool = block.get("tool") or {}
        output = tool.get("output")
        if not output:
            continue
        name = tool.get("name") or tool.get("callId") or ""
        if len(output) > tool_max_len:
            output = output[:tool_max_len] + "…"
        tool_lines.append(f"[工具调用结果 {name}] {output}")
    return "\n".join([text, *tool_lines]) if tool_lines else text


def block_seq_base(blocks: Any) -> dict[str, int]:
    """各 kind 的 seq 基线（现有最大值），resume 续号与终态快照共用。"""
    base: dict[str, int] = {}
    if not isinstance(blocks, list):
        return base
    for b in blocks:
        if not isinstance(b, dict):
            continue
        kind = b.get("kind")
        seq = b.get("seq") or 0
        if isinstance(kind, str) and seq > base.get(kind, 0):
            base[kind] = seq
    return base
