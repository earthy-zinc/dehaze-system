"""消息读模型：DB 实体 → 契约 Message。

历史读模型与流式终态共用本模块：同一条消息从 message.end 与刷新后的消息列表读出
完全相同的形状（blocks/plan/pendingInterrupt/usage/artifactIds/memoryIds/suggestions），
否则刷新前后渲染会分叉。
"""

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_message import SysAiMessage
from app.models.schema.ai_contract_gen import (
    Block,
    Interrupt,
    Message,
    Plan,
    SubAgentUsage,
    Usage,
)
from app.repository.ai_artifact_repository import ai_artifact_repository
from app.repository.ai_message_repository import AiMessageRepository


def _iso(value: datetime | None) -> str | None:
    """契约时间统一为本地 ISO-8601（秒精度，无时区偏移），与 Java/Go 读模型一致。"""
    return value.strftime("%Y-%m-%dT%H:%M:%S") if value else None


def build_usage(msg: SysAiMessage) -> Usage:
    """token/积分口径与 usage 事件一致；子 Agent 分项在收尾时落 sub_agent_usage。"""
    sub_agents = None
    raw_sub_agents = getattr(msg, "sub_agent_usage", None)
    if isinstance(raw_sub_agents, list) and raw_sub_agents:
        sub_agents = [SubAgentUsage.model_validate(item) for item in raw_sub_agents]
    return Usage(
        inputTokens=msg.input_tokens or 0,
        outputTokens=msg.output_tokens or 0,
        cachedInputTokens=msg.cached_input_tokens or 0,
        credits=msg.credits or 0,
        subAgents=sub_agents,
    )


def build_blocks(raw: Any) -> list[Block]:
    if not isinstance(raw, list):
        return []
    return [Block.model_validate(b) for b in raw if isinstance(b, dict)]


def build_message(
    msg: SysAiMessage,
    *,
    artifact_ids: list[int] | None = None,
    now: datetime | None = None,
) -> Message:
    """构造契约 Message；artifact_ids 由调用方批量预取（避免逐条查询的 N+1）。"""
    pending = msg.pending_interrupt
    return Message(
        id=msg.id,
        conversationId=msg.conversation_id,
        parentMessageId=msg.parent_message_id,
        role=msg.role,
        status=msg.status,
        model=msg.model,
        error=msg.error,
        blocks=build_blocks(msg.blocks),
        plan=Plan.model_validate(msg.plan) if isinstance(msg.plan, dict) else None,
        pendingInterrupt=interrupt_view(pending, now=now),
        usage=build_usage(msg),
        artifactIds=artifact_ids or [],
        memoryIds=list(msg.used_memory_ids or []),
        suggestions=list(msg.suggestions or []),
        edited=bool(msg.edited),
        createTime=_iso(msg.create_time) or "",
        updateTime=_iso(msg.update_time),
    )


def interrupt_view(pending: Any, *, now: datetime | None = None) -> Interrupt | None:
    """持久化的 pending_interrupt → 契约 Interrupt。

    过期判定在此按读时计算：expiresAt 已过的中断仍返回（前端需展示"不可续"），
    但 actionable=false，避免把过期中断当成可恢复入口。
    """
    if not isinstance(pending, dict):
        return None
    interrupt = Interrupt.model_validate(pending)
    current = now or datetime.now()
    try:
        expired = datetime.fromisoformat(interrupt.expiresAt) <= current
    except ValueError:
        expired = True
    return interrupt.model_copy(update={"actionable": not expired})


async def load_artifact_ids(db: AsyncSession, msg_ids: list[int]) -> dict[int, list[int]]:
    """批量查询消息关联的产物 ID（一次 IN 查询，按消息分组）。"""
    if not msg_ids:
        return {}
    artifacts = await ai_artifact_repository.list_by_message_ids(db, msg_ids)
    grouped: dict[int, list[int]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.message_id, []).append(artifact.id)
    return grouped


async def load_sibling_positions(
    repository: AiMessageRepository,
    db: AsyncSession,
    conv_id: int,
    msgs: list[SysAiMessage],
) -> dict[int, tuple[int, int]]:
    """批量计算 assistant 消息的兄弟序号：{message_id: (1 起序号, 兄弟总数)}。

    仅 parentMessageId 非空（分叉下的助手回复）参与，root 级与非 assistant 保持 null；
    一次 IN 查询取全部兄弟，避免逐条消息查询的 N+1。列表、详情与 message.start 共用
    本方法，保证刷新前后与流式首帧的分支切换器元数据同口径。
    """
    msg_ids = {m.id for m in msgs}
    parent_ids = sorted(
        {
            m.parent_message_id
            for m in msgs
            if m.role == "assistant" and m.parent_message_id is not None
        }
    )
    if not parent_ids:
        return {}
    grouped = await repository.list_assistant_sibling_ids(db, conv_id, parent_ids)
    positions: dict[int, tuple[int, int]] = {}
    for sibling_ids in grouped.values():
        count = len(sibling_ids)
        for index, msg_id in enumerate(sibling_ids, start=1):
            if msg_id in msg_ids:
                positions[msg_id] = (index, count)
    return positions


async def build_messages(
    db: AsyncSession,
    msgs: list[SysAiMessage],
    *,
    now: datetime | None = None,
    sibling_positions: dict[int, tuple[int, int]] | None = None,
) -> list[Message]:
    """批量构造契约 Message（产物 ID 一次批量查询）。

    sibling_positions 由调用方经 load_sibling_positions 计算（{message_id: (序号, 总数)}）；
    分支列表等不分页视角不传，siblingIndex/siblingCount 保持 null。
    """
    artifact_map = await load_artifact_ids(db, [m.id for m in msgs])
    position_map = sibling_positions or {}
    messages: list[Message] = []
    for m in msgs:
        message = build_message(m, artifact_ids=artifact_map.get(m.id, []), now=now)
        position = position_map.get(m.id)
        if position is not None:
            index, count = position
            message = message.model_copy(update={"siblingIndex": index, "siblingCount": count})
        messages.append(message)
    return messages


async def build_message_detail(
    db: AsyncSession,
    msg: SysAiMessage,
    *,
    now: datetime | None = None,
    sibling_position: tuple[int, int] | None = None,
) -> Message:
    artifact_map = await load_artifact_ids(db, [msg.id])
    detail = build_message(msg, artifact_ids=artifact_map.get(msg.id, []), now=now)
    if sibling_position is not None:
        index, count = sibling_position
        detail = detail.model_copy(update={"siblingIndex": index, "siblingCount": count})
    return detail
