"""长期记忆注入服务：推理前按三层机制组装长期记忆并注入上下文。

三层注入（对齐设计文档 §7.4）：
- 常驻注入（Always-on）：语义记忆 is_preference=1 的偏好/身份，全量注入，不检索不省略
- 场景触发注入（Trigger-based）：程序记忆 metadata.skill 匹配当前任务类型，命中注入工作流
- 检索注入（Retrieval-based）：ES 向量检索 + 三维权重排序，降级 MySQL LIKE

inject_memories 首个位置参数 db 用于记忆查询（常驻/场景触发/检索三层均需访问
数据库），返回值 (system_block_text, injected_list) 二元组供推理层消费并落注入可见性。
"""

import math
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_memory_repository import ai_memory_repository
from app.service.ai.service.memory_es_service import search_memories

# 检索注入 Top N（可配置，ES/降级共用）
DEFAULT_RETRIEVAL_LIMIT = 5
# 三维权重（Relevance 语义相关 / Recency 近因 / Importance 重要性）
_WEIGHT_RELEVANCE = 0.5
_WEIGHT_RECENCY = 0.3
_WEIGHT_IMPORTANCE = 0.2
# Recency 衰减半衰期（天）：exp(-Δt/30d)
_RECENCY_HALF_LIFE_DAYS = 30


def _recency_score(last_accessed_at: datetime | None, create_time: datetime | None) -> float:
    """近因性分数：越近访问越接近 1，按 30 天半衰期指数衰减。"""
    reference = last_accessed_at or create_time or datetime.now()
    delta_days = max(0.0, (datetime.now() - reference).total_seconds() / 86400)
    return math.exp(-delta_days / _RECENCY_HALF_LIFE_DAYS)


def _memory_importance(m: dict | object) -> float:
    """取记忆重要性（0-100），映射到 0-1。"""
    importance = m["importance"] if isinstance(m, dict) else m.importance
    return max(0.0, min(100.0, float(importance))) / 100.0


def _memory_id(m: dict | object) -> int:
    return m["id"] if isinstance(m, dict) else m.id


def _sort_retrieved(memories: list[dict]) -> list[dict]:
    """三维权重排序：Score = α×Relevance + β×Recency + γ×Importance。

    Relevance 取 ES kNN 分数并在候选集内归一化到 0-1；MySQL 降级路径无真实分数，
    统一视为 1.0（均命中关键词）。Recency 按 last_accessed_at/create_time 指数衰减。
    """
    if not memories:
        return []
    max_relevance = max((float(m.get("relevance", 0.0)) for m in memories), default=1.0)
    if max_relevance <= 0:
        max_relevance = 1.0

    def _score(m: dict) -> float:
        relevance = float(m.get("relevance", 1.0)) / max_relevance
        recency = _recency_score(m.get("last_accessed_at"), m.get("create_time"))
        importance = _memory_importance(m)
        return (
            _WEIGHT_RELEVANCE * relevance
            + _WEIGHT_RECENCY * recency
            + _WEIGHT_IMPORTANCE * importance
        )

    return sorted(memories, key=_score, reverse=True)


def _resolve_task_type(query: str, skills: list[str]) -> str | None:
    """任务类型兜底：调用方未传 task_type 时，按最后用户消息关键词匹配用户已有 skill。

    避免场景触发层因任务类型缺失而永不生效。匹配规则：
    - 若某 skill 关键字出现在查询文本中（不区分大小写），返回该 skill；
    - 同时校验 skill 是否与查询有语义重叠（skill 全含于 query 或反向包含），
      降低误命中。
    """
    lowered = query.lower()
    for skill in skills:
        if not skill:
            continue
        skill_lower = skill.lower()
        if skill_lower in lowered or lowered in skill_lower:
            return skill
    return None


async def _retrieval_layer(
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int,
) -> list[dict]:
    """检索注入层：ES 向量检索（三维权排序）+ 降级 MySQL LIKE，命中重激活。"""
    memories = await search_memories(user_id, query, top_n=limit)
    if not memories:
        memories = [
            {
                "id": m.id,
                "memory_type": m.memory_type,
                "content": m.content,
                "importance": m.importance,
                "last_accessed_at": m.last_accessed_at,
                "create_time": m.create_time,
            }
            for m in await ai_memory_repository.search_by_keyword(db, user_id, query, limit=limit)
        ]
    if not memories:
        return []
    memories = _sort_retrieved(memories)[:limit]
    # 命中重激活（重置衰减计时器，类似人类"复习巩固"）
    for m in memories:
        await ai_memory_repository.touch(db, _memory_id(m))
    return memories


async def inject_memories(
    db: AsyncSession,
    user_id: int,
    query: str,
    task_type: str | None = None,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> tuple[str | None, list[dict]]:
    """推理前按三层机制组装长期记忆。

    Args:
        db: 数据库会话。
        user_id: 用户 ID。
        query: 当前对话内容（用于检索注入的语义向量/关键词）。
        task_type: 当前任务类型（如 dehaze），命中程序记忆做场景触发注入；None 跳过该层。
        limit: 检索注入 Top N。

    Returns:
        (system_block_text, injected_list)：
        - system_block_text：拼装好的 system 补充块文本，无任何注入时返回 None。
        - injected_list：注入可见性清单，元素为
          {memory_id, memory_type, content, source}，供推理层落库展示。
    """
    if not query or len(query) < 2:
        return None, []

    injected: list[dict] = []

    preferences = await ai_memory_repository.list_preferences(db, user_id)
    for m in preferences:
        injected.append(
            {
                "memory_id": m.id,
                "memory_type": m.memory_type,
                "content": m.content,
                "source": "preference",
            }
        )

    # 调用方未传 task_type 时，兜底按最后用户消息关键词匹配用户已有 skill，避免该层永不生效
    effective_task_type = task_type
    if not effective_task_type:
        effective_task_type = _resolve_task_type(
            query, await ai_memory_repository.list_skills(db, user_id)
        )
    if effective_task_type:
        skill_memories = await ai_memory_repository.list_by_skill(db, user_id, effective_task_type)
        for m in skill_memories:
            injected.append(
                {
                    "memory_id": m.id,
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "source": "skill",
                }
            )

    retrieval = await _retrieval_layer(db, user_id, query, limit)
    for m in retrieval:
        injected.append(
            {
                "memory_id": _memory_id(m),
                "memory_type": m["memory_type"],
                "content": m["content"],
                "source": "retrieval",
            }
        )

    if not injected:
        return None, []

    sections = []
    preference_items = [i for i in injected if i["source"] == "preference"]
    if preference_items:
        body = "\n".join(f"- {i['content']}" for i in preference_items)
        sections.append("【用户画像】以下是关于该用户的长期偏好，请在回复时始终参考：\n" + body)

    skill_items = [i for i in injected if i["source"] == "skill"]
    if skill_items:
        body = "\n".join(f"- {i['content']}" for i in skill_items)
        sections.append("【工作流提示】当前任务可参考该用户的历史处理习惯/工作流程：\n" + body)

    retrieval_items = [i for i in injected if i["source"] == "retrieval"]
    if retrieval_items:
        body = "\n".join(f"- {i['content']}" for i in retrieval_items)
        sections.append("【相关记忆】以下是与当前对话相关的历史记忆，请在回复时参考：\n" + body)

    return "\n\n".join(sections), injected
