"""长期记忆提取服务：推理后从对话中提取值得记忆的信息。

职责：
- LLM 提取（含 metadata 结构化 + 已有记忆去重输入）
- 五因子重要性评分（0.3×Emotion + 0.25×Frequency + 0.2×Recency + 0.15×Novelty + 0.1×ExplicitMark）
- 保存 + 上限淘汰、敏感信息（PII）写入前过滤
- 反思整合、合并去重
"""

import json
import logging
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from app.config import settings
from app.database import get_db_session
from app.infrastructure.llm.llm_client import llm_client
from app.models.entity.sys_ai_memory import SysAiMemory
from app.repository.ai_memory_repository import ai_memory_repository
from app.service.ai.memory_es_service import sync_memory
from app.utils.pii import mask_pii

logger = logging.getLogger(__name__)

# 匹配 ```json ... ``` 或 ``` ... ``` 代码块，提取其中的 JSON 内容
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)

# 提取输入中携带的已有记忆条数（供 LLM 去重）
_DEDUP_MEMORY_LIMIT = 50

# 五因子重要性权重（对齐设计文档 §7.3）
_IMPORTANCE_WEIGHTS = {
    "emotion": 0.3,
    "frequency": 0.25,
    "recency": 0.2,
    "novelty": 0.15,
    "explicit_mark": 0.1,
}

_EXTRACTION_PROMPT = """分析以下对话，提取值得长期记忆的信息。

只提取以下类型的信息，并按类型给出对应的 metadata：
1. semantic（语义记忆）：用户偏好/事实/领域知识
   metadata: {{"category": "preference|fact", "fact": "该事实的要点", "is_preference": true|false}}
2. procedural（程序记忆）：操作技能/流程/工具使用方法
   metadata: {{"skill": "对应任务类型(如dehaze/evaluate)", "steps": "操作步骤要点",
              "params": "常用参数"}}
3. episodic（情景记忆）：带时空标签的事件记录
   metadata: {{"timestamp": "事件时间", "event": "事件描述", "outcome": "结果",
              "user_feedback": "用户反馈"}}

不需要提取的：
- 临时性的任务描述（如"帮我去雾这张图片"）
- 具体的图片处理结果
- 闲聊内容

不要提取已存在的记忆（避免重复）：已有记忆列表如下，若新信息与之语义重复则忽略。

已有记忆列表：
{existing_memories}

以 JSON 数组格式返回，每条记忆包含：
[{{"type": "semantic|procedural|episodic", "content": "记忆内容", "metadata": {{...}}}}]

如果没有值得记忆的信息，返回空数组 []。

对话内容：
"""

_IMPORTANCE_PROMPT = """请为以下记忆的重要性各维度打分（每个维度 0-100 的整数）。

记忆内容：{content}

从以下五个维度评分：
- emotion：情感强度（用户情绪越强烈分值越高）
- frequency：频率（信息被多次提及越重要）
- recency：时效性（与当前任务/近况的相关度）
- novelty：信息增益（与已有记忆相比的新颖程度）
- explicit_mark：显式标记（用户是否主动要求"记住"，是则给 100）

若用户明确说过"记住/记住这一点/别忘了"等显式记忆指令，则 explicit_mark 必须为 100，
且整体重要性为 100。

仅返回 JSON 对象，不要任何解释：
{{"emotion": 0-100, "frequency": 0-100, "recency": 0-100, "novelty": 0-100, "explicit_mark": 0-100}}
"""

_REFLECTION_PROMPT = """回顾以下近期记忆，分析其中的规律和模式，提取更高层次的抽象洞察。

输出格式（JSON 数组），每条洞察包含：
[{"type": "semantic|procedural", "content": "抽象洞察", "metadata": {...}}]

- semantic：抽象知识、事实、用户偏好
- procedural：操作技能、工作流程、工具使用方法

如果没有可提取的规律，返回空数组 []。

近期记忆：
"""

_MERGE_PROMPT = """以下两条记忆语义重复，请合并为一条更完整、统一的表述。

记忆1: {content_a}
记忆2: {content_b}

只返回合并后的记忆内容，不要任何解释或额外文字。
"""


def _compute_importance(factors: dict) -> int:
    """按五因子加权计算重要性（0-100）。

    显式"记住"指令（explicit_mark=100）直接得满分，其余按权重加权。
    """
    if factors.get("explicit_mark", 0) >= 100:
        return 100
    score = sum(
        _IMPORTANCE_WEIGHTS[name] * max(0, min(100, int(factors.get(name, 0) or 0)))
        for name in _IMPORTANCE_WEIGHTS
    )
    return int(round(score))


async def _score_importance(db, model_id: str, content: str) -> int:
    """独立 LLM prompt 对记忆做五因子评分，返回加权重要性（0-100）。"""
    try:
        raw = await _llm_text(
            db,
            model_id,
            _IMPORTANCE_PROMPT.format(content=content),
            system_prompt="你是记忆重要性评估助手，按五因子为记忆打分。",
            max_tokens=120,
        )
    except Exception as e:
        logger.warning("重要性评分失败: %s", e)
        return 50
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        factors = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("重要性评分非 JSON: %s", raw[:200])
        return 50
    if not isinstance(factors, dict):
        return 50
    return _compute_importance(factors)


async def extract_memories(user_id: int, model_id: str, messages: list[dict]) -> list[dict]:
    """从对话中提取记忆，返回提取的记忆列表。

    返回格式：[{"memory_type", "content", "metadata", "importance"}]
    - 输入注入该用户已有记忆列表供 LLM 去重（避免重复提取）
    - 写入前做 PII 过滤（命中则脱敏或整条丢弃）
    - importance 由五因子评分计算
    """
    conv_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-6:] if m.get("content"))
    if len(conv_text) < 50:
        return []

    async with get_db_session() as db:
        existing = await ai_memory_repository.get_active_by_user(
            db, user_id, limit=_DEDUP_MEMORY_LIMIT
        )
        existing_text = "\n".join(f"- {m.content}" for m in existing) or "（无）"

        try:
            content = ""
            async for chunk in llm_client.stream_chat(
                db,
                model_id,
                [
                    {
                        "role": "user",
                        "content": _EXTRACTION_PROMPT.format(existing_memories=existing_text)
                        + conv_text,
                    }
                ],
                system_prompt="你是记忆提取助手，只提取值得长期记忆的关键信息。",
                temperature=0,
                max_tokens=600,
            ):
                if chunk.type == "text_delta":
                    content += chunk.content

            content = content.strip()
            match = _JSON_BLOCK_RE.search(content)
            if match:
                content = match.group(1).strip()

            items = json.loads(content)
            if not isinstance(items, list):
                return []
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []

    result = []
    for item in items:
        if not isinstance(item, dict) or "content" not in item:
            continue
        raw_content = item["content"]
        # PII 过滤：命中则脱敏保留；若脱敏后无实质内容则整条丢弃
        masked = mask_pii(raw_content)
        if not masked or masked == "***":
            continue
        memory_type = item.get("type", "semantic")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
        async with get_db_session() as db:
            importance = await _score_importance(db, model_id, masked)
        result.append(
            {
                "memory_type": memory_type,
                "content": masked,
                "metadata": metadata,
                "importance": importance,
            }
        )
    return result


async def save_extracted_memories(user_id: int, memories: list[dict]) -> int:
    """保存提取的记忆到数据库，返回保存数量。

    写入后检查记忆上限，超出时归档重要性最低且最久未访问的记忆。
    """
    if not memories:
        return 0
    async with get_db_session() as db:
        for m in memories:
            memory = SysAiMemory(
                user_id=user_id,
                memory_type=m["memory_type"],
                content=m["content"],
                metadata_=m.get("metadata"),
                importance=m["importance"],
                source="conversation",
            )
            memory = await ai_memory_repository.create(db, memory)
            # 异步同步到 ES 向量索引（ES 未启用时静默跳过）
            await sync_memory(
                {
                    "id": memory.id,
                    "user_id": user_id,
                    "memory_type": m["memory_type"],
                    "content": m["content"],
                    "importance": m["importance"],
                    "status": 1,
                    "archived": 0,
                    "deleted": 0,
                }
            )

        count = await ai_memory_repository.count_active(db, user_id)
        if count > settings.AI_MEMORY_MAX_COUNT:
            await ai_memory_repository.archive_least_important(
                db, user_id, count - settings.AI_MEMORY_MAX_COUNT
            )
    return len(memories)


async def _llm_text(db, model_id: str, prompt: str, system_prompt: str, max_tokens: int) -> str:
    """调用 LLM 返回纯文本结果（聚合流式 text_delta）"""
    content = ""
    async for chunk in llm_client.stream_chat(
        db,
        model_id,
        [{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        temperature=0,
        max_tokens=max_tokens,
    ):
        if chunk.type == "text_delta":
            content += chunk.content
    return content.strip()


async def reflect_and_consolidate(db, user_id: int, model_id: str | None = None) -> int:
    """反思整合：查询用户近 7 天情景记忆，调用 LLM 分析规律，生成抽象洞察。

    洞察写入 sys_ai_memory（source=reflection，memory_type=procedural/semantic）。
    返回新增洞察数量。
    """
    model_id = model_id or settings.AI_DEFAULT_MODEL
    since = datetime.now() - timedelta(days=7)
    episodic = await ai_memory_repository.list_recent_episodic(db, user_id, since)
    if not episodic:
        return 0

    memory_text = "\n".join(f"- {m.content}" for m in episodic)
    try:
        content = await _llm_text(
            db,
            model_id,
            _REFLECTION_PROMPT + memory_text,
            system_prompt="你是记忆反思助手，从近期记忆中提炼抽象洞察。",
            max_tokens=500,
        )
    except Exception as e:
        logger.warning("Memory reflection failed: %s", e)
        return 0

    match = _JSON_BLOCK_RE.search(content)
    if match:
        content = match.group(1).strip()
    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Memory reflection: LLM 返回非 JSON: %s", content[:200])
        return 0
    if not isinstance(items, list):
        return 0

    saved = 0
    for item in items:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        memory_type = item.get("type", "semantic")
        importance = await _score_importance(db, model_id, item["content"])
        memory = SysAiMemory(
            user_id=user_id,
            memory_type=memory_type,
            content=item["content"],
            metadata_=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
            importance=importance,
            source="reflection",
        )
        await ai_memory_repository.create(db, memory)
        saved += 1
    return saved


async def merge_duplicates(db, user_id: int, model_id: str | None = None) -> int:
    """合并去重：检测语义重复记忆，调用 LLM 合并为更完整的单一条目。

    同 memory_type 分组内文本相似度 > 0.9 的记忆对：
    - 调用 LLM 合并为统一表述
    - importance 取较高值，access_count 累加
    - 旧记忆软删除，保留合并后的新记忆
    返回合并次数。
    """
    model_id = model_id or settings.AI_DEFAULT_MODEL
    memories = await ai_memory_repository.get_active_by_user(db, user_id, limit=1000)
    if len(memories) < 2:
        return 0

    groups: dict[str, list[SysAiMemory]] = {}
    for m in memories:
        groups.setdefault(m.memory_type, []).append(m)

    merged_count = 0
    consumed: set[int] = set()
    for mtype, group in groups.items():
        for i in range(len(group)):
            a = group[i]
            if a.id in consumed:
                continue
            for j in range(i + 1, len(group)):
                b = group[j]
                if b.id in consumed:
                    continue
                if SequenceMatcher(None, a.content, b.content).ratio() <= 0.9:
                    continue
                try:
                    merged_content = await _llm_text(
                        db,
                        model_id,
                        _MERGE_PROMPT.format(content_a=a.content, content_b=b.content),
                        system_prompt="你是记忆合并助手，将重复记忆合并为统一表述。",
                        max_tokens=200,
                    )
                except Exception as e:
                    logger.warning("Memory merge failed: %s", e)
                    continue
                if not merged_content:
                    continue

                new_memory = SysAiMemory(
                    user_id=user_id,
                    memory_type=mtype,
                    content=merged_content,
                    metadata_=a.metadata_ or b.metadata_,
                    importance=max(a.importance, b.importance),
                    access_count=a.access_count + b.access_count,
                    source="merge",
                )
                await ai_memory_repository.create(db, new_memory)
                await ai_memory_repository.soft_delete_with_time(db, [a.id, b.id])
                consumed.add(a.id)
                consumed.add(b.id)
                merged_count += 1
                break
    return merged_count
