"""步骤摘要服务：对一条消息的全部推理步骤生成一句话概括（设计文档 §5.4 两级展示）

一级展示"步骤摘要"由 LLM 对 agent_thought 每步生成一句概括（做了什么 + 结果如何），
在 synthesize_response 阶段异步触发（不阻塞主回复），失败记 warning 日志，不影响推理结果。

一次 LLM 调用批量概括全部步骤（输入 steps 列表，输出 JSON 数组），
按 position 回填各 thought 的 summary 字段。
"""

import json
import logging

from app.dependencies.redis import get_redis_client
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository
from app.service.ai.llm_client import llm_client

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """请为以下 AI 推理过程的每个步骤生成一句话概括（做什么 + 结果如何）。

每个步骤以 JSON 对象描述，包含 thought（思考/工具说明）、tool（工具名，可为空）、
observation（结果摘要）。请按输入顺序为每个步骤输出对应的一句话概括。

严格返回 JSON 数组，每个元素是一条概括字符串，不要任何解释或额外文字：
["步骤1概括", "步骤2概括", ...]

推理步骤：
"""


async def _generate_summaries(db, model_id: str, steps: list[dict]) -> list[str] | None:
    """一次 LLM 调用批量生成步骤概括（temperature=0，失败返回 None）。"""
    steps_text = json.dumps(steps, ensure_ascii=False, indent=1)
    content = ""
    redis = await get_redis_client()
    async for chunk in llm_client.stream_chat(
        db,
        redis,
        model_id,
        [{"role": "user", "content": _SUMMARY_PROMPT + steps_text}],
        system_prompt="你是推理步骤摘要助手，为每一步生成一句话概括。",
        temperature=0,
        max_tokens=800,
    ):
        if chunk.type == "text_delta":
            content += chunk.content
    content = content.strip()
    if content.startswith("```"):
        # 剥离 ```json ... ``` 代码块
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        summaries = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("步骤摘要 LLM 返回非 JSON，丢弃: %s", content[:200])
        return None
    if not isinstance(summaries, list):
        return None
    return [str(s).strip() for s in summaries if isinstance(s, str) and s.strip()]


async def summarize_steps(message_id: int, model_id: str) -> None:
    """对本条消息全部推理步骤生成一句话概括并回填 summary 字段（失败记 warning 日志）。

    无 steps 或 LLM 调用失败时记 warning 日志返回，不影响主推理结果。
    """
    if not message_id or not model_id:
        return
    from app.database import get_db_session

    try:
        async with get_db_session() as db:
            thoughts = await ai_agent_thought_repository.list_by_message(db, message_id)
            if not thoughts:
                return
            steps = [
                {
                    "thought": t.thought,
                    "tool": t.tool,
                    "observation": (t.observation or "")[:200],
                }
                for t in thoughts
            ]
            try:
                summaries = await _generate_summaries(db, model_id, steps)
            except Exception as e:
                logger.warning("步骤摘要生成失败 msg_id=%s: %s", message_id, e)
                return
            if not summaries:
                return
            for thought, summary in zip(thoughts, summaries, strict=False):
                await ai_agent_thought_repository.update(db, thought, {"summary": summary})
    except Exception:
        logger.warning("步骤摘要处理异常 msg_id=%s", message_id, exc_info=True)


# 供推理层异步触发时引用（防止后台任务被垃圾回收）
def schedule_step_summaries(message_id: int, model_id: str) -> None:
    """异步触发步骤摘要生成（不阻塞主回复）。"""
    import asyncio

    from app.service.ai.reasoning_service import _pending_tasks

    async def _run() -> None:
        try:
            await summarize_steps(message_id, model_id)
        except Exception:
            logger.warning("步骤摘要后台任务异常 msg_id=%s", message_id, exc_info=True)

    task = asyncio.create_task(_run())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
