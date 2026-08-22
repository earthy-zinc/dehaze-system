"""Token 统计与积分换算"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.ai_model_repository import ai_model_repository


async def calculate_credits(
    db: AsyncSession,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> int:
    """按模型计费比例换算积分

    credits = inputTokens × inputRate + cachedInputTokens × cachedRate + outputTokens × outputRate
    """
    model = await ai_model_repository.get_by_model_id(db, model_id)
    if not model:
        return 0
    credits = (
        input_tokens * float(model.input_rate)
        + cached_input_tokens * float(model.cached_rate)
        + output_tokens * float(model.output_rate)
    )
    return int(credits)
