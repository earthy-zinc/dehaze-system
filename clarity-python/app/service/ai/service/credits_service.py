"""Token 统计与积分换算"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.service.ai_model_price_service import ai_model_price_service


async def calculate_credits(
    db: AsyncSession,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> int:
    """按模型用户售价换算积分（sys_ai_model_price，见 AI模型管理 §2.12）

    消息侧预计算（展示用），真实扣费以 settle 为准；未配置价格返回 0（配置缺失由结算侧暴露）。
    """
    result = await ai_model_price_service.calculate(
        db,
        model_id,
        None,
        datetime.now(),
        input_tokens,
        cached_input_tokens,
        output_tokens,
    )
    return result["credits"]
