"""算法推荐编排服务：图像特征分析 → 推荐算法 → 注册产物 → 推送 interrupt"""

import logging

from app.database import get_db_session
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.service.ai_artifact_service import ai_artifact_service
from app.service.recommendation_service import recommendation_service

logger = logging.getLogger(__name__)


async def recommend_algorithm(
    conv_id: int,
    msg_id: int,
    user_id: int,
    image_url: str,
    user_query: str,
    stream_session_id: str,
) -> dict:
    """算法推荐编排

    1. 调用推荐管理模块分析图像特征
    2. 调用推荐管理模块获取推荐算法
    3. 注册 artifact（type=algorithm_recommend, ref_type=sys_recommendation）
    4. 推送 interrupt 事件（type=confirm）等待用户确认
    5. 返回推荐结果摘要
    """
    async with get_db_session() as db:
        analysis = await recommendation_service.analyze(None, image_url)

        # 2. 推荐算法（内部创建 sys_recommendation 记录）
        algorithms = await recommendation_service.get_algorithms(
            db,
            user_id,
            None,
            analysis.imageMd5,
        )

        if not algorithms:
            return {"recommendation": None, "alternatives": []}

        # 3. 注册 artifact（summary 只存业务摘要，不含图片 URL）
        top = algorithms[0]
        alternatives = algorithms[1:]
        summary = {
            "algorithm": {
                "recommendationId": top.recommendationId,
                "algorithmId": top.algorithmId,
                "algorithmName": top.algorithmName,
                "reason": top.reason,
                "effectDescription": top.effectDescription,
            },
            "matchScore": top.matchScore,
            "alternatives": [
                {
                    "algorithmId": a.algorithmId,
                    "algorithmName": a.algorithmName,
                    "matchScore": a.matchScore,
                    "reason": a.reason,
                }
                for a in alternatives
            ],
        }
        artifact = await ai_artifact_service.register_artifact(
            db,
            conv_id,
            msg_id,
            artifact_type="algorithm_recommend",
            ref_type="sys_recommendation",
            ref_id=top.recommendationId,
            summary=summary,
        )

    # 4. 持久化 interrupt（供 resume 恢复；SSE 推送由 SseEventConverter 统一处理 __interrupt__）
    thread_id = f"{conv_id}:{msg_id}"
    interrupt_data = {
        "type": "confirm",
        "stream_session_id": stream_session_id,
        "data": {
            "artifactId": artifact.id,
            "recommendation": summary["algorithm"],
            "alternatives": summary["alternatives"],
            "imageFeatures": {
                "hazeLevel": analysis.hazeLevel,
                "sceneType": analysis.sceneType,
                "lighting": analysis.lighting,
            },
        },
    }
    await interrupt_handler.save_interrupt(thread_id, "confirm", interrupt_data)

    return summary, interrupt_data


async def handle_user_confirmation(
    conv_id: int,
    msg_id: int,
    user_id: int,
    confirmed: bool,
    algorithm_id: int | None = None,
) -> dict:
    """处理用户对推荐结果的确认/拒绝

    confirmed=True: 用户接受推荐（可指定 algorithm_id 表示选择了备选算法）
    confirmed=False: 用户拒绝推荐
    """
    thread_id = f"{conv_id}:{msg_id}"
    interrupt = await interrupt_handler.get_interrupt(thread_id)
    if not interrupt:
        return {"status": "no_interrupt"}

    await interrupt_handler.clear_interrupt(thread_id)

    if not confirmed:
        return {"status": "rejected", "algorithmId": None}

    # 用户确认，提交正向反馈到推荐管理模块（使用 sys_recommendation 记录 id）
    interrupt_data = interrupt.get("data") or {}
    confirm_payload = interrupt_data.get("data") or {}
    recommendation = confirm_payload.get("recommendation") or {}
    recommendation_id = recommendation.get("recommendationId")
    if recommendation_id:
        async with get_db_session() as db:
            await recommendation_service.submit_feedback(db, recommendation_id, True)

    return {"status": "accepted", "algorithmId": algorithm_id}
