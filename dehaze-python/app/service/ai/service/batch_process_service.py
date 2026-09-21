"""批量处理调度服务：任务分解 → 逐张处理 → 进度推送 → 结果摘要

用户上传多张图片并描述统一处理需求时，将每张图片作为一个子任务，
按配置策略（serial/parallel/auto）调度预测服务（异步任务模式，立即返回 logId），
注册产物并推送结构化进度，全部完成后生成结果摘要。
"""

import asyncio
import logging

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.repository.file_repository import file_repository
from app.service.ai_artifact_service import ai_artifact_service
from app.service.prediction.prediction_service import prediction_service

logger = logging.getLogger(__name__)

# 批量处理并发与规模上限：纯技术参数，随版本调整，不提供运行时配置
BATCH_MAX_PARALLEL = 5
BATCH_STRATEGY = "auto"
BATCH_MAX_IMAGES = 20
# 异步阈值：超过该数量走 async_wait 中断（提交后台任务 + 回调自动恢复），
# 不超过保持同步直返，避免轻任务中断体验劣化
BATCH_ASYNC_THRESHOLD = 3


def _system_object_name(image_url: str) -> str | None:
    """把本系统存储 URL 还原为 object_name，非本系统地址返回 None。"""
    for base in settings.FILE_STORAGE_BASE_URLS.values():
        prefix = base.rstrip("/") + "/"
        if image_url.startswith(prefix):
            return image_url[len(prefix) :]
    return None


async def validate_owned_image_url(db, user_id: int, image_url: str) -> None:
    """校验 image_url 是本系统、且归属当前用户的图片地址。

    地址由 LLM 经工具入参给出，预测侧会据此下载图片——放行任意地址即为 SSRF
    （内网探测）与越权处理他人图片的入口，故只认本系统存储地址且 sys_file 归属本人。
    """
    object_name = _system_object_name(image_url)
    if not object_name:
        raise BusinessException(ResultCode.PARAM_ERROR, "图片地址必须是本系统产物地址")
    file_info = await file_repository.get_by_object_name(db, object_name)
    if not file_info or file_info.deleted != 0 or file_info.create_by != user_id:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图片不存在或不属于当前用户")


def _resolve_strategy(total: int) -> str:
    """解析批量调度策略：serial / parallel / auto（图片数 <= 3 串行，否则并行）"""
    strategy = BATCH_STRATEGY
    if strategy == "auto":
        return "serial" if total <= 3 else "parallel"
    return strategy


async def process_batch(
    conv_id: int,
    msg_id: int,
    user_id: int,
    image_urls: list[str],
    algorithm_id: int,
    stream_session_id: str,
) -> dict:
    """批量处理调度：按策略调度预测 → 注册产物 → 推送进度 → 生成摘要

    预测服务为异步任务模式（立即返回 logId），产物先注册、结果由后台任务完成，
    用户通过产物/进度事件感知处理进度。评估需参考图且依赖预测完成，暂不执行。
    """
    image_urls = image_urls[:BATCH_MAX_IMAGES]
    total = len(image_urls)

    if _resolve_strategy(total) == "parallel":
        results = await _run_parallel(
            conv_id,
            msg_id,
            user_id,
            image_urls,
            algorithm_id,
            stream_session_id,
        )
    else:
        results = []
        for index, image_url in enumerate(image_urls):
            results.append(
                await _process_single(
                    conv_id,
                    msg_id,
                    user_id,
                    image_url,
                    algorithm_id,
                    index,
                    total,
                    stream_session_id,
                )
            )

    summary = _build_summary(total, results)
    await sse_emitter_manager.send_event(
        stream_session_id,
        "content_block.delta",
        {"index": total, "total": total, "status": "summary", "summary": summary},
    )
    return summary


async def _run_parallel(
    conv_id: int,
    msg_id: int,
    user_id: int,
    image_urls: list[str],
    algorithm_id: int,
    stream_session_id: str,
) -> list[dict]:
    """并行调度：用信号量控制并发数，asyncio.gather 收集结果"""
    semaphore = asyncio.Semaphore(BATCH_MAX_PARALLEL)
    total = len(image_urls)

    async def _wrapped(index: int, image_url: str) -> dict:
        async with semaphore:
            return await _process_single(
                conv_id,
                msg_id,
                user_id,
                image_url,
                algorithm_id,
                index,
                total,
                stream_session_id,
            )

    return await asyncio.gather(*(_wrapped(i, u) for i, u in enumerate(image_urls)))


async def _process_single(
    conv_id: int,
    msg_id: int,
    user_id: int,
    image_url: str,
    algorithm_id: int,
    index: int,
    total: int,
    stream_session_id: str,
) -> dict:
    """处理单张图片：校验图片归属 → 调用预测服务 → 注册 artifact → 评估 → 推送进度"""
    try:
        async with get_db_session() as db:
            await validate_owned_image_url(db, user_id, image_url)
        pred_result = await prediction_service.predict(
            algorithm_id=algorithm_id,
            image_url=image_url,
            user_id=user_id,
        )
        pred_log_id = pred_result.get("logId")

        async with get_db_session() as db:
            artifact = await ai_artifact_service.register_artifact(
                db,
                conv_id,
                msg_id,
                artifact_type="image_result",
                ref_type="sys_pred_log",
                ref_id=pred_log_id,
                summary={"algorithmId": algorithm_id},
            )

        # 评估环节：metric_report 由 evaluation_service.evaluate 异步评估完成后注册
        # （ref_type=sys_eval_log，summary=评估指标）；批量流程缺少参考图（gt_url），
        # 待调用方提供参考图并传入 conv_id/msg_id 时触发，暂不执行且不阻断主流程。
        metrics = None
        metric_artifact_id = None

        await sse_emitter_manager.send_event(
            stream_session_id,
            "content_block.delta",
            {
                "index": index,
                "total": total,
                "imageId": image_url,
                "status": "success",
                "artifactId": artifact.id,
            },
        )
        return {
            "index": index,
            "image_id": image_url,
            "status": "success",
            "pred_log_id": pred_log_id,
            "metrics": metrics,
            "metric_artifact_id": metric_artifact_id,
            "artifacts": [artifact.id],
        }
    except Exception as e:
        logger.warning("Batch process failed for image %d: %s", index, e)
        await sse_emitter_manager.send_event(
            stream_session_id,
            "content_block.delta",
            {
                "index": index,
                "total": total,
                "imageId": image_url,
                "status": "failed",
                "artifactId": None,
            },
        )
        return {
            "index": index,
            "image_id": image_url,
            "status": "failed",
            "error": str(e),
            "pred_log_id": None,
            "metrics": None,
            "metric_artifact_id": None,
            "artifacts": [],
        }


def _build_summary(total: int, results: list[dict]) -> dict:
    """生成批量结果摘要：平均指标、最佳/最差、失败列表"""
    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    # 平均指标（仅统计有指标的图片）
    metric_values = {"psnr": [], "ssim": []}
    for r in success:
        metrics = r.get("metrics") or {}
        for key in metric_values:
            value = metrics.get(key)
            if value is not None:
                metric_values[key].append(value)
    avg_metrics = {
        key: (sum(values) / len(values)) if values else None
        for key, values in metric_values.items()
    }

    # 最佳/最差（按 PSNR 排序）
    ranked = [r for r in success if (r.get("metrics") or {}).get("psnr") is not None]
    ranked.sort(key=lambda r: r["metrics"]["psnr"], reverse=True)
    best = {"imageId": ranked[0]["image_id"], "metrics": ranked[0]["metrics"]} if ranked else None
    worst = (
        {"imageId": ranked[-1]["image_id"], "metrics": ranked[-1]["metrics"]} if ranked else None
    )

    failures = [{"imageId": r["image_id"], "reason": r.get("error")} for r in failed]

    return {
        "total": total,
        "success": len(success),
        "failed": len(failed),
        "avg_metrics": avg_metrics,
        "best": best,
        "worst": worst,
        "failures": failures,
        "results": results,
    }
