"""
对比报告服务 —— 生成/下载去雾效果对比报告（异步任务模式）

复用 sys_eval_log 表存储报告任务，reportHtml 写入 result JSON 字段。
POST 立即返回 taskId + status=processing，asyncio.create_task 后台生成 HTML。
"""

import asyncio
import json
import logging
from datetime import datetime

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.models.base import set_current_user_id
from app.models.entity.sys_log import SysEvalLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository, pred_log_repository
from app.service.prediction_service import prediction_service

logger = logging.getLogger(__name__)


class CompareService:
    """对比报告服务"""

    async def generate_report(
        self,
        log_id: int,
        user_id: int,
    ) -> dict:
        """
        提交对比报告生成任务（异步）

        根据 pred_log_id 查询预测日志，用其中的 algorithmId/originUrl/predUrl 生成报告。

        Returns:
            {"taskId": int, "status": int}
        """
        async with get_db_session() as db:
            pred_log = await pred_log_repository.get_by_id(db, log_id)
        if not pred_log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "处理记录不存在")
        if pred_log.status != LogStatus.COMPLETED.value:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "处理任务尚未完成，无法生成报告")

        algorithm_id = pred_log.algorithm_id
        origin_url = pred_log.origin_url
        result_url = pred_log.pred_url

        # 校验算法存在
        await prediction_service.get_algorithm(algorithm_id)

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = SysEvalLog(
                    algorithm_id=algorithm_id,
                    pred_url=result_url,
                    gt_url=origin_url,
                    pred_md5="",
                    gt_md5="",
                    time=0,
                    status=LogStatus.PROCESSING.value,
                )
                db.add(log)
                await db.flush()
                await db.refresh(log)
                task_id = log.id
        finally:
            set_current_user_id(None)

        # 提交异步任务
        loop = asyncio.get_running_loop()
        background_task = loop.create_task(
            self._generate_async(
                task_id=task_id,
                algorithm_id=algorithm_id,
                origin_url=origin_url,
                result_url=result_url,
                user_id=user_id,
            )
        )

        # 注册到 TaskTracker，支持优雅关闭与全局任务视图
        try:
            from app.service.task_tracker import get_task_tracker

            await get_task_tracker().register(
                task_id=f"compare:{task_id}",
                task=background_task,
                task_type="compare",
                metadata={"task_id": task_id, "algorithm_id": algorithm_id, "user_id": user_id},
            )
        except Exception as e:
            logger.warning("对比报告任务追踪注册失败（不影响执行）: %s", e)

        return {"taskId": task_id, "status": LogStatus.PROCESSING.value}

    async def get_report_status(self, task_id: int) -> dict:
        """查询报告任务状态"""
        async with get_db_session() as db:
            log = await eval_log_repository.get_by_id(db, task_id)
            if not log:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "报告不存在")

            resp = {
                "taskId": log.id,
                "status": log.status,
            }
            if log.status == LogStatus.COMPLETED.value:
                resp["downloadUrl"] = f"/api/v1/compare/report/{log.id}?download=true"
            elif log.status == LogStatus.FAILED.value:
                resp["errorMessage"] = log.error_message
            return resp

    async def get_report_html(self, task_id: int) -> str:
        """获取已完成报告的 HTML 内容（用于文件流下载）"""
        async with get_db_session() as db:
            log = await eval_log_repository.get_by_id(db, task_id)
            if not log:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "报告不存在")
            if log.status == LogStatus.PROCESSING.value:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "报告尚未生成完成")
            if log.status == LogStatus.FAILED.value:
                raise BusinessException(
                    ResultCode.SYSTEM_EXECUTION_ERROR,
                    f"报告生成失败：{log.error_message or '未知错误'}",
                )
            if not log.result:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "报告内容为空")

            if isinstance(log.result, dict):
                report_html = log.result.get("reportHtml", "")
            elif isinstance(log.result, str):
                try:
                    result_dict = json.loads(log.result)
                    report_html = result_dict.get("reportHtml", "")
                except json.JSONDecodeError:
                    report_html = ""
            else:
                report_html = ""

            if not report_html:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "报告内容为空")
            return report_html

    async def _generate_async(
        self,
        task_id: int,
        algorithm_id: int,
        origin_url: str,
        result_url: str,
        user_id: int,
    ) -> None:
        """异步生成 HTML 报告"""
        set_current_user_id(user_id)
        try:
            # 获取算法名称
            algorithm_name = "未知算法"
            try:
                algo = await prediction_service.get_algorithm(algorithm_id)
                algorithm_name = algo.name or "未知算法"
            except Exception:
                pass

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html = self._build_report_html(
                algorithm_name=algorithm_name,
                generated_at=now_str,
                origin_url=origin_url,
                result_url=result_url,
                algorithm_id=algorithm_id,
                metrics_html="",
            )

            result_json = json.dumps(
                {
                    "reportHtml": html,
                    "generatedAt": now_str,
                },
                ensure_ascii=False,
            )

            async with get_db_session() as db:
                await eval_log_repository.update_result(
                    db=db,
                    log_id=task_id,
                    result=result_json,
                    time_ms=0,
                )

            logger.info("对比报告生成完成: taskId=%s", task_id)

        except Exception as e:
            error_msg = str(e)
            logger.error("对比报告生成失败: taskId=%s, error=%s", task_id, error_msg, exc_info=True)
            try:
                async with get_db_session() as db:
                    await eval_log_repository.update_status(
                        db=db,
                        log_id=task_id,
                        status=LogStatus.FAILED.value,
                        error_message=error_msg,
                        time_ms=0,
                    )
            except Exception as update_err:
                logger.error("更新报告失败状态失败: taskId=%s, error=%s", task_id, update_err)
        finally:
            set_current_user_id(None)

    @staticmethod
    def _build_report_html(
        algorithm_name: str,
        generated_at: str,
        origin_url: str,
        result_url: str,
        algorithm_id: int,
        metrics_html: str = "",
    ) -> str:
        """构建对比报告 HTML"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>去雾效果对比报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5; color: #333; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; padding: 30px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .meta {{ font-size: 14px; opacity: 0.85; }}
        .section {{ padding: 24px 30px; border-bottom: 1px solid #eee; }}
        .section:last-child {{ border-bottom: none; }}
        .section h2 {{ font-size: 18px; color: #667eea; margin-bottom: 16px; }}
        .comparison {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .image-card {{ flex: 1; min-width: 280px; }}
        .image-card .label {{ font-size: 14px; color: #666; margin-bottom: 8px; font-weight: 500; }}
        .image-card img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e0e0; }}
        .info-grid {{ display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
        .info-item {{ background: #f8f9ff; padding: 12px 16px; border-radius: 6px; }}
        .info-item .label {{ font-size: 12px; color: #999; margin-bottom: 4px; }}
        .info-item .value {{ font-size: 16px; font-weight: 500; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>去雾效果对比报告</h1>
            <div class="meta">算法：{algorithm_name} | 生成时间：{generated_at}</div>
        </div>
        <div class="section">
            <h2>图片对比</h2>
            <div class="comparison">
                <div class="image-card">
                    <div class="label">原图</div>
                    <img src="{origin_url}" alt="原图" onerror="this.style.display='none'" />
                </div>
                <div class="image-card">
                    <div class="label">处理结果</div>
                    <img src="{result_url}" alt="处理结果" onerror="this.style.display='none'" />
                </div>
            </div>
        </div>
        <div class="section">
            <h2>处理信息</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="label">算法名称</div>
                    <div class="value">{algorithm_name}</div>
                </div>
                <div class="info-item">
                    <div class="label">算法ID</div>
                    <div class="value">{algorithm_id}</div>
                </div>
                <div class="info-item">
                    <div class="label">生成时间</div>
                    <div class="value">{generated_at}</div>
                </div>
                <div class="info-item">
                    <div class="label">任务状态</div>
                    <div class="value">已完成</div>
                </div>
            </div>
        </div>
        {metrics_html}
        <div class="footer">
            本报告由 Dehaze 系统自动生成
        </div>
    </div>
</body>
</html>"""


compare_service = CompareService()
