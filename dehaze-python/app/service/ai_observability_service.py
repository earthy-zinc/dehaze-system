"""AI 可观测性查询服务（F-M08-013 后端实现 §2.6）

明细查询复用 ai_trace_repository / ai_llm_call_repository 既有能力；
聚合（总览/消耗/趋势）基于 sys_ai_trace 现有字段在 service 层直查
（与 billing/cost_stat_service 等聚合实践一致），不引入额外存储。
"""

import csv
import io
from typing import Any

from fastapi.responses import StreamingResponse
from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.constants import MAX_ROWS
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_llm_call import SysAiLlmCall
from app.models.entity.sys_ai_trace import SysAiTrace
from app.models.schema.ai_conversation import AgentThoughtResult
from app.models.schema.ai_observability import (
    CostItem,
    CostTrendItem,
    CostsQuery,
    CostsResult,
    LlmCallItem,
    SummaryResult,
    TraceArtifactItem,
    TraceBillingItem,
    TraceDetailResult,
    TraceItem,
    TraceMessageItem,
    TracePageQuery,
    TrendItem,
    TrendsQuery,
)
from app.models.schema.common import PageResult
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository
from app.repository.ai_artifact_repository import ai_artifact_repository
from app.repository.ai_billing_repository import ai_billing_repository
from app.repository.ai_llm_call_repository import ai_llm_call_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_trace_repository import ai_trace_repository

# 采集链路实际写入的配额拒绝类 error_type：中断类型 quota（dehaze_hooks_middleware）
# 与计费 stop_reason（billing_service / reasoning_service 计费拒绝收尾路径）
_QUOTA_REJECT_ERROR_TYPES = (
    "quota",
    "quota_exceeded",
    "precharge_blocked",
    "arrears",
    "balance_exceeded",
)
# 高风险调用：推理步数超阈值（防循环观测，见可观测性后端实现.md §2.4）
_HIGH_RISK_STEP_THRESHOLD = 40


def _risky_tool_call_exists() -> Any:
    """高风险工具调用谓词：该过程链存在"发起工具调用但调用失败/超时"的 LLM 调用。

    工具执行异常经恢复中间件兜住转为 ToolMessage，不落到 error_type；
    采集侧工具痕迹在 sys_ai_llm_call.tool_call（仅发起工具调用的轮次非空），
    故以"tool_call 非空且调用未成功"作为工具调用失败的查询侧口径。
    """
    return (
        select(SysAiLlmCall.id)
        .where(
            SysAiLlmCall.trace_id == SysAiTrace.trace_id,
            SysAiLlmCall.tool_call.is_not(None),
            SysAiLlmCall.status != 1,
        )
        .exists()
    )

_COST_METRICS = [
    func.count().label("trace_count"),
    func.coalesce(func.sum(SysAiTrace.total_tokens), 0).label("total_tokens"),
    func.coalesce(func.sum(SysAiTrace.prompt_tokens), 0).label("prompt_tokens"),
    func.coalesce(func.sum(SysAiTrace.completion_tokens), 0).label("completion_tokens"),
    func.coalesce(func.sum(SysAiTrace.cached_tokens), 0).label("cached_tokens"),
]


class AiObservabilityService:
    async def summary(self, db: AsyncSession) -> SummaryResult:
        """异常总览统计：状态分布 + 配额拒绝 + 高风险调用"""
        status_counts = await ai_trace_repository.count_by_status(db)
        quota_rejected = await self._count(
            db, SysAiTrace.error_type.in_(_QUOTA_REJECT_ERROR_TYPES)
        )
        high_risk = await self._count(
            db,
            or_(
                SysAiTrace.step_count >= _HIGH_RISK_STEP_THRESHOLD,
                _risky_tool_call_exists(),
            ),
        )
        return SummaryResult(
            total=sum(status_counts.values()),
            success_count=status_counts.get(1, 0),
            failed_count=status_counts.get(2, 0),
            interrupted_count=status_counts.get(3, 0),
            timeout_count=status_counts.get(4, 0),
            quota_rejected=quota_rejected,
            high_risk_calls=high_risk,
        )

    @staticmethod
    async def _count(db: AsyncSession, condition: Any) -> int:
        stmt = select(func.count()).select_from(SysAiTrace).where(condition)
        return (await db.execute(stmt)).scalar() or 0

    async def list_traces(self, db: AsyncSession, query: TracePageQuery) -> PageResult[TraceItem]:
        """过程链分页检索（会话/用户/状态/智能体/模型/失败类型/关键词/能力维度/时间）"""
        stmt = self._filtered_stmt(query)
        items, total = await ai_trace_repository.paginate(
            db, stmt, query.pageNum, query.pageSize
        )
        return PageResult(list=[TraceItem.model_validate(t) for t in items], total=total)

    @staticmethod
    def _filtered_stmt(query: TracePageQuery) -> Select:
        stmt = select(SysAiTrace)
        if query.conversationId is not None:
            stmt = stmt.where(SysAiTrace.conversation_id == query.conversationId)
        if query.userId is not None or query.keyword is not None:
            # 用户归属与标题关键词共用一次会话表关联，避免重复 join 产生笛卡尔放大
            stmt = stmt.join(
                SysAiConversation, SysAiTrace.conversation_id == SysAiConversation.id
            )
        if query.userId is not None:
            stmt = stmt.where(SysAiConversation.user_id == query.userId, SysAiConversation.deleted == 0)
        if query.status is not None:
            stmt = stmt.where(SysAiTrace.status == query.status)
        if query.agentCode is not None:
            stmt = stmt.where(SysAiTrace.agent_code == query.agentCode)
        if query.model is not None:
            stmt = stmt.where(SysAiTrace.model == query.model)
        if query.errorType is not None:
            stmt = stmt.where(SysAiTrace.error_type == query.errorType)
        if query.keyword is not None:
            pattern = f"%{query.keyword}%"
            stmt = stmt.where(
                or_(SysAiTrace.trace_id.like(pattern), SysAiConversation.title.like(pattern))
            )
        if query.capability is not None:
            # 能力维度：匹配 context_snapshot.items[].type 构成项（kb/tools 待采集侧补写后自然生效）
            stmt = stmt.where(
                func.json_search(
                    SysAiTrace.context_snapshot,
                    "one",
                    query.capability,
                    None,
                    "$.items[*].type",
                ).is_not(None)
            )
        if query.startTime is not None:
            stmt = stmt.where(SysAiTrace.create_time >= query.startTime)
        if query.endTime is not None:
            stmt = stmt.where(SysAiTrace.create_time <= query.endTime)
        return stmt.order_by(SysAiTrace.create_time.desc(), SysAiTrace.id.desc())

    async def get_trace(
        self, db: AsyncSession, trace_id: str, user_id: int, *, admin: bool
    ) -> TraceDetailResult:
        """过程链详情：上下文快照 + LLM 调用回放（按 seq 正序）。

        管理员可查全量；普通用户仅可查自己会话的过程链，
        跨会话访问与不存在一律 404（A0401），不暴露他人过程链存在性。
        """
        trace = await ai_trace_repository.get_by_trace_id(db, trace_id)
        if trace is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "过程链不存在")
        if not admin:
            owner_id = (
                await db.execute(
                    select(SysAiConversation.user_id).where(
                        SysAiConversation.id == trace.conversation_id,
                        SysAiConversation.deleted == 0,
                    )
                )
            ).scalar_one_or_none()
            if owner_id != user_id:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "过程链不存在")
        calls = await ai_llm_call_repository.list_by_trace(db, trace_id)
        detail = TraceDetailResult.model_validate(trace)
        detail.llm_calls = [LlmCallItem.model_validate(c) for c in calls]
        if trace.message_id is not None:
            thoughts = await ai_agent_thought_repository.list_by_message(db, trace.message_id)
            detail.thoughts = [AgentThoughtResult.model_validate(t) for t in thoughts]
            # 计费记录优先按 request_id=trace_id 关联（LLM 调用级精确归因），
            # 无命中时回退 message_id（补记/兼容场景）
            billing_rows = await ai_billing_repository.list_by_request_id(db, trace_id)
            if not billing_rows:
                billing_rows = await ai_billing_repository.list_by_message(db, trace.message_id)
            detail.billing = [TraceBillingItem.model_validate(b) for b in billing_rows]
            artifacts = await ai_artifact_repository.list_by_message(db, trace.message_id)
            detail.artifacts = [TraceArtifactItem.model_validate(a) for a in artifacts]
        messages, _ = await ai_message_repository.list_by_conversation(
            db, trace.conversation_id, 1, 1000
        )
        detail.messages = [TraceMessageItem.model_validate(m) for m in messages]
        return detail

    async def costs(self, db: AsyncSession, query: CostsQuery) -> CostsResult:
        """资源消耗聚合：按模型/智能体/用户维度分页聚合 + 按日Token趋势（与计费口径一致）"""
        stmt = self._cost_stmt(query.dimension, query)
        rows, total = await ai_trace_repository.paginate_rows(
            db, stmt, query.pageNum, query.pageSize
        )
        items = [CostItem(**row) for row in rows]

        day = func.date(SysAiTrace.create_time).label("date")
        trend_rows = (await db.execute(self._base_cost_stmt(query, [day]).group_by(day).order_by(day))).all()
        trend = [
            CostTrendItem(
                date=str(row.date),
                trace_count=row.trace_count,
                total_tokens=row.total_tokens,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                cached_tokens=row.cached_tokens,
            )
            for row in trend_rows
        ]
        return CostsResult(items=items, total=total, trend=trend)

    def _cost_stmt(self, dimension: str, query: CostsQuery) -> Select:
        if dimension == "model":
            dim_col = SysAiTrace.model.label("model")
        elif dimension == "agent":
            dim_col = SysAiTrace.agent_code.label("agent_code")
        else:
            dim_col = SysAiConversation.user_id.label("user_id")
        return self._base_cost_stmt(query, [dim_col]).group_by(dim_col)

    @staticmethod
    def _base_cost_stmt(query: CostsQuery, columns: list) -> Select:
        stmt = select(*columns, *_COST_METRICS)
        if query.dimension == "user":
            stmt = stmt.join(
                SysAiConversation, SysAiTrace.conversation_id == SysAiConversation.id
            )
        if query.startTime is not None:
            stmt = stmt.where(SysAiTrace.create_time >= query.startTime)
        if query.endTime is not None:
            stmt = stmt.where(SysAiTrace.create_time <= query.endTime)
        return stmt

    async def trends(self, db: AsyncSession, query: TrendsQuery) -> list[TrendItem]:
        """性能趋势：按维度+日期聚合调用量/成功率/平均延迟（首Token延迟取成功调用口径）"""
        if query.dimension == "model":
            dim_col = SysAiTrace.model.label("dimension")
        else:
            dim_col = SysAiTrace.agent_code.label("dimension")
        day = func.date(SysAiTrace.create_time).label("date")
        stmt = (
            select(
                dim_col,
                day,
                func.count().label("call_count"),
                func.sum(case((SysAiTrace.status == 1, 1), else_=0)).label("success_count"),
                func.avg(SysAiTrace.first_token_ms).label("avg_first_token_ms"),
                func.avg(SysAiTrace.duration_ms).label("avg_duration_ms"),
            )
            .group_by(dim_col, day)
            .order_by(day, dim_col)
        )
        if query.startTime is not None:
            stmt = stmt.where(SysAiTrace.create_time >= query.startTime)
        if query.endTime is not None:
            stmt = stmt.where(SysAiTrace.create_time <= query.endTime)

        rows = (await db.execute(stmt)).all()
        items: list[TrendItem] = []
        for row in rows:
            success_rate = (
                round(row.success_count / row.call_count * 100, 2) if row.call_count else 0.0
            )
            items.append(
                TrendItem(
                    model=row.dimension if query.dimension == "model" else None,
                    agent_code=row.dimension if query.dimension == "agent" else None,
                    date=str(row.date),
                    call_count=row.call_count,
                    success_count=row.success_count,
                    success_rate=success_rate,
                    avg_first_token_ms=round(float(row.avg_first_token_ms), 2)
                    if row.avg_first_token_ms is not None
                    else None,
                    avg_duration_ms=round(float(row.avg_duration_ms), 2)
                    if row.avg_duration_ms is not None
                    else None,
                )
            )
        return items

    async def export_traces(self, db: AsyncSession, query: TracePageQuery) -> StreamingResponse:
        """过程链导出（CSV，UTF-8 BOM 便于 Excel 打开），按检索条件全量导出并限行数"""
        stmt = self._filtered_stmt(query)
        count = await ai_trace_repository.count(db, stmt)
        if count > MAX_ROWS:
            raise BusinessException(
                ResultCode.EXPORT_ROWS_EXCEED_LIMIT,
                f"导出行数 {count} 超出限制 {MAX_ROWS}",
            )
        traces = list((await db.execute(stmt)).scalars().all())

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "trace_id",
                "conversation_id",
                "message_id",
                "agent_code",
                "model",
                "status",
                "error_type",
                "duration_ms",
                "first_token_ms",
                "llm_call_count",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "step_count",
                "create_time",
            ]
        )
        for t in traces:
            writer.writerow(
                [
                    t.trace_id,
                    t.conversation_id,
                    t.message_id,
                    t.agent_code,
                    t.model,
                    t.status,
                    t.error_type,
                    t.duration_ms,
                    t.first_token_ms,
                    t.llm_call_count,
                    t.total_tokens,
                    t.prompt_tokens,
                    t.completion_tokens,
                    t.cached_tokens,
                    t.step_count,
                    t.create_time.isoformat(sep=" ") if t.create_time else "",
                ]
            )
        # BOM 头：Excel 识别 UTF-8 CSV 中文字段
        payload = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
        return StreamingResponse(
            iter([payload]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ai_traces.csv"'},
        )


ai_observability_service = AiObservabilityService()
