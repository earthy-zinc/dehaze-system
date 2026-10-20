"""AI 可观测性查询服务（F-M08-013 后端实现 §2.6）

明细查询复用 ai_trace_repository / ai_llm_call_repository 既有能力；
聚合（总览/消耗/趋势）基于 sys_ai_trace 现有字段在 service 层直查
（与 billing/cost_stat_service 等聚合实践一致），不引入额外存储。
"""

import csv
import io
from datetime import datetime, timedelta
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
    CostsQuery,
    CostsResult,
    CostTrendItem,
    LlmCallItem,
    SummaryResult,
    TimelineConversation,
    TimelineEvent,
    TimelineMessage,
    TimelineResult,
    TimelineRound,
    TimelineTrace,
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
from app.repository.ai_conversation_repository import ai_conversation_repository
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


# 事件交织排序的同类事件优先级（同刻事件按业务序，见审计级重构设计 §4.2）
_EVENT_PRIORITY = {
    "input": 0,
    "context": 1,
    "system_event": 2,
    "llm_call": 3,
    "tool_exec": 4,
    "billing": 5,
}


def _slim_input_snapshot(snapshot: dict | None) -> dict | None:
    """输入快照瘦身：保留按角色计数/token 估算/工具数/用户/系统提示 token 数，
    去掉 messages.items 全文、system_content 与 tools 定义清单（全文唯一通道走 rawRequest）"""
    if snapshot is None:
        return None
    slim: dict = {}
    if isinstance(snapshot.get("messages"), dict):
        slim["messages"] = {k: v for k, v in snapshot["messages"].items() if k != "items"}
    for key in ("system_tokens", "tool_count", "user_id"):
        if key in snapshot:
            slim[key] = snapshot[key]
    return slim


class AiObservabilityService:
    async def summary(self, db: AsyncSession) -> SummaryResult:
        """异常总览统计：状态分布 + 配额拒绝 + 高风险调用"""
        status_counts = await ai_trace_repository.count_by_status(db)
        quota_rejected = await self._count(db, SysAiTrace.error_type.in_(_QUOTA_REJECT_ERROR_TYPES))
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
        items, total = await ai_trace_repository.paginate(db, stmt, query.pageNum, query.pageSize)
        results = [TraceItem.model_validate(t) for t in items]
        if results:
            # 会话标题批量回填（检索行透出会话归属，供前端跳转会话时间线）
            rows = (
                await db.execute(
                    select(SysAiConversation.id, SysAiConversation.title).where(
                        SysAiConversation.id.in_({t.conversation_id for t in items})
                    )
                )
            ).all()
            titles = {r[0]: r[1] for r in rows}
            for item in results:
                item.conversation_title = titles.get(item.conversation_id)
        return PageResult(list=results, total=total)

    @staticmethod
    def _filtered_stmt(query: TracePageQuery) -> Select:
        stmt = select(SysAiTrace)
        if query.conversationId is not None:
            stmt = stmt.where(SysAiTrace.conversation_id == query.conversationId)
        if query.userId is not None or query.keyword is not None:
            # 用户归属与标题关键词共用一次会话表关联，避免重复 join 产生笛卡尔放大
            stmt = stmt.join(SysAiConversation, SysAiTrace.conversation_id == SysAiConversation.id)
        if query.userId is not None:
            stmt = stmt.where(
                SysAiConversation.user_id == query.userId, SysAiConversation.deleted == 0
            )
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

    async def get_conversation_timeline(
        self,
        db: AsyncSession,
        conversation_id: int,
        user_id: int,
        *,
        admin: bool,
        include_raw: bool = True,
    ) -> TimelineResult:
        """会话审计时间线：轮次（消息分支链 user→assistant 配对）+ 轮内事件按 ts 交织。

        管理员可查全量；普通用户仅可查自己会话，跨会话/不存在一律 404（A0401），
        不暴露存在性。include_raw=False 省略 raw 原始报文供轻量预览。
        """
        if admin:
            conv = await ai_conversation_repository.get_by_id(db, conversation_id)
        else:
            conv = await ai_conversation_repository.get_by_id_and_user(db, conversation_id, user_id)
        if conv is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        # 沿当前激活分支链取全量消息（分支对话时间序与链序可能不一致，以链为准）
        chain_start = conv.current_branch_message_id
        if chain_start is None:
            chain_start = await ai_message_repository.get_last_message_id(db, conversation_id)
        chain = await ai_message_repository.get_chain_by_id(db, conversation_id, chain_start)

        rounds = self._split_rounds(chain)
        traces = await ai_trace_repository.list_by_conversation(db, conversation_id)
        self._attach_traces(rounds, traces)

        calls_map = await ai_llm_call_repository.list_by_traces(db, [t.trace_id for t in traces])
        assistant_ids = [r["assistant"].id for r in rounds if r["assistant"]]
        thoughts_map = await ai_agent_thought_repository.list_by_messages(db, assistant_ids)
        # 计费按 request_id=trace_id 精确归因，无 request_id 的按 message_id
        # 挂到该消息的主对话 trace（与 get_trace 详情的关联口径一致）
        billing_by_request: dict[str, list] = {}
        billing_by_message: dict[int, list] = {}
        for b in await ai_billing_repository.list_by_conversation(db, conversation_id):
            if b.request_id:
                billing_by_request.setdefault(b.request_id, []).append(b)
            elif b.message_id is not None:
                # 无 request_id 且无 message_id 的账单无法归因到任何轮次，跳过
                billing_by_message.setdefault(b.message_id, []).append(b)

        timeline_rounds: list[TimelineRound] = []
        for r in rounds:
            timeline_traces: list[TimelineTrace] = []
            for idx, trace in enumerate(r["traces"]):
                is_primary = idx == 0 and trace.trace_type == "conversation"
                billing = billing_by_request.get(trace.trace_id)
                if not billing and is_primary and r["assistant"]:
                    billing = billing_by_message.get(r["assistant"].id, [])
                events = self._trace_events(
                    trace,
                    calls_map.get(trace.trace_id, []),
                    thoughts_map.get(r["assistant"].id, [])
                    if is_primary and r["assistant"]
                    else [],
                    billing or [],
                    r["user"] if is_primary else None,
                    include_raw=include_raw,
                )
                timeline_traces.append(
                    TimelineTrace(
                        trace_id=trace.trace_id,
                        trace_type=trace.trace_type,
                        status=trace.status,
                        error_type=trace.error_type,
                        error_detail=trace.error_detail,
                        model=trace.model,
                        duration_ms=trace.duration_ms,
                        create_time=trace.create_time,
                        events=events,
                    )
                )
            timeline_rounds.append(
                TimelineRound(
                    user_message=TimelineMessage.model_validate(r["user"]) if r["user"] else None,
                    assistant_message=TimelineMessage.model_validate(r["assistant"])
                    if r["assistant"]
                    else None,
                    traces=timeline_traces,
                )
            )
        return TimelineResult(
            conversation=TimelineConversation.model_validate(conv),
            rounds=timeline_rounds,
        )

    @staticmethod
    def _split_rounds(chain: list) -> list[dict]:
        """消息链切轮次：user 开新轮，assistant 归属当前轮（链首 assistant 单独成轮）"""
        rounds: list[dict] = []
        for msg in chain:
            if msg.role == "user" or not rounds:
                rounds.append(
                    {"user": msg if msg.role == "user" else None, "assistant": None, "traces": []}
                )
            if msg.role == "assistant":
                rounds[-1]["assistant"] = msg
        return rounds

    @staticmethod
    def _attach_traces(rounds: list[dict], traces: list) -> None:
        """trace 归属轮次：有 message_id 按消息配对（resume 多 trace 并列同一轮）；
        无 message_id（summary/memory_extraction 旁路）挂触发时点（create_time）最新轮次；
        主对话 trace 在前、旁路在后，旁路不混入主调用序列"""
        for trace in traces:
            target = None
            if trace.message_id is not None:
                for r in rounds:
                    matched = {m.id for m in (r["user"], r["assistant"]) if m is not None}
                    if trace.message_id in matched:
                        target = r
                        break
            if target is None:
                # 轮次按触发时间正序，取 create_time 之前（含）的最近一轮
                for r in rounds:
                    trigger = (r["user"] or r["assistant"]).create_time
                    if trigger <= trace.create_time:
                        target = r
                    else:
                        break
                target = target or (rounds[0] if rounds else None)
            if target is not None:
                target["traces"].append(trace)
        for r in rounds:
            r["traces"].sort(
                key=lambda t: (
                    t.trace_type != "conversation",
                    t.create_time,
                    t.id,
                )
            )

    @staticmethod
    def _trace_events(
        trace,
        calls: list,
        thoughts: list,
        billing_rows: list,
        user_message,
        *,
        include_raw: bool,
    ) -> list[TimelineEvent]:
        """织入单条 trace 的事件流：input/context/system_event/llm_call/tool_exec/billing。

        ts 锚点：llm_call 用 start_time（无值为 NULL，排序沉底），thought/billing 用
        create_time，context/system_event 用 trace.create_time-duration 近似（trace 表
        无精确起始时刻字段，此为既有锚点口径）；同刻事件按业务序（llm_call.seq/
        thought.position）稳定排序，ts 为 NULL 的事件沉底。
        """
        events: list[TimelineEvent] = []
        approx_start = trace.create_time - timedelta(milliseconds=trace.duration_ms)
        snapshot = trace.context_snapshot or {}
        events.append(
            TimelineEvent(kind="context", ts=approx_start, snapshot=trace.context_snapshot)
        )
        events.extend(
            TimelineEvent(
                kind="system_event", ts=approx_start, event=item.get("event"), detail=item
            )
            for item in snapshot.get("events", [])
        )
        for call in calls:
            item = LlmCallItem.model_validate(call)
            events.append(
                TimelineEvent(
                    kind="llm_call",
                    ts=item.start_time,
                    seq=item.seq,
                    model=item.model,
                    status=item.status,
                    duration_ms=item.duration_ms,
                    first_token_ms=item.first_token_ms,
                    prompt_tokens=item.prompt_tokens,
                    completion_tokens=item.completion_tokens,
                    cached_tokens=item.cached_tokens,
                    tool_call=item.tool_call,
                    attempts=item.attempts,
                    raw_request=item.raw_request if include_raw else None,
                    raw_response=item.raw_response if include_raw else None,
                    # summary 恒为纯计数单形状（消息/工具全文唯一通道走 rawRequest），
                    # raw 为 NULL 即无原始报文，无全文兜底形状
                    summary={
                        "inputSnapshot": _slim_input_snapshot(item.input_snapshot),
                        "outputSnapshot": item.output_snapshot,
                    },
                )
            )
        events.extend(
            TimelineEvent(
                kind="tool_exec",
                ts=t.create_time,
                position=t.position,
                tool=t.tool,
                thought=t.thought,
                tool_input=t.tool_input,
                observation=t.observation,
                status=t.status,
                latency_ms=t.latency_ms,
                agent_code=t.agent_code,
                is_subagent=t.is_subagent,
            )
            for t in thoughts
        )
        events.extend(
            TimelineEvent(
                kind="billing",
                ts=b.create_time,
                bill_type=b.bill_type,
                credits=b.credits,
                tokens={
                    "input": b.input_tokens,
                    "output": b.output_tokens,
                    "cached": b.cached_input_tokens,
                },
            )
            for b in billing_rows
        )
        if user_message is not None:
            events.append(
                TimelineEvent(
                    kind="input",
                    ts=user_message.create_time,
                    message=TimelineMessage.model_validate(user_message),
                )
            )
        events.sort(
            key=lambda e: (
                e.ts is None,  # ts 缺失（start_time 为 NULL 的调用）沉底
                e.ts or datetime.min,
                _EVENT_PRIORITY[e.kind],
                e.seq if e.seq is not None else 0,
                e.position if e.position is not None else 0,
            )
        )
        return events

    async def export_timeline(
        self, db: AsyncSession, conversation_id: int, user_id: int
    ) -> StreamingResponse:
        """会话时间线整体导出（JSON 全量含 raw 原始报文），复用管理端导出权限口径"""
        result = await self.get_conversation_timeline(db, conversation_id, user_id, admin=True)
        payload = result.model_dump_json(by_alias=True).encode("utf-8")
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="conversation_{conversation_id}_timeline.json"'
                )
            },
        )

    async def costs(self, db: AsyncSession, query: CostsQuery) -> CostsResult:
        """资源消耗聚合：按模型/智能体/用户维度分页聚合 + 按日Token趋势（与计费口径一致）"""
        stmt = self._cost_stmt(query.dimension, query)
        rows, total = await ai_trace_repository.paginate_rows(
            db, stmt, query.pageNum, query.pageSize
        )
        items = [CostItem(**row) for row in rows]

        day = func.date(SysAiTrace.create_time).label("date")
        trend_rows = (
            await db.execute(self._base_cost_stmt(query, [day]).group_by(day).order_by(day))
        ).all()
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
            stmt = stmt.join(SysAiConversation, SysAiTrace.conversation_id == SysAiConversation.id)
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
                # 首 Token 延迟取成功调用口径：失败/中断/超时链路即便收到过首 Token 也不计入
                func.avg(case((SysAiTrace.status == 1, SysAiTrace.first_token_ms))).label(
                    "avg_first_token_ms"
                ),
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
