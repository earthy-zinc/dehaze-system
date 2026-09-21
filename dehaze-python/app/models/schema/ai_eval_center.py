"""评测中心 Schema 模型（F-M08-014 跨 Agent 聚合接口）

覆盖评测总览 / 历史趋势 / run 对比 / 判分状态 / 人工复核的请求与响应模型。
字段名与 ORM/dict 一致（snake_case），序列化输出 camelCase。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.schema.common import OrmResult


class EvalAgentOverviewItem(OrmResult):
    agent_id: int = Field(description="Agent ID")
    agent_code: str = Field(description="Agent 编码")
    agent_name: str = Field(description="Agent 名称")
    run_id: int | None = Field(default=None, description="最近一次已完成评测记录ID(未评测为null)")
    run_time: datetime | None = Field(default=None, description="最近一次评测时间")
    trigger_type: str | None = Field(default=None, description="触发方式(manual/publish)")
    gate_status: str = Field(description="门禁状态(passed:通过;failed:未通过;none:未评测)")
    total_score: float | None = Field(default=None, description="四维总分(均值,0-100)")
    dimensions: dict[str, float] | None = Field(default=None, description="四维得分")
    degraded: bool = Field(description="相对上次评测是否退化(超过 ai_eval.regression_threshold)")
    high_risk_failed: bool = Field(description="最近评测是否存在高风险样本失败")


class EvalTrendItem(OrmResult):
    run_id: int = Field(description="评测记录ID")
    agent_id: int = Field(description="Agent ID")
    agent_name: str = Field(description="Agent 名称")
    trigger_type: str = Field(description="触发方式(manual/publish)")
    status: int = Field(description="执行状态(2:通过;3:失败)")
    total_score: float | None = Field(default=None, description="四维总分(均值,0-100)")
    dimensions: dict[str, float] | None = Field(default=None, description="四维得分")
    create_time: datetime | None = Field(default=None, description="评测时间")


class EvalRunScoreSnapshot(OrmResult):
    run_id: int = Field(description="评测记录ID")
    total_score: float | None = Field(default=None, description="四维总分(均值,0-100)")
    dimensions: dict[str, float] | None = Field(default=None, description="四维得分")
    sample_count: int = Field(description="样本数")
    pass_rate: float | None = Field(default=None, description="样本通过率(0-1)")
    create_time: datetime | None = Field(default=None, description="评测时间")


class EvalSampleDiffItem(OrmResult):
    sample_id: int = Field(description="样本ID")
    task_goal: str = Field(description="任务目标")
    current_passed: bool | None = Field(default=None, description="本次评测通过状态")
    base_passed: bool | None = Field(default=None, description="基准评测通过状态")
    current_score: float | None = Field(default=None, description="本次样本总分")
    base_score: float | None = Field(default=None, description="基准样本总分")
    score_delta: float | None = Field(default=None, description="总分变化(本次-基准)")


class EvalSampleDiff(OrmResult):
    added: list[EvalSampleDiffItem] = Field(
        default_factory=list, description="仅本次评测包含的样本"
    )
    removed: list[EvalSampleDiffItem] = Field(
        default_factory=list, description="仅基准评测包含的样本"
    )
    changed: list[EvalSampleDiffItem] = Field(
        default_factory=list, description="两次均包含但得分/通过状态有差异的样本"
    )
    unchanged_count: int = Field(default=0, description="两次得分与状态均一致的样本数")


class EvalRunCompareResult(OrmResult):
    run_id: int = Field(description="本次评测记录ID")
    base_run_id: int = Field(description="基准评测记录ID")
    agent_id: int = Field(description="Agent ID")
    current: EvalRunScoreSnapshot = Field(description="本次评测得分快照")
    base: EvalRunScoreSnapshot = Field(description="基准评测得分快照")
    dimension_diff: dict[str, float] = Field(description="四维得分差(本次-基准)")
    sample_diff: EvalSampleDiff = Field(description="样本级差异")


class JudgeReviewStats(OrmResult):
    total: int = Field(description="复核项总数")
    pending: int = Field(description="待复核数")
    reviewed: int = Field(description="已复核数")
    agree_count: int = Field(description="人工判定与判分一致数")
    disagree_count: int = Field(description="人工判定与判分不一致数")
    agreement_rate: float = Field(description="判分一致率(百分比0-100,已复核口径)")


class JudgeStatusResult(OrmResult):
    consistency_state: str = Field(
        description="一致性状态(normal:达标;drifted:漂移;insufficient_data:复核样本不足)"
    )
    drift_paused: bool = Field(
        description="漂移门禁暂停提示(一致率低于阈值时为true,提示暂停依赖判分的门禁判定)"
    )
    consistency_threshold: int = Field(description="一致性阈值(百分比,sys_dict ai_eval)")
    review_stats: JudgeReviewStats = Field(description="人工复核统计")


class EvalReviewItem(OrmResult):
    id: int = Field(description="复核记录ID")
    run_id: int = Field(description="评测记录ID")
    sample_id: int = Field(description="样本ID")
    agent_id: int = Field(description="Agent ID")
    agent_name: str | None = Field(default=None, description="Agent 名称")
    judge_passed: bool = Field(description="判分模型判定(true:通过;false:失败)")
    risk_level: str = Field(description="样本风险等级快照")
    status: int = Field(description="复核状态(1:待复核;2:已复核)")
    agree: bool | None = Field(
        default=None, description="人工判定(true:与判分一致;false:不一致;未复核为null)"
    )
    remark: str | None = Field(default=None, description="复核备注")
    create_time: datetime | None = Field(default=None, description="生成时间")


class EvalReviewQueueResult(OrmResult):
    items: list[EvalReviewItem] = Field(description="复核项列表(待复核优先)")
    pending: int = Field(description="待复核数")
    reviewed: int = Field(description="已复核数")


class EvalReviewSubmitForm(BaseModel):
    agree: bool = Field(..., description="人工判定(true:与判分一致;false:不一致)")
    remark: str | None = Field(default=None, max_length=500, description="复核备注")


class EvalReviewDetailResult(OrmResult):
    """复核详情（GET /eval-center/runs/{runId}/samples/{sampleId}）"""

    run_id: int = Field(description="评测记录ID")
    agent_id: int = Field(description="Agent ID")
    agent_name: str | None = Field(default=None, description="Agent 名称")
    sample_id: int = Field(description="样本ID")
    task_goal: str = Field(description="任务目标")
    allowed_input: str | None = Field(default=None, description="允许输入(样本已删除时为null)")
    expected_result: str | None = Field(default=None, description="期望结果(样本已删除时为null)")
    expected_process: str | None = Field(default=None, description="期望过程(样本已删除时为null)")
    forbidden_behavior: str | None = Field(default=None, description="禁止行为(样本已删除时为null)")
    tools: list[str] | None = Field(
        default=None, description="样本声明的可用工具(样本已删除时为null)"
    )
    risk_level: str = Field(description="样本风险等级快照")
    judge_passed: bool = Field(description="判分模型判定(true:通过;false:失败)")
    actual_output: str | None = Field(default=None, description="本次评测实际输出")
    error: str | None = Field(default=None, description="样本执行异常信息(正常执行为null)")
    scores: dict[str, float] = Field(
        default_factory=dict, description="四维得分明细(键为四维指标名,无判分结果时为空)"
    )
    notes: dict[str, str] = Field(default_factory=dict, description="四维判分说明(键为四维指标名)")


class EvalFailedSampleItem(OrmResult):
    """手动触发评测返回的失败样本明细（scores 内键为四维指标名，保持 snake_case）"""

    sample_id: int = Field(description="样本ID")
    task_goal: str = Field(description="任务目标")
    risk_level: str = Field(description="风险等级")
    passed: bool = Field(description="通过状态(恒为false)")
    error: str | None = Field(default=None, description="执行异常信息")
    scores: dict[str, float] = Field(default_factory=dict, description="四维得分")
    notes: dict[str, str] = Field(default_factory=dict, description="各维度差异说明")
    metrics: dict[str, Any] = Field(default_factory=dict, description="效率指标(步数/延迟/token)")


class EvalRunGateResult(OrmResult):
    """评测门禁判定结果（异步任务 succeeded 时的 result）"""

    run_id: int | None = Field(description="评测执行记录ID")
    passed: bool = Field(description="门禁是否通过")
    degraded: bool = Field(
        default=False, description="相对上次完成评测总分退化超过阈值（退化即门禁不通过）"
    )
    insufficient_eval: bool = Field(
        default=False, description="回归集样本不足（无考题可判），门禁阻断而非放行"
    )
    score_summary: dict[str, Any] | None = Field(
        default=None, description="四维评分聚合(dimensions 内键为四维指标名)"
    )
    failed_samples: list[EvalFailedSampleItem] = Field(
        default_factory=list, description="失败样本明细"
    )


class EvalTaskAcceptedResult(OrmResult):
    """异步评测任务受理结果（POST /agents/{agentId}/eval/runs）"""

    task_id: str = Field(description="评测任务ID(轮询 GET /eval/tasks/{taskId})")


class EvalTaskProgress(OrmResult):
    done: int = Field(description="已执行样本数")
    total: int = Field(description="样本总数")


class EvalTaskStatusResult(OrmResult):
    """异步评测任务状态（GET /agents/{agentId}/eval/tasks/{taskId}）"""

    task_id: str = Field(description="评测任务ID")
    status: str = Field(
        description="任务状态(pending:排队;running:执行中;succeeded:完成;failed:失败)"
    )
    progress: EvalTaskProgress = Field(description="执行进度")
    error: str | None = Field(default=None, description="失败原因(failed 时有值)")
    result: EvalRunGateResult | None = Field(
        default=None, description="门禁判定结果(succeeded 时有值)"
    )
