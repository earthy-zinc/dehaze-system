"""TeamBuilder：基于 langgraph-supervisor 构建 Team 团队图

设计文档 §6 Team 团队协作。Team（is_team=true）作为会话入口时，由专职
supervisor（Team Lead 模型 + 调度提示词）负责任务分解、成员路由与结果汇总，
成员 Agent 各自保留独立的系统提示词/工具/模型/权限与护栏，专注自身领域。

成员两类：
- 本地成员：经 DeepAgentBuilder 编译为 deep agent 子图（Pregel），
  create_supervisor 为每个成员暴露 transfer_to_<成员名> 移交工具
- 远程成员（endpoint_id 非空）：不编译为本地子图，构造为 supervisor 可直接
  调用的远程 task 工具（§5.4，外部账单仅记录状态与耗时，不计入平台配额）
"""

import logging

from langgraph_supervisor import create_supervisor

from app.core.exceptions import BusinessException
from app.infrastructure.llm.client.dehaze_chat_model import DehazeChatModel
from app.service.ai.builders.deep_agent_builder import (
    DeepAgentBuilder,
    _build_remote_tool,
    _load_endpoint,
    _make_ctx,
    register_graph_ctx_template,
)

logger = logging.getLogger(__name__)


class TeamBuilder:
    """构建基于 langgraph-supervisor 的 Team 团队图。"""

    @staticmethod
    async def build_team(
        db,
        redis,
        lead_snapshot: dict,
        member_snapshots: list[dict],
        remote_members: list[dict] | None = None,
        checkpointer=None,
    ):
        """构建 Team 团队编译图。

        Args:
            db: 异步数据库会话。
            redis: 异步 Redis 客户端。
            lead_snapshot: Team Lead（主 Agent）快照。
            member_snapshots: 本地团队成员（子 Agent）已发布快照列表。
            remote_members: 远程 A2A 成员关联项 [{agent_id, endpoint_id}]，构造为
                supervisor 可直接调用的远程 task 工具。
            checkpointer: LangGraph Checkpointer。

        Returns:
            编译后的 supervisor 图（CompiledStateGraph）。
        """
        # 本地成员编译为 deep agent 子图；携带成员身份（被 supervisor transfer_to_* 移交时
        # 其模型调用用量按成员 agentCode 归属）
        agents = []
        for snap in member_snapshots:
            compiled = await DeepAgentBuilder.build_from_snapshot(
                db,
                redis,
                snap,
                checkpointer=checkpointer,
                subagent={"name": snap.get("name") or "", "priority": 0},
            )
            agents.append(compiled)

        # 远程成员构造为 supervisor 可直接调用的远程 task 工具
        remote_tools = []
        failures: list[str] = []
        for rel in remote_members or []:
            endpoint_id = int(rel.get("endpoint_id") or 0)
            endpoint = await _load_endpoint(db, endpoint_id)
            if not endpoint:
                # 配置错误不得静默跳过（对齐 _build_subagents）：否则团队会少一个远程成员
                # 上线而无人知晓，聚合所有失败项一并报错暴露。
                failures.append(f"远程成员 {rel.get('agent_id')}: 端点 {endpoint_id} 不可用")
                continue
            remote_tools.append(
                _build_remote_tool(endpoint, f"remote_{rel.get('agent_id')}", endpoint.name, {})
            )
        if failures:
            raise BusinessException("Team 远程成员配置不完整，拒绝构建团队: " + "; ".join(failures))

        lead_config = lead_snapshot["config"]
        # langgraph-supervisor 的 parallel_tool_calls 为 bool（是否允许并行移交）；
        # max_parallel=1 语义为串行，>=2 才开放并行移交
        parallel_tool_calls = int(lead_config["max_parallel"]) > 1

        supervisor_model = DehazeChatModel(model=lead_snapshot.get("model_id", ""))
        supervisor_graph = create_supervisor(
            agents=agents,
            model=supervisor_model,
            prompt=lead_snapshot.get("system_prompt") or "",
            tools=remote_tools or None,
            output_mode="last_message",
            parallel_tool_calls=parallel_tool_calls,
            supervisor_name=lead_snapshot.get("name") or "supervisor",
        )
        compiled = supervisor_graph.compile(checkpointer=checkpointer)
        # 登记 run ctx 模板（成员 deep agent 子图的 before_agent 节点经预置共享 ctx
        # 取得会话标识，与 DeepAgentBuilder 同一机制）
        register_graph_ctx_template(compiled, _make_ctx(lead_snapshot, lead_config))
        return compiled
