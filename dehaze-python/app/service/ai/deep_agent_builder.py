"""DeepAgentBuilder：从 Agent 版本快照组装 deepagents 图

核心桥接（设计文档 §3.2/§3.3/§5）：把数据库驱动的 Agent 配置（系统提示词、
模型、推理参数、Skills、MCP 命名空间、子 Agent、安全护栏）翻译成
create_deep_agent 入参，返回标准 CompiledStateGraph，供 reasoning_service
以 astream(subgraphs=True) 运行。

- 模型：DehazeChatModel 包装 LlmClient（计费透出 usage）
- 工具：DehazeToolsBuilder 按 Agent 配置装载业务工具
- 护栏：GuardrailMiddleware 按 config.guardrails + 命名空间授权配置
- 计费/生命周期：DehazeHooksMiddleware 桥接 AgentHooks
- 子 Agent：递归构造 SubAgent spec（独立提示词/工具/模型/权限，不继承父，防递归）
- 推理范式：reasoning_mode=auto 时先跑 complexity_evaluator 决定 direct/react
- 安全默认：通过 HarnessProfile 排除 execute 工具（dehaze Agent 不执行任意 shell 命令）
"""

import asyncio
import logging
import time
from typing import Any

from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.subagents import SubAgent
from deepagents.profiles import HarnessProfile, register_harness_profile
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.config import settings
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent_endpoint import SysAiAgentEndpoint
from app.service.ai.a2a_client import a2a_client
from app.service.ai.a2a_task_mapper import a2a_task_mapper
from app.service.ai.capability_constraints import CapabilityConstraintsMiddleware
from app.service.ai.complexity_evaluator import evaluate_complexity
from app.service.ai.dehaze_chat_model import DehazeChatModel
from app.service.ai.dehaze_hooks_middleware import DehazeAgentState, DehazeHooksMiddleware
from app.service.ai.dehaze_tools_builder import build_business_tools
from app.service.ai.guardrail_middleware import GuardrailMiddleware
from app.service.ai.mcp_namespace_prefilter import McpNamespacePrefilterMiddleware
from app.service.ai.paradigm_middleware import ParadigmMiddleware
from app.service.ai.prompt_composer import compose_system_prompt
from app.service.ai.tool_failure_guard import ToolFailureGuardMiddleware
from app.service.ai_agent_service import AgentService

logger = logging.getLogger(__name__)

# deepagents 环境默认配置（provider 为 DehazeChatModel 的 llm_type）：
# 排除 execute 工具，dehaze 业务 Agent 不应执行任意 shell 命令（安全默认）。
# 幂等注册，仅一次。
_HARNESS_PROFILE_KEY = "dehazechatmodel"
_HARNESS_PROFILE_REGISTERED = False


def _ensure_harness_profile() -> None:
    global _HARNESS_PROFILE_REGISTERED
    if _HARNESS_PROFILE_REGISTERED:
        return
    register_harness_profile(
        _HARNESS_PROFILE_KEY,
        HarnessProfile(excluded_tools=frozenset({"execute"})),
    )
    _HARNESS_PROFILE_REGISTERED = True


# 远程 A2A 子 Agent 轮询参数（§5.4.5：外部账单，仅记录调用状态与耗时，不计入平台配额）
_A2A_POLL_INTERVAL = 1.0
_A2A_POLL_TIMEOUT = 120.0
_A2A_TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled", "rejected"})


def _build_filesystem_permissions(raw: Any) -> list[FilesystemPermission] | None:
    """将快照 permissions（dict 列表）转为 deepagents FilesystemPermission 列表。

    快照 permissions 为 {operations, paths, mode} 结构（mode: allow/deny/interrupt），
    非法条目跳过；无权限配置返回 None（create_deep_agent 不启用文件系统护栏）。
    """
    if not isinstance(raw, list) or not raw:
        return None
    result: list[FilesystemPermission] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                FilesystemPermission(
                    operations=item.get("operations") or [],
                    paths=item.get("paths") or [],
                    mode=item.get("mode", "allow"),
                )
            )
        except (TypeError, ValueError):
            logger.warning("跳过非法文件系统权限配置: %s", item)
            continue
    return result or None


def _make_ctx(snapshot: dict, config: dict) -> dict:
    """创建共享运行时上下文（计费贯穿 + 步数/预算护栏 + 任务状态）。

    config 为三级合并后的完整配置（get_published_snapshot 保证），推理参数默认值
    唯一来源于 sys_dict 种子（ai_reasoning_defaults），不在此处硬编码兜底；
    max_steps 未显式配置时以 ReAct 基础范式默认值为准。
    """
    return {
        "max_steps": int(config.get("max_steps") or config["max_steps_react"]),
        "token_budget": int(config["token_budget"]),
        "tool_timeout": int(config.get("tool_timeout") or settings.AI_REASONING_TOOL_TIMEOUT),
        "retry_max": int(config.get("retry_max") or settings.AI_REASONING_RETRY_MAX),
        "token_used": 0,
        "step_count": 0,
        "task_type": "",
        "task_algorithm": "",
        "task_params": {},
        "task_status": "",
        "task_id": "",
        "task_artifacts": [],
        "_model_id": snapshot.get("model_id", ""),
    }


def _build_agent_core(snapshot: dict, ctx: dict) -> dict:
    """构造单个 Agent（主 Agent 或子 Agent）的核心配置。

    返回 {model, tools, middleware}，由 create_deep_agent / SubAgent 消费。
    """
    config = snapshot["config"]
    guardrails = config.get("guardrails") or {}
    mcp_namespaces = snapshot.get("mcp_namespaces") or []
    tools: list[BaseTool] = build_business_tools(ctx)
    model = DehazeChatModel(model=snapshot.get("model_id", ""))
    middleware = [
        DehazeHooksMiddleware(ctx),
        GuardrailMiddleware(guardrails, mcp_namespaces),
        # 多步推理范式编排：图恒定装载，仅按运行时 state.reasoning_mode 介入
        ParadigmMiddleware(model, config, ctx),
        # 能力扩展（F-M08-006）：MCP 命名空间预筛选动态注入（消息层，不进图缓存键）
        McpNamespacePrefilterMiddleware(agent_namespaces=mcp_namespaces or None),
        # 容量/任务清单约束包装 write_file、write_todos（置于失败保护外层，
        # 使容量超限等约束错误不计入连续失败）
        CapabilityConstraintsMiddleware(),
        # 连续失败保护：置于最内层以观察真实工具执行结果
        ToolFailureGuardMiddleware(),
    ]
    return {"model": model, "tools": tools, "middleware": middleware}


async def _load_endpoint(db, endpoint_id: int):
    """按端点 ID 加载启用的外部 A2A 端点，不存在或禁用返回 None。"""
    if not endpoint_id:
        return None
    endpoint = await db.get(SysAiAgentEndpoint, endpoint_id)
    if not endpoint or endpoint.status != 1:
        logger.warning("远程端点不可用：endpoint_id=%s", endpoint_id)
        return None
    return endpoint


def _build_remote_tool(endpoint, shadow_name: str, shadow_desc: str, ctx: dict) -> BaseTool:
    """为远程 A2A 子 Agent 构造 task 工具（普通异步工具，不占平台 Token 配额）。

    工具内部经 A2AClient message/send → tasks/get 轮询获取产物，经 A2ATaskMapper
    反解为文本摘要返回主 Agent 上下文；仅记录调用状态与耗时（§5.4.5）。
    """

    @tool(shadow_name, description=shadow_desc or f"调用远程子 Agent「{shadow_name}」处理任务")
    async def remote_sub_agent_call(task_input: str) -> str:
        """将任务委托给外部 A2A 子 Agent，返回其结果摘要。"""
        started = time.monotonic()
        ctx["task_type"] = "a2a"
        ctx["task_status"] = "processing"
        ctx["task_algorithm"] = ""
        try:
            task = await a2a_client.message_send(
                endpoint, [{"role": "user", "content": task_input}]
            )
        except Exception as e:
            ctx["task_status"] = "failed"
            ctx["task_id"] = ""
            logger.warning("远程子 Agent %s 发起失败: %s", shadow_name, e)
            return f"远程子 Agent「{shadow_name}」调用失败: {e}"

        ctx["task_id"] = task.id
        deadline = started + _A2A_POLL_TIMEOUT
        while task.status not in _A2A_TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                ctx["task_status"] = "failed"
                return f"远程子 Agent「{shadow_name}」任务 {task.id} 轮询超时"
            await asyncio.sleep(_A2A_POLL_INTERVAL)
            try:
                task = await a2a_client.task_get(endpoint, task.id)
            except Exception as e:
                ctx["task_status"] = "failed"
                logger.warning("远程子 Agent %s 轮询失败: %s", shadow_name, e)
                return f"远程子 Agent「{shadow_name}」任务状态查询失败: {e}"

        ctx["task_status"] = "completed" if task.status == "completed" else "failed"
        if task.status != "completed":
            return f"远程子 Agent「{shadow_name}」任务未完成，状态: {task.status}"

        # 产物 + 最终 agent 消息反解为文本摘要返回主 Agent 上下文
        parts = [a2a_task_mapper.artifact_to_context(a)["text"] for a in task.artifacts]
        for msg in task.history or []:
            if msg.role == "agent":
                text = msg.to_text()
                if text:
                    parts.append(text)
        return f"[{shadow_name}] " + "\n".join(p for p in parts if p)

    return remote_sub_agent_call


async def _build_subagents(
    db, redis, snapshot: dict, parent_ctx: dict
) -> tuple[list[SubAgent], list[BaseTool]]:
    """构造子 Agent 能力列表。

    返回 (本地 SubAgent spec 列表, 远程 A2A 子 Agent 工具列表)：
    - endpoint_id 非空的关联项构造远程 task 工具（外部账单，仅记录状态耗时）；
    - 其余为本地子 Agent，递归构造独立提示词/工具/模型/权限，防递归。
    """
    subagents: list[SubAgent] = []
    remote_tools: list[BaseTool] = []
    for rel in snapshot.get("subagents") or []:
        if rel.get("endpoint_id"):
            endpoint = await _load_endpoint(db, rel.get("endpoint_id"))
            if not endpoint:
                continue
            shadow_name = f"remote_{rel.get('agent_id')}"
            remote_tools.append(
                _build_remote_tool(endpoint, shadow_name, endpoint.name, parent_ctx)
            )
            continue
        try:
            sub_snapshot = await AgentService().get_published_snapshot(
                db, redis, rel.get("agent_id")
            )
        except BusinessException as e:
            # 子 Agent 无已发布版本或不存在（业务异常）：运行时容错跳过，不中断主 Agent 构建
            logger.warning("子 Agent %s 无已发布版本，跳过: %s", rel.get("agent_id"), e)
            continue
        if not sub_snapshot:
            continue
        sub_ctx = dict(parent_ctx)
        sub_ctx.update({"_model_id": sub_snapshot.get("model_id", "")})
        core = _build_agent_core(sub_snapshot, sub_ctx)
        subagents.append(
            {
                "name": sub_snapshot.get("name", f"agent_{rel.get('agent_id')}"),
                "description": sub_snapshot.get("description", ""),
                "system_prompt": compose_system_prompt(sub_snapshot, None),
                "model": core["model"],
                "tools": core["tools"],
                "middleware": core["middleware"],
            }
        )
    return subagents, remote_tools


class DeepAgentBuilder:
    """从 Agent 版本快照组装 deepagents 图的构建器。"""

    @staticmethod
    async def build_from_snapshot(
        db,
        redis,
        snapshot: dict,
        checkpointer=None,
    ) -> CompiledStateGraph:
        """按快照构建单个 deep agent 编译图（含子 Agent）。

        Args:
            db: 异步数据库会话。
            redis: 异步 Redis 客户端。
            snapshot: Agent 已发布版本快照（契约见 get_published_snapshot）。
            checkpointer: LangGraph Checkpointer（复用 dehaze RedisSaver）。

        Returns:
            编译后的 deep agent 图（CompiledStateGraph）。
        """
        _ensure_harness_profile()
        config = snapshot["config"]
        ctx = _make_ctx(snapshot, config)
        core = _build_agent_core(snapshot, ctx)
        subagents, remote_tools = await _build_subagents(db, redis, snapshot, ctx)

        # 图按 Agent 版本缓存复用：system_prompt 仅含"稳定层+Agent 人设"（随快照稳定），
        # 会话场景提示词随会话变化，不得进入图缓存键，由 reasoning 层运行时注入。
        system_prompt = compose_system_prompt(snapshot, None)
        # 文件系统权限：快照 permissions（operations/paths/mode）转 FilesystemPermission，
        # deepagents 据此自动生成 interrupt_on —— mode=interrupt 的写操作触发用户确认
        # （危险操作护栏）
        permissions = _build_filesystem_permissions(snapshot.get("permissions"))
        return create_deep_agent(
            model=core["model"],
            tools=core["tools"] + remote_tools,
            system_prompt=system_prompt,
            middleware=core["middleware"],
            subagents=subagents or None,
            permissions=permissions,
            checkpointer=checkpointer,
            state_schema=DehazeAgentState,
            name=snapshot.get("name") or None,
        )

    @staticmethod
    async def resolve_reasoning_mode(
        snapshot: dict, messages: list[dict], model_id: str
    ) -> tuple[str, int]:
        """确定实际推理范式与最大步数。

        reasoning_mode=auto 时先跑 complexity_evaluator（direct/react/plan_execute/
        reflexion）；其余值表示固定范式。返回 (reasoning_mode, max_steps)。
        """
        mode = snapshot.get("reasoning_mode") or "auto"
        config = snapshot.get("config") or {}
        if mode == "auto":
            result = await evaluate_complexity({"messages": messages, "model_id": model_id})
            mode = result["reasoning_mode"]
        default_steps = {
            "direct": 1,
            "plan_execute": int(config["max_steps_plan"]),
            "reflexion": int(config["max_steps_reflexion"]),
        }.get(mode, int(config["max_steps_react"]))
        max_steps = int(config.get("max_steps") or default_steps)
        return mode, max_steps
