"""DehazeToolsBuilder：按 Agent 配置装载 dehaze 业务工具为 deepagents tools

将 clarity 专属能力（算法推荐、批量处理、知识库检索与发现、产物发现、Skill 加载、
MCP 网关元工具、任务状态查询）封装为 LangChain StructuredTool。工具所需的运行时上下文
（conv_id/user_id/msg_id/stream_session_id/model_id）通过共享 ctx 字典传入——该字典由
DeepAgentBuilder 创建，DehazeHooksMiddleware 在 abefore_agent 时从 state 填充业务标识，
业务工具在执行时从 ctx 读取并回写任务状态，供 get_task_status 查询。
"""

import json
import logging

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt

from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.clients.mcp_gateway_client import mcp_gateway_client
from app.infrastructure.clients.web_search_client import (
    check_search_quota,
    format_websearch_results,
    web_search_client,
)
from app.infrastructure.sandbox.code_sandbox import code_sandbox
from app.service.ai.builders.knowledge_base_tool import knowledge_base_client
from app.service.ai.middleware.async_resume import submit_batch_task
from app.service.ai.middleware.interrupt_handler import ConfirmKind, interrupt_handler
from app.service.ai.middleware.run_context import current_run_ctx
from app.service.ai.service.algorithm_recommend_service import recommend_algorithm
from app.service.ai.service.batch_process_service import (
    BATCH_ASYNC_THRESHOLD,
    process_batch,
    validate_owned_image_url,
)
from app.service.ai.service.skill_manager import skill_manager
from app.service.ai_artifact_service import ai_artifact_service

logger = logging.getLogger(__name__)


def _skill_names_desc() -> str:
    """skill_load 描述中的可用 Skills 列表。

    每次构图（build_business_tools）时动态读取 skill_manager 内存缓存——
    启动播种（lifecycle.refresh_index）晚于模块导入，模块级常量会定死为空列表。
    """
    return ", ".join(s["name"] for s in skill_manager.discover_skills()) or "（暂无可用 Skill）"


def _get_task_status_snapshot(ctx: dict) -> str:
    """序列化当前任务状态快照（任务类型/算法/参数/状态/产物），供 get_task_status 返回"""
    return json.dumps(
        {
            "task_type": ctx.get("task_type", ""),
            "model": ctx.get("task_model", ""),
            "params": ctx.get("task_params", {}),
            "status": ctx.get("task_status", ""),
            "task_id": ctx.get("task_id", ""),
            "artifacts": ctx.get("task_artifacts", []),
        },
        ensure_ascii=False,
    )


async def _degrade_to_kb(query: str, prefix: str) -> str:
    """网络搜索降级路径：自动执行知识库检索并拼前缀。

    网络搜索不可用/配额尽时在同一次调用内降级为知识库检索（对齐 §5.1 失败降级表与
    T-MF-132）。知识库未接入（空结果）时给出明确提示。
    """
    results = await knowledge_base_client.retrieve(query, top_k=5)
    if not results:
        return prefix + "\n知识库暂无可检索内容"
    return prefix + "\n" + knowledge_base_client.format_results(results)


def _format_sandbox_result(result: dict) -> str:
    """将沙箱执行结果封装为结构化文本（stdout/stderr/exitCode/超时）。"""
    parts: list[str] = []
    if result.get("timedOut"):
        parts.append(result.get("stderr", ""))
    else:
        if result.get("stdout"):
            parts.append(f"exitCode: {result.get('exitCode')}\nstdout:\n{result['stdout']}")
        if result.get("stderr"):
            parts.append(f"stderr:\n{result['stderr']}")
    if result.get("truncated", {}).get("stdout") or result.get("truncated", {}).get("stderr"):
        parts.append("（输出已按长度上限截断）")
    if not parts:
        parts.append(f"exitCode: {result.get('exitCode')}（无输出）")
    return "\n".join(parts)


def build_business_tools(ctx: dict) -> list[StructuredTool]:
    """按 Agent 配置装载 clarity 业务工具。

    Args:
        ctx: 静态配置模板 dict（由 DeepAgentBuilder 创建）。工具执行时经
            current_run_ctx 取当次 run 的运行时上下文（图缓存复用下防跨 run 串扰），
            无 run 上下文（单测直调）回退该模板。

    Returns:
        LangChain StructuredTool 列表。
    """
    # 闭包内 ctx 会重新绑定，模板需以独立名字捕获
    template = ctx

    async def _algorithm_recommend(image_url: str, query: str) -> str:
        ctx = current_run_ctx(template)
        # LLM 给出的地址必须为本系统本人产物，否则预测侧会去下载任意/他人 URL
        async with get_db_session() as db:
            await validate_owned_image_url(db, ctx["user_id"], image_url)
        summary, interrupt_data = await recommend_algorithm(
            ctx["conversation_id"],
            ctx["message_id"],
            ctx["user_id"],
            image_url,
            query,
            ctx["stream_session_id"],
        )
        top = summary.get("model") if summary else None
        if not top:
            return "未匹配到合适的模型"
        # 暂停图等待用户确认（interrupt_data 为完整确认载荷，由 SseEventConverter
        # 统一推送 interrupt SSE）；resume 时 interrupt() 同步返回确认数据（图从中断处继续）
        interrupt(interrupt_data)
        # 更新任务状态，供 get_task_status 查询
        ctx.update(
            {
                "task_type": "dehaze",
                "task_model": str(top.get("modelId", "")),
                "task_params": {"image_url": image_url, "query": query},
                "task_id": "",
                "task_status": "processing",
                "task_artifacts": [
                    {
                        "id": top.get("recommendationId"),
                        "type": "algorithm_recommend",
                        "summary": {
                            "modelName": top.get("modelName"),
                            "reason": top.get("reason"),
                        },
                    }
                ],
            }
        )
        return f"推荐模型: {top.get('modelName')}，原因: {top.get('reason')}"

    async def _async_batch_process(image_urls: list[str], model: str) -> str:
        """大批量异步处理：提交后台任务 → 进入 async_wait 中断，任务完成回调自动 resume。

        中断点保存 async_wait 数据（taskId/taskType/预计耗时）供断线恢复与回调反查；
        resume 时 interrupt() 返回任务结果摘要，据此更新任务状态并返回给 LLM。
        """
        ctx = current_run_ctx(template)
        thread_id = f"{ctx['conversation_id']}:{ctx['message_id']}"
        task_id = submit_batch_task(
            conv_id=ctx["conversation_id"],
            msg_id=ctx["message_id"],
            user_id=ctx["user_id"],
            image_urls=image_urls,
            model=model,
            stream_session_id=ctx["stream_session_id"],
            thread_id=thread_id,
        )
        # assistant 消息 task_id 写异步任务 ID（供前端进度追踪/回调反查）
        try:
            async with get_db_session() as db:
                from app.repository.ai_message_repository import ai_message_repository

                await ai_message_repository.update_task_id(db, ctx["message_id"], task_id)
                await db.commit()
        except Exception:
            logger.warning("异步任务 ID 落库失败: task_id=%s", task_id, exc_info=True)
        interrupt_data = {
            "type": "async_wait",
            "stream_session_id": ctx["stream_session_id"],
            "data": {
                "taskId": task_id,
                "taskType": "batch_process",
                "estDuration": f"约 {len(image_urls) * 5} 秒",  # 粗估，供前端进度展示
                "imageCount": len(image_urls),
            },
        }
        await interrupt_handler.save_interrupt(thread_id, "async_wait", interrupt_data)
        # 暂停图等待异步任务完成；resume 时 interrupt() 返回 {async_task: summary}
        resume = interrupt(interrupt_data)
        summary = (resume or {}).get("async_task") or {}
        ctx.update(
            {
                "task_type": "dehaze",
                "task_model": str(model),
                "task_params": {"image_urls": image_urls},
                "task_status": "completed" if summary.get("failed") == 0 else "failed",
                "task_id": task_id,
                "task_artifacts": [
                    {
                        "id": r["pred_log_id"],
                        "type": "image_result",
                        "summary": {"modelId": model},
                    }
                    for r in summary.get("results", [])
                    if r.get("pred_log_id")
                ],
            }
        )
        total = summary.get("total", len(image_urls))
        success = summary.get("success", 0)
        failed = summary.get("failed", 0)
        return f"批量处理完成：共 {total} 张，成功 {success} 张，失败 {failed} 张"

    async def _batch_process(image_urls: list[str], query: str, model: str = "") -> str:
        ctx = current_run_ctx(template)
        # LLM 给出的地址必须为本系统本人产物，否则预测侧会去下载任意/他人 URL
        async with get_db_session() as db:
            for image_url in image_urls:
                await validate_owned_image_url(db, ctx["user_id"], image_url)
        # 大批量（>异步阈值）采用异步提交 + interrupt(async_wait)：提交后台任务后暂停图，
        # 任务完成回调自动 resume；小批量保持同步直返，避免轻任务中断体验劣化。
        if len(image_urls) > BATCH_ASYNC_THRESHOLD:
            return await _async_batch_process(image_urls, model)
        summary = await process_batch(
            ctx["conversation_id"],
            ctx["message_id"],
            ctx["user_id"],
            image_urls,
            model,
            ctx["stream_session_id"],
        )
        ctx.update(
            {
                "task_type": "dehaze",
                "task_model": str(model),
                "task_params": {"image_urls": image_urls},
                "task_status": "completed" if summary["failed"] == 0 else "failed",
                "task_artifacts": [
                    {
                        "id": r["pred_log_id"],
                        "type": "image_result",
                        "summary": {"modelId": model},
                    }
                    for r in summary.get("results", [])
                    if r.get("pred_log_id")
                ],
            }
        )
        return (
            f"批量处理完成：共 {summary['total']} 张，"
            f"成功 {summary['success']} 张，失败 {summary['failed']} 张"
        )

    async def _get_task_status() -> str:
        return _get_task_status_snapshot(current_run_ctx(template))

    def _skill_load(skill_name: str) -> str:
        instruction = skill_manager.load_skill(skill_name)
        # 渐进披露第二级：返回完整指令（§2 Skills 渐进式加载），截断会使指令失效
        return instruction if instruction else "Skill 未找到"

    async def _mcp_lookup_tool(query: str) -> str:
        result = await mcp_gateway_client.lookup_tool(query)
        return result[:4000] if result else "无匹配工具"

    async def _mcp_execute_tool(tool_name: str, arguments: dict) -> str:
        result = await mcp_gateway_client.execute_tool(tool_name, arguments)
        return result[:4000] if result else "工具执行完成"

    async def _visual_read(artifact_id: int) -> str:
        # 多模态视觉读取：评估图片效果时使用（与用户主动要求记住的行为无关）。
        # 多模态调用的 input_tokens 归集到 ctx，随本次推理计入 Token 消耗。
        ctx = current_run_ctx(template)
        redis = await get_redis_client()
        async with get_db_session() as db:
            text, input_tokens = await ai_artifact_service.visual_read(
                db, redis, ctx["user_id"], artifact_id, ctx.get("model_id")
            )
            ctx["multimodal_input_tokens"] = ctx.get("multimodal_input_tokens", 0) + input_tokens
            return text

    async def _web_search(query: str, max_results: int = 8) -> str:
        """网络搜索：配额超限或服务不可用时自动降级为知识库检索（§5.1）。"""
        ctx = current_run_ctx(template)
        max_results = max(5, min(int(max_results), 10))  # Top 5-10 上限
        redis = await get_redis_client()
        allowed = await check_search_quota(redis, ctx["user_id"])
        if not allowed:
            # 配额用尽：提示并自动降级知识库检索
            return await _degrade_to_kb(
                query, "搜索配额已用尽，可尝试知识库检索。已降级为知识库检索："
            )
        results = await web_search_client.search(query, max_results)
        if not results:
            # 网络搜索不可用/超时：自动降级知识库检索
            return await _degrade_to_kb(query, "网络搜索不可用，已降级为知识库检索：")
        return format_websearch_results(results)

    async def _knowledge_base_search(query: str, top_k: int = 5) -> str:
        """知识库检索：返回带来源引用的结果，命中内容注入上下文供回答引用溯源。"""
        ctx = current_run_ctx(template)
        results = await knowledge_base_client.retrieve(query, top_k=top_k, user_id=ctx["user_id"])
        if not results:
            return "知识库暂无可检索内容"
        return knowledge_base_client.format_results(results)

    async def _list_knowledge_bases() -> str:
        """列出当前用户可见的知识库（id/名称/简介），供模型选择检索范围。"""
        ctx = current_run_ctx(template)
        bases = await knowledge_base_client.list_bases(ctx["user_id"])
        if not bases:
            return "当前没有可用的知识库"
        return "\n".join(
            f"- id={b['id']} {b['name']}" + (f"：{b['description']}" if b["description"] else "")
            for b in bases
        )

    async def _list_artifacts() -> str:
        """列出本会话已生成的产物（图片结果/报告等），供引用与 visual_read 取 id。"""
        ctx = current_run_ctx(template)
        async with get_db_session() as db:
            page = await ai_artifact_service.list_by_conversation(
                db, ctx["conversation_id"], ctx["user_id"], 1, 20
            )
        if not page.list:
            return "本会话暂无可用的产物"
        return "\n".join(
            f"- id={a.id} type={a.type} " + json.dumps(a.summary or {}, ensure_ascii=False)[:200]
            for a in page.list
        )

    async def _execute_code(code: str, language: str = "python", timeout: int = 60) -> str:
        """受限沙箱执行 Python 脚本或 Shell 命令；高风险能力需用户确认（§2.2）。

        Shell 一律确认；Python 仅在命中网络访问/子进程/沙箱外文件读写时确认——
        这两类能力与 Shell 同级，否则 python 模式可绕过确认访问宿主网络与文件。
        """
        language = (language or "python").lower()
        if language == "shell":
            rejection = code_sandbox.check_command_policy(code)
            if rejection:
                return rejection
            action = "execute_shell_command"
            impact = "将在受限沙箱中执行该 Shell 命令，请确认命令内容与影响范围。"
        else:
            risk = code_sandbox.check_python_risk(code) if language == "python" else None
            if risk is None:
                result = await code_sandbox.execute_code(code, language, timeout)
                return _format_sandbox_result(result)
            action = "execute_python_code"
            impact = f"该代码包含{risk}（与 Shell 同级的高风险操作），请确认影响范围。"

        ctx = current_run_ctx(template)
        interrupt_data = {
            "type": "confirm",
            "stream_session_id": ctx.get("stream_session_id"),
            "data": {
                "confirmKind": ConfirmKind.DANGEROUS_OP,
                "action": action,
                "command": code,
                "impact": impact,
            },
        }
        # 危险操作确认与算法推荐一致：持久化中断点，否则 resume 端点查不到中断、用户无法确认
        await interrupt_handler.save_interrupt(
            f"{ctx['conversation_id']}:{ctx['message_id']}", "confirm", interrupt_data
        )
        resume = interrupt(interrupt_data)
        if isinstance(resume, dict) and resume.get("confirmed") is False:
            if language == "shell":
                return "用户拒绝了该 Shell 命令的执行"
            return "用户拒绝了该代码的执行"
        result = await code_sandbox.execute_code(code, language, timeout)
        return _format_sandbox_result(result)

    return [
        StructuredTool.from_function(
            name="web_search",
            description=(
                "检索实时网络信息，返回标题+摘要+来源链接（Top 5-10 条）。"
                "适用：时效性强或训练知识之外的信息（最新算法/论文/新闻、当前价格与事件）。"
                "单用户每小时有限额；配额用尽或服务不可用时会自动降级为知识库检索。"
            ),
            coroutine=_web_search,
        ),
        StructuredTool.from_function(
            name="knowledge_base_search",
            description=(
                "检索平台知识库（RAG），返回带小节路径与来源引用的命中片段。"
                "适用：回答本项目/平台的产品设计、系统架构、API 规范、测试与部署等文档类或"
                "业务知识问题时，必须先调用本工具获取真实内容，不要凭记忆作答；"
                "回答引用命中内容时标注来源。"
            ),
            coroutine=_knowledge_base_search,
        ),
        StructuredTool.from_function(
            name="list_knowledge_bases",
            description=(
                "列出当前账号可见的知识库（id/名称/简介）。"
                "不确定有哪些知识库、或需要按库判断检索范围时先调用；"
                "一般知识问答直接调 knowledge_base_search 即可，无需先列库。"
            ),
            coroutine=_list_knowledge_bases,
        ),
        StructuredTool.from_function(
            name="list_artifacts",
            description=(
                "列出本会话已生成的产物（图片处理结果/报告等，含产物 id、类型与摘要）。"
                "适用：需要引用历史产物、或为 visual_read 获取 artifact_id 时。"
            ),
            coroutine=_list_artifacts,
        ),
        StructuredTool.from_function(
            name="execute_code",
            description=(
                "在受限沙箱（与宿主隔离，工作目录为临时目录）中执行 Python 脚本或 Shell 命令，"
                "默认 60s 超时。适用：数据计算、统计、格式转换等。"
                "Shell 命令与高风险 Python（网络访问/子进程/沙箱外文件读写）需用户确认后执行。"
            ),
            coroutine=_execute_code,
        ),
        StructuredTool.from_function(
            name="algorithm_recommend",
            description=(
                "根据用户图片与需求推荐去雾算法，并推送确认卡片等待用户选择。"
                "适用：用户上传图片并要求去雾/增强但未指定模型时。"
            ),
            coroutine=_algorithm_recommend,
        ),
        StructuredTool.from_function(
            name="batch_process",
            description=(
                "批量处理多张图片（去雾/增强等）。适用：用户给出多张图片要求统一处理；"
                "图片数量超过异步阈值时自动转后台任务并等待结果。"
            ),
            coroutine=_batch_process,
        ),
        StructuredTool.from_function(
            name="skill_load",
            description=(
                "加载指定 Skill 的完整工作流指令并注入上下文，随后按其步骤执行。"
                f"可用 Skills: {_skill_names_desc()}。"
                "适用：任务有既定标准流程（如去雾效果评估报告生成、算法选型决策）时"
                "先加载对应 Skill。"
            ),
            func=_skill_load,
        ),
        StructuredTool.from_function(
            name="mcp_lookup_tool",
            description=(
                "在平台后端 API 工具目录中按自然语言查找匹配的 MCP 工具，返回工具名与参数 schema。"
                "适用：需要调用平台系统能力但不知道具体工具名时先查找，再用 mcp_execute_tool 执行。"
            ),
            coroutine=_mcp_lookup_tool,
        ),
        StructuredTool.from_function(
            name="mcp_execute_tool",
            description=(
                "按 tool_name 与 arguments 调用 mcp_lookup_tool 找到的 MCP 工具执行后端 API。"
            ),
            coroutine=_mcp_execute_tool,
        ),
        StructuredTool.from_function(
            name="get_task_status",
            description=(
                "查询当前任务状态（任务类型、算法模型、参数、状态、产物列表）。"
                "适用：用户询问处理进度、或需要确认上一步产物时。"
            ),
            coroutine=_get_task_status,
        ),
        StructuredTool.from_function(
            name="visual_read",
            description=(
                "读取指定产物图片并交由多模态模型评价其视觉效果。"
                "入参 artifact_id（来自产物引用行或 list_artifacts 返回）。"
                "适用：评估去雾/增强结果质量时。受每日视觉读取次数限制。"
            ),
            coroutine=_visual_read,
        ),
    ]
