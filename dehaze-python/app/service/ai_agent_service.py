"""智能体管理服务：CRUD、启停、复制、删除校验、默认 Agent、关联管理、缓存"""

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.models.schema.ai_agent import (
    AgentCreate,
    AgentDetail,
    AgentListItem,
    AgentSubAgentsForm,
    AgentUpdate,
    SubAgentItem,
)
from app.models.schema.common import PageResult
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.ai_skill_repository import ai_skill_repository

# 默认 Agent 编码（后端实现 §2.1 / §2.11.12，系统预置且不可删除）
DEFAULT_AGENT_CODE = "default"

# 缓存 Key / TTL（后端实现 §4.3）
_AGENT_DETAIL_KEY = "ai:agent:{agent_code}"
_AGENT_DETAIL_TTL = 1800
_AGENT_SKILLS_KEY = "ai:agent:{agent_id}:skills"
_AGENT_SKILLS_TTL = 1800
_AGENT_MCP_KEY = "ai:agent:{agent_id}:mcp"
_AGENT_MCP_TTL = 1800
_AGENT_SUBAGENTS_KEY = "ai:agent:{agent_id}:subagents"
_AGENT_SUBAGENTS_TTL = 1800
_AGENT_VERSION_SNAPSHOT_KEY = "ai:agent:{agent_id}:version:{version_no}"
# 版本快照不可变，缓存按长期生效（版本记录本身永久保留）
_AGENT_VERSION_SNAPSHOT_TTL = 30 * CACHE_TTL_HOUR
_AGENT_PUBLISHED_KEY = "ai:agent:{agent_id}:published"
_AGENT_PUBLISHED_TTL = 1800
_AGENT_ENABLED_LIST_KEY = "ai:agent:list:enabled"
_AGENT_ENABLED_LIST_TTL = 600


async def _clear_agent_caches(redis: Redis, agent: SysAiAgent) -> None:
    """Agent 更新/启停/删除时失效相关缓存。"""
    cache = CacheService(redis)
    await cache.delete(_AGENT_DETAIL_KEY.format(agent_code=agent.agent_code))
    await cache.delete(_AGENT_SKILLS_KEY.format(agent_id=agent.id))
    await cache.delete(_AGENT_MCP_KEY.format(agent_id=agent.id))
    await cache.delete(_AGENT_SUBAGENTS_KEY.format(agent_id=agent.id))
    await cache.delete(_AGENT_PUBLISHED_KEY.format(agent_id=agent.id))
    await cache.delete(_AGENT_ENABLED_LIST_KEY)


async def _get_agent_or_404(db: AsyncSession, agent_id: int) -> SysAiAgent:
    agent = await ai_agent_repository.get_by_id(db, agent_id)
    if not agent:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")
    return agent


async def _load_subagent_items(db: AsyncSession, parent_agent_id: int) -> list[SubAgentItem]:
    """加载子 Agent 关联详情（含名称/编码/描述，供 AgentDetail 展示）。"""
    links = await ai_agent_repository.list_subagents(db, parent_agent_id)
    if not links:
        return []
    sub_ids = [link.subagent_agent_id for link in links]
    sub_agents = {a.id: a for a in await ai_agent_repository.get_by_ids(db, sub_ids)}
    items: list[SubAgentItem] = []
    for link in links:
        sub = sub_agents.get(link.subagent_agent_id)
        items.append(
            SubAgentItem(
                agent_id=link.subagent_agent_id,
                agent_name=sub.name if sub else "",
                agent_code=sub.agent_code if sub else "",
                description=sub.description if sub else "",
                endpoint_id=link.endpoint_id,
                priority=link.priority,
            )
        )
    return items


class AgentService:
    @staticmethod
    async def ensure_default_agent(db: AsyncSession, redis: Redis) -> None:
        """应用启动时确保默认 Agent（agent_code='default'）存在且有已发布版本（§2.11.12）。

        默认 Agent 是未指定 Agent 时的会话兜底，运行面（ReasoningService._load_snapshot）
        按已发布版本快照组装推理图；若只建主表记录而无已发布版本，任何默认会话推理都会
        因"该 Agent 暂无已发布版本"失败。故此处同时保证至少存在一条已发布版本（幂等）。
        """
        existing = await ai_agent_repository.get_by_code(db, DEFAULT_AGENT_CODE)
        if not existing or existing.deleted:
            default = SysAiAgent(
                agent_code=DEFAULT_AGENT_CODE,
                name="默认助手",
                description="平台通用助手，未指定 Agent 时的默认选择",
                system_prompt="你是一个乐于助人的通用助手，请用简洁清晰的语言回答用户的问题。",
                model_id=settings.AI_DEFAULT_MODEL,
                reasoning_mode="auto",
                is_subagent=0,
                is_team=0,
                is_exposed=0,
                status=1,
            )
            await ai_agent_repository.create(db, default)
            await db.flush()
            await db.refresh(default)
            existing = default

        # 无已发布版本则序列化当前可编辑态为初始已发布版本（status=2，v1），保证默认会话可推理。
        # 若已发布版本的生效 config 缺推理默认键（如历史快照在 sys_dict 默认补齐前生成，
        # resolved_config 为空导致 deep_agent_builder 缺 max_steps_* 快速失败），则按当前
        # 可编辑态重新发布新版本，保证运行面拿到完整默认配置（幂等自愈）。
        published = await ai_agent_version_repository.get_latest_published(db, existing.id)
        need_publish = published is None
        if published is not None:
            snapshot_cfg = (published.snapshot or {}).get("resolved_config") or {}
            if "max_steps_react" not in snapshot_cfg:
                need_publish = True
        if need_publish:
            from app.service.ai_agent_version_service import AgentVersionService

            snapshot = await AgentVersionService._build_snapshot(db, redis, existing)
            version_no = await ai_agent_version_repository.next_version_no(db, existing.id)
            db.add(
                SysAiAgentVersion(
                    agent_id=existing.id,
                    version_no=version_no,
                    snapshot=snapshot,
                    status=2,
                    change_note="默认 Agent 初始发布",
                    operator_id=None,
                )
            )
            await db.flush()
        await _clear_agent_caches(redis, existing)

    @staticmethod
    async def _build_detail(db: AsyncSession, agent: SysAiAgent) -> AgentDetail:
        skills = await ai_agent_repository.list_skill_names(db, agent.id)
        mcp = await ai_agent_repository.list_mcp_namespaces(db, agent.id)
        subagents = await _load_subagent_items(db, agent.id)
        return AgentDetail.model_validate(
            {
                "id": agent.id,
                "agent_code": agent.agent_code,
                "name": agent.name,
                "description": agent.description,
                "model_id": agent.model_id,
                "reasoning_mode": agent.reasoning_mode,
                "is_subagent": agent.is_subagent,
                "is_team": agent.is_team,
                "is_exposed": agent.is_exposed,
                "status": agent.status,
                "sort_order": agent.sort_order,
                "create_time": agent.create_time,
                "system_prompt": agent.system_prompt,
                "config": agent.config,
                "permissions": agent.permissions,
                "skills": skills,
                "mcp_namespaces": mcp,
                "subagents": subagents,
            }
        )

    @staticmethod
    async def list_agents(
        db: AsyncSession,
        redis: Redis,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> PageResult[AgentListItem]:
        agents, total = await ai_agent_repository.paginate_agents(db, page, size, keyword, status)
        return PageResult(list=[AgentListItem.model_validate(a) for a in agents], total=total)

    @staticmethod
    async def list_enabled(db: AsyncSession, redis: Redis) -> list[AgentListItem]:
        """可选 Agent 列表（status=1 且非子 Agent，缓存 ai:agent:list:enabled）。"""
        cache = CacheService(redis)
        cached = await cache.get_json(_AGENT_ENABLED_LIST_KEY)
        if cached is None:
            agents = await ai_agent_repository.list_enabled(db)
            cached = [AgentListItem.model_validate(a).model_dump(mode="json") for a in agents]
            await cache.set_json(_AGENT_ENABLED_LIST_KEY, cached, _AGENT_ENABLED_LIST_TTL)
        return [AgentListItem.model_validate(item) for item in cached]

    @staticmethod
    async def get_detail(db: AsyncSession, redis: Redis, agent_id: int) -> AgentDetail:
        agent = await _get_agent_or_404(db, agent_id)
        return await AgentService._build_detail(db, agent)

    @staticmethod
    async def get_by_code(db: AsyncSession, redis: Redis, agent_code: str) -> AgentDetail | None:
        """按编码查询 Agent 详情（缓存 ai:agent:{agent_code}，30 分钟）。"""
        cache = CacheService(redis)
        key = _AGENT_DETAIL_KEY.format(agent_code=agent_code)
        cached = await cache.get_json(key)
        if cached is not None:
            return AgentDetail.model_validate(cached)
        agent = await ai_agent_repository.get_by_code(db, agent_code)
        if not agent:
            return None
        detail = await AgentService._build_detail(db, agent)
        # mode="json" 将 datetime 等类型转为 JSON 兼容值，避免 json.dumps 序列化失败
        await cache.set_json(key, detail.model_dump(mode="json"), _AGENT_DETAIL_TTL)
        return detail

    @staticmethod
    async def create_agent(db: AsyncSession, redis: Redis, form: AgentCreate) -> AgentDetail:
        # agent_code 唯一性校验绕过软删查全表（类别②，删除后不可复用）
        existing = await ai_agent_repository.get_by_code(db, form.agent_code)
        if existing:
            if existing.deleted:
                raise BusinessException(
                    ResultCode.DATA_EXISTS, "该 Agent 编码已被历史记录占用，不可复用"
                )
            raise BusinessException(ResultCode.DATA_EXISTS, "Agent 编码已存在")
        agent = SysAiAgent(
            agent_code=form.agent_code,
            name=form.name,
            description=form.description,
            system_prompt=form.system_prompt,
            model_id=form.model_id,
            reasoning_mode=form.reasoning_mode,
            config=form.config.model_dump(exclude_none=True) if form.config else None,
            is_subagent=int(form.is_subagent),
            is_team=int(form.is_team),
            is_exposed=int(form.is_exposed),
            permissions=form.permissions,
            sort_order=form.sort_order,
            status=form.status,
        )
        await ai_agent_repository.create(db, agent)
        await CacheService(redis).delete(_AGENT_ENABLED_LIST_KEY)
        return await AgentService._build_detail(db, agent)

    @staticmethod
    async def update_agent(
        db: AsyncSession, redis: Redis, agent_id: int, form: AgentUpdate
    ) -> AgentDetail:
        agent = await _get_agent_or_404(db, agent_id)
        data = form.model_dump(exclude_unset=True)
        if "config" in data and data["config"] is not None:
            data["config"] = data["config"].model_dump(exclude_none=True)
        for key in ("is_subagent", "is_team", "is_exposed"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        for key, value in data.items():
            if hasattr(agent, key) and key not in ("id", "agent_code"):
                setattr(agent, key, value)
        await db.flush()
        await _clear_agent_caches(redis, agent)
        return await AgentService._build_detail(db, agent)

    @staticmethod
    async def set_status(db: AsyncSession, redis: Redis, agent_id: int, status: int) -> None:
        agent = await _get_agent_or_404(db, agent_id)
        agent.status = status
        await db.flush()
        await _clear_agent_caches(redis, agent)

    @staticmethod
    async def delete_agent(db: AsyncSession, redis: Redis, agent_id: int) -> None:
        agent = await _get_agent_or_404(db, agent_id)
        if agent.agent_code == DEFAULT_AGENT_CODE:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "默认 Agent 不可删除")
        conversation_refs = await ai_agent_repository.count_conversation_references(
            db, agent.agent_code
        )
        if conversation_refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"存在 {conversation_refs} 个会话正在使用该 Agent，请先解绑",
            )
        subagent_refs = await ai_agent_repository.count_subagent_references(db, agent_id)
        if subagent_refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"该 Agent 被 {subagent_refs} 个 Agent 作为子 Agent 引用，请先解绑",
            )
        await ai_agent_repository.soft_delete_by_ids(db, [agent_id])
        await _clear_agent_caches(redis, agent)

    @staticmethod
    async def copy_agent(
        db: AsyncSession, redis: Redis, agent_id: int, new_code: str
    ) -> AgentDetail:
        """复制 Agent（基本信息 + 配置，不复制关联关系，编码需重新指定）。"""
        source = await _get_agent_or_404(db, agent_id)
        existing = await ai_agent_repository.get_by_code(db, new_code)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "Agent 编码已存在")
        copy = SysAiAgent(
            agent_code=new_code,
            name=source.name,
            description=source.description,
            system_prompt=source.system_prompt,
            model_id=source.model_id,
            reasoning_mode=source.reasoning_mode,
            config=source.config,
            is_subagent=source.is_subagent,
            is_team=source.is_team,
            is_exposed=source.is_exposed,
            permissions=source.permissions,
            sort_order=source.sort_order,
            status=1,
        )
        await ai_agent_repository.create(db, copy)
        await CacheService(redis).delete(_AGENT_ENABLED_LIST_KEY)
        return await AgentService._build_detail(db, copy)

    # ── 关联管理（覆盖式更新）────────────────────────────

    @staticmethod
    async def set_skills(
        db: AsyncSession, redis: Redis, agent_id: int, skill_names: list[str]
    ) -> None:
        agent = await _get_agent_or_404(db, agent_id)
        # 引用完整性：关联的 Skill 必须存在于 sys_ai_skill（未删）
        if skill_names:
            existing = set(await ai_skill_repository.list_names_existing(db, skill_names))
            missing = sorted(set(skill_names) - existing)
            if missing:
                raise BusinessException(
                    ResultCode.DATA_NOT_FOUND,
                    f"以下 Skill 不存在: {', '.join(missing[:5])}",
                )
        await ai_agent_repository.replace_skills(db, agent_id, skill_names)
        await db.flush()
        await CacheService(redis).delete(_AGENT_SKILLS_KEY.format(agent_id=agent_id))
        await _clear_agent_caches(redis, agent)

    @staticmethod
    async def set_mcp(
        db: AsyncSession, redis: Redis, agent_id: int, mcp_namespaces: list[str]
    ) -> None:
        agent = await _get_agent_or_404(db, agent_id)
        await ai_agent_repository.replace_mcp_namespaces(db, agent_id, mcp_namespaces)
        await db.flush()
        await CacheService(redis).delete(_AGENT_MCP_KEY.format(agent_id=agent_id))
        await _clear_agent_caches(redis, agent)

    @staticmethod
    async def set_subagents(
        db: AsyncSession, redis: Redis, agent_id: int, form: AgentSubAgentsForm
    ) -> None:
        agent = await _get_agent_or_404(db, agent_id)
        items = [
            {"agent_id": s.agent_id, "endpoint_id": s.endpoint_id, "priority": s.priority}
            for s in form.subagents
        ]
        await ai_agent_repository.replace_subagents(db, agent_id, items)
        await db.flush()
        await CacheService(redis).delete(_AGENT_SUBAGENTS_KEY.format(agent_id=agent_id))
        await _clear_agent_caches(redis, agent)

    # ── 版本快照读取（契约）──────────────────────────────

    @staticmethod
    async def get_version_detail(
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        version_no: int,
    ) -> tuple[SysAiAgentVersion, dict]:
        """取版本元数据与发布快照，版本不存在抛 A0401（版本详情端点用）"""
        version = await ai_agent_version_repository.get_by_agent_and_version(
            db, agent_id, version_no
        )
        if not version:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
        snapshot = await AgentService.get_published_snapshot(db, redis, agent_id, version_no)
        return version, snapshot

    @staticmethod
    async def get_published_snapshot(
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        version_no: int | None = None,
    ) -> dict:
        """读取已发布版本快照（未指定版本号取当前已发布版本）。

        契约返回结构：{name, description, system_prompt, model_id, reasoning_mode,
        config(含guardrails), permissions, is_subagent, is_team, is_exposed,
        skills, mcp_namespaces, subagents}。

        返回时 config 字段替换为已冻结的 resolved_config 内容（两级合并后的生效配置），
        保证运行面读取的是发布时点确定的配置，不依赖运行时 sys_dict 再合并。
        """
        cache = CacheService(redis)
        if version_no is None:
            published_key = _AGENT_PUBLISHED_KEY.format(agent_id=agent_id)
            cached_no = await cache.get_json(published_key)
            if cached_no is not None:
                version_no = int(cached_no)
            else:
                published = await ai_agent_version_repository.get_latest_published(db, agent_id)
                if not published:
                    raise BusinessException(
                        ResultCode.RESOURCE_NOT_FOUND, "该 Agent 暂无已发布版本"
                    )
                version_no = published.version_no
                await cache.set_json(published_key, version_no, _AGENT_PUBLISHED_TTL)
            version_key = _AGENT_VERSION_SNAPSHOT_KEY.format(
                agent_id=agent_id, version_no=version_no
            )
            cached_snapshot = await cache.get_json(version_key)
            if cached_snapshot is not None:
                return AgentService._apply_resolved_config(cached_snapshot)
            version = await ai_agent_version_repository.get_by_agent_and_version(
                db, agent_id, version_no
            )
            if not version:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
            snapshot = version.snapshot or {}
            await cache.set_json(version_key, snapshot, _AGENT_VERSION_SNAPSHOT_TTL)
            return AgentService._apply_resolved_config(snapshot)

        version_key = _AGENT_VERSION_SNAPSHOT_KEY.format(agent_id=agent_id, version_no=version_no)
        cached = await cache.get_json(version_key)
        if cached is not None:
            return AgentService._apply_resolved_config(cached)
        version = await ai_agent_version_repository.get_by_agent_and_version(
            db, agent_id, version_no
        )
        if not version:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
        snapshot = version.snapshot or {}
        await cache.set_json(version_key, snapshot, _AGENT_VERSION_SNAPSHOT_TTL)
        return AgentService._apply_resolved_config(snapshot)

    @staticmethod
    def _apply_resolved_config(snapshot: dict) -> dict:
        """将快照中冻结的 resolved_config 替换到返回的 config 字段（契约字段不变）。"""
        resolved = snapshot.get("resolved_config")
        if resolved is None:
            return dict(snapshot)
        result = dict(snapshot)
        result["config"] = resolved
        return result

    @staticmethod
    async def test_agent(
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        message: str,
    ) -> dict:
        """测试预览：构建独立会话运行当前已发布版本，返回 final_response + usage。

        与评测执行器同机制：独立线程上下文，不落库、不污染生产会话。
        """
        import uuid

        from app.service.ai.deep_agent_builder import DeepAgentBuilder

        snapshot = await AgentService.get_published_snapshot(db, redis, agent_id)
        graph = await DeepAgentBuilder().build_from_snapshot(db, redis, snapshot)

        config = snapshot["config"]
        initial_state = {
            "messages": [{"role": "user", "content": message}],
            "user_id": None,
            "conversation_id": 0,
            "message_id": 0,
            "model_id": (snapshot or {}).get("model_id", ""),
            "system_prompt": (snapshot or {}).get("system_prompt"),
            "stream_session_id": f"test:{uuid.uuid4()}",
            "step_count": 0,
            "token_used": 0,
            "token_budget": config.get("token_budget", 0),
            "thoughts": [],
            "isolated_token_pool": True,
        }
        run_config = {"configurable": {"thread_id": f"test:{agent_id}:{uuid.uuid4()}"}}
        result = await graph.ainvoke(initial_state, config=run_config)
        return {
            "final_response": result.get("final_response", ""),
            "usage": result.get("usage") or {},
        }


agent_service = AgentService()
