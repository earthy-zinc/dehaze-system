"""智能体管理服务：CRUD、启停、复制、删除校验、默认 Agent、关联管理、缓存"""

from langchain_core.runnables import RunnableConfig
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import defer_after_commit
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.base import get_current_user_id
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
from app.repository.ai_agent_eval_repository import (
    ai_agent_eval_dataset_repository,
    ai_agent_eval_run_repository,
    ai_agent_eval_sample_repository,
)
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.ai_mcp_namespace_repository import ai_mcp_namespace_repository
from app.repository.ai_skill_repository import ai_skill_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
from app.service.ai_agent_version_service import agent_version_service

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


def _defer_audit(db: AsyncSession, **kwargs) -> None:
    """登记提交后写审计日志。

    危险操作若在事务回滚后仍留痕，会误导事后追溯；故审计与缓存失效同挂提交后回调。
    回调契约是协程（defer_after_commit 会 await），而 create_audit_async 是同步
    fire-and-forget，需在此包装为 async。
    """

    async def _write() -> None:
        mongo_audit_log_repository.create_audit_async(**kwargs)

    defer_after_commit(db, _write)


def _defer_cache_delete(db: AsyncSession, redis: Redis, *keys: str) -> None:
    """登记事务提交后的缓存失效。

    提交前失效会被并发请求击穿回填：此时新值尚未提交，读到的是旧数据，回填的脏缓存
    将存活到 TTL 到期。
    """

    async def _delete() -> None:
        cache = CacheService(redis)
        for key in keys:
            await cache.delete(key)

    defer_after_commit(db, _delete)


def _defer_clear_agent_caches(db: AsyncSession, redis: Redis, agent: SysAiAgent) -> None:
    """Agent 更新/启停/删除时失效相关缓存（事务提交后执行）。"""
    _defer_cache_delete(
        db,
        redis,
        _AGENT_DETAIL_KEY.format(agent_code=agent.agent_code),
        _AGENT_SKILLS_KEY.format(agent_id=agent.id),
        _AGENT_MCP_KEY.format(agent_id=agent.id),
        _AGENT_SUBAGENTS_KEY.format(agent_id=agent.id),
        _AGENT_PUBLISHED_KEY.format(agent_id=agent.id),
        _AGENT_ENABLED_LIST_KEY,
    )


async def _get_agent_or_404(repository, db: AsyncSession, agent_id: int) -> SysAiAgent:
    """Agent 不存在抛 A0404（统一 404 码）。"""
    agent = await repository.get_by_id(db, agent_id)
    if not agent:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")
    return agent


async def _load_subagent_items(
    repository, db: AsyncSession, parent_agent_id: int
) -> list[SubAgentItem]:
    """加载子 Agent 关联详情（含名称/编码/描述，供 AgentDetail 展示）。"""
    links = await repository.list_subagents(db, parent_agent_id)
    if not links:
        return []
    sub_ids = [link.subagent_agent_id for link in links]
    sub_agents = {a.id: a for a in await repository.get_by_ids(db, sub_ids)}
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
    def __init__(
        self,
        ai_agent_repository=ai_agent_repository,
        ai_agent_version_repository=ai_agent_version_repository,
        ai_skill_repository=ai_skill_repository,
        agent_version_service=agent_version_service,
    ):
        self.ai_agent_repository = ai_agent_repository
        self.ai_agent_version_repository = ai_agent_version_repository
        self.ai_skill_repository = ai_skill_repository
        self.agent_version_service = agent_version_service

    async def ensure_default_agent(self, db: AsyncSession, redis: Redis) -> None:
        """应用启动时确保默认 Agent（agent_code='default'）存在且有已发布版本（§2.11.12）。

        默认 Agent 是未指定 Agent 时的会话兜底，运行面（ReasoningService._load_snapshot）
        按已发布版本快照组装推理图；若只建主表记录而无已发布版本，任何默认会话推理都会
        因"该 Agent 暂无已发布版本"失败。故此处同时保证至少存在一条已发布版本（幂等）。
        """
        existing = await self.ai_agent_repository.get_by_code(db, DEFAULT_AGENT_CODE)
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
            await self.ai_agent_repository.create(db, default)
            await db.flush()
            await db.refresh(default)
            existing = default

        # 无已发布版本则序列化当前可编辑态为初始已发布版本（status=2，v1），保证默认会话可推理。
        # 若已发布版本的生效 config 缺推理默认键（如历史快照在 sys_dict 默认补齐前生成，
        # resolved_config 为空导致 deep_agent_builder 缺 max_steps_* 快速失败），则按当前
        # 可编辑态重新发布新版本，保证运行面拿到完整默认配置（幂等自愈）。
        published = await self.ai_agent_version_repository.get_latest_published(db, existing.id)
        need_publish = published is None
        if published is not None:
            snapshot_cfg = (published.snapshot or {}).get("resolved_config") or {}
            if "max_steps_react" not in snapshot_cfg:
                need_publish = True
        if need_publish:
            await self.agent_version_service._write_draft(
                db, redis, existing, None, "默认 Agent 初始发布", status=2
            )
        _defer_clear_agent_caches(db, redis, existing)

    async def _build_detail(self, db: AsyncSession, agent: SysAiAgent) -> AgentDetail:
        skills = await self.ai_agent_repository.list_skill_names(db, agent.id)
        mcp = await self.ai_agent_repository.list_mcp_namespaces(db, agent.id)
        subagents = await _load_subagent_items(self.ai_agent_repository, db, agent.id)
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
                "tags": agent.tags or [],
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

    async def list_agents(
        self,
        db: AsyncSession,
        redis: Redis,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
        agent_type: str | None = None,
    ) -> PageResult[AgentListItem]:
        agents, total = await self.ai_agent_repository.paginate_agents(
            db, page, size, keyword, status, agent_type
        )
        items = []
        if agents:
            agent_ids = [a.id for a in agents]
            skill_counts = await self.ai_agent_repository.count_skills_by_agent_ids(db, agent_ids)
            mcp_counts = await self.ai_agent_repository.count_mcp_by_agent_ids(db, agent_ids)
            sub_counts = await self.ai_agent_repository.count_subagents_by_agent_ids(db, agent_ids)
            for a in agents:
                item = AgentListItem.model_validate(a)
                item.skill_count = skill_counts.get(a.id, 0)
                item.mcp_count = mcp_counts.get(a.id, 0)
                item.sub_agent_count = sub_counts.get(a.id, 0)
                items.append(item)
        return PageResult(list=items, total=total)

    async def list_enabled(self, db: AsyncSession, redis: Redis) -> list[AgentListItem]:
        """可选 Agent 列表（status=1 且非子 Agent，缓存 ai:agent:list:enabled）。"""
        cache = CacheService(redis)
        cached = await cache.get_json(_AGENT_ENABLED_LIST_KEY)
        if cached is None:
            agents = await self.ai_agent_repository.list_enabled(db)
            cached = [AgentListItem.model_validate(a).model_dump(mode="json") for a in agents]
            await cache.set_json(_AGENT_ENABLED_LIST_KEY, cached, _AGENT_ENABLED_LIST_TTL)
        return [AgentListItem.model_validate(item) for item in cached]

    async def get_detail(self, db: AsyncSession, redis: Redis, agent_id: int) -> AgentDetail:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        return await self._build_detail(db, agent)

    async def get_by_code(
        self, db: AsyncSession, redis: Redis, agent_code: str
    ) -> AgentDetail | None:
        """按编码查询 Agent 详情（缓存 ai:agent:{agent_code}，30 分钟）。"""
        cache = CacheService(redis)
        key = _AGENT_DETAIL_KEY.format(agent_code=agent_code)
        cached = await cache.get_json(key)
        if cached is not None:
            return AgentDetail.model_validate(cached)
        agent = await self.ai_agent_repository.get_by_code(db, agent_code)
        if not agent:
            return None
        detail = await self._build_detail(db, agent)
        # mode="json" 将 datetime 等类型转为 JSON 兼容值，避免 json.dumps 序列化失败
        await cache.set_json(key, detail.model_dump(mode="json"), _AGENT_DETAIL_TTL)
        return detail

    async def create_agent(self, db: AsyncSession, redis: Redis, form: AgentCreate) -> AgentDetail:
        # agent_code 唯一性校验（活跃行；唯一键含 deleted，软删后可重建同 code）
        existing = await self.ai_agent_repository.get_by_code(db, form.agent_code)
        if existing:
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
            tags=form.tags,
            sort_order=form.sort_order,
            status=form.status,
        )
        await self.ai_agent_repository.create(db, agent)
        _defer_cache_delete(db, redis, _AGENT_ENABLED_LIST_KEY)
        return await self._build_detail(db, agent)

    async def update_agent(
        self, db: AsyncSession, redis: Redis, agent_id: int, form: AgentUpdate
    ) -> AgentDetail:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
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
        _defer_clear_agent_caches(db, redis, agent)
        return await self._build_detail(db, agent)

    async def set_status(self, db: AsyncSession, redis: Redis, agent_id: int, status: int) -> None:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        agent.status = status
        await db.flush()
        _defer_clear_agent_caches(db, redis, agent)

    async def delete_agent(self, db: AsyncSession, redis: Redis, agent_id: int) -> None:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        if agent.agent_code == DEFAULT_AGENT_CODE:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "默认 Agent 不可删除")
        conversation_refs = await self.ai_agent_repository.count_conversation_references(
            db, agent.agent_code
        )
        if conversation_refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"存在 {conversation_refs} 个会话正在使用该 Agent，请先解绑",
            )
        subagent_refs = await self.ai_agent_repository.count_subagent_references(db, agent_id)
        if subagent_refs > 0:
            raise BusinessException(
                ResultCode.DATA_BIND_EXISTS,
                f"该 Agent 被 {subagent_refs} 个 Agent 作为子 Agent 引用，请先解绑",
            )
        # 评测资产按 agent_id 挂载，Agent 软删后即失去清理入口：评测集列表与评测中心
        # 聚合会残留无人认领的孤儿数据（agent_id 不复用，重建同 agent_code 得到新 id），
        # 故删除 Agent 时一并清理。样本表无逻辑删除列，随所属评测集物理删除；评测执行
        # 记录为只追加轨迹、同样无逻辑删除列，随 Agent 整体物理清理。
        eval_datasets = await ai_agent_eval_dataset_repository.list_by_agent(db, agent_id)
        eval_dataset_ids = [d.id for d in eval_datasets]
        sample_count = 0
        if eval_dataset_ids:
            sample_count = await ai_agent_eval_sample_repository.delete_by_datasets(
                db, eval_dataset_ids
            )
            await ai_agent_eval_dataset_repository.soft_delete_by_ids(db, eval_dataset_ids)
        run_count = await ai_agent_eval_run_repository.delete_by_agent(db, agent_id)

        await self.ai_agent_repository.soft_delete_by_ids(db, [agent_id])
        _defer_clear_agent_caches(db, redis, agent)
        _defer_audit(
            db,
            operator_id=get_current_user_id(),
            target_type="ai_agent",
            target_id=agent_id,
            action="delete",
            module="ai_agent",
            before_value={"agent_code": agent.agent_code, "name": agent.name},
            after_value={
                "eval_datasets_soft_deleted": len(eval_dataset_ids),
                "eval_samples_deleted": sample_count,
                "eval_runs_deleted": run_count,
            },
        )

    async def copy_agent(
        self, db: AsyncSession, redis: Redis, agent_id: int, new_code: str
    ) -> AgentDetail:
        """复制 Agent（基本信息 + 配置，不复制关联关系，编码需重新指定）。"""
        source = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        existing = await self.ai_agent_repository.get_by_code(db, new_code)
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
            tags=source.tags,
            sort_order=source.sort_order,
            status=1,
        )
        await self.ai_agent_repository.create(db, copy)
        _defer_cache_delete(db, redis, _AGENT_ENABLED_LIST_KEY)
        return await self._build_detail(db, copy)

    # ── 关联管理（覆盖式更新）────────────────────────────

    async def set_skills(
        self, db: AsyncSession, redis: Redis, agent_id: int, skill_names: list[str]
    ) -> None:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        # 引用完整性：关联的 Skill 必须存在于 sys_ai_skill（未删）
        if skill_names:
            existing = set(await self.ai_skill_repository.list_names_existing(db, skill_names))
            missing = sorted(set(skill_names) - existing)
            if missing:
                raise BusinessException(
                    ResultCode.RESOURCE_NOT_FOUND,
                    f"以下 Skill 不存在: {', '.join(missing[:5])}",
                )
        await self.ai_agent_repository.replace_skills(db, agent_id, skill_names)
        await db.flush()
        _defer_clear_agent_caches(db, redis, agent)

    async def set_mcp(
        self, db: AsyncSession, redis: Redis, agent_id: int, mcp_namespaces: list[str]
    ) -> None:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        # 引用完整性：命名空间必须已在注册 MCP Server 下声明，否则运行时装载不到任何工具
        if mcp_namespaces:
            registered = set(
                await ai_mcp_namespace_repository.list_registered_names(db, mcp_namespaces)
            )
            missing = sorted(set(mcp_namespaces) - registered)
            if missing:
                raise BusinessException(
                    ResultCode.RESOURCE_NOT_FOUND,
                    f"以下 MCP 命名空间未注册: {', '.join(missing[:5])}",
                )
        await self.ai_agent_repository.replace_mcp_namespaces(db, agent_id, mcp_namespaces)
        await db.flush()
        _defer_clear_agent_caches(db, redis, agent)

    async def set_subagents(
        self, db: AsyncSession, redis: Redis, agent_id: int, form: AgentSubAgentsForm
    ) -> None:
        agent = await _get_agent_or_404(self.ai_agent_repository, db, agent_id)
        items = [
            {"agent_id": s.agent_id, "endpoint_id": s.endpoint_id, "priority": s.priority}
            for s in form.subagents
        ]
        child_ids = [item["agent_id"] for item in items]
        if agent_id in child_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "子 Agent 不能是自身")
        await self._ensure_subagents_exist(db, child_ids)
        await self._ensure_subagents_acyclic(db, agent_id, child_ids)
        await self.ai_agent_repository.replace_subagents(db, agent_id, items)
        await db.flush()
        _defer_clear_agent_caches(db, redis, agent)

    async def _ensure_subagents_exist(self, db: AsyncSession, child_ids: list[int]) -> None:
        if not child_ids:
            return
        found = {a.id for a in await self.ai_agent_repository.get_by_ids(db, child_ids)}
        missing = sorted(set(child_ids) - found)
        if missing:
            raise BusinessException(
                ResultCode.RESOURCE_NOT_FOUND,
                f"以下子 Agent 不存在: {', '.join(str(i) for i in missing[:5])}",
            )

    async def _ensure_subagents_acyclic(
        self, db: AsyncSession, agent_id: int, child_ids: list[int]
    ) -> None:
        """校验绑定后不形成子 Agent 环：环会让推理期子 Agent 展开无限递归。"""
        path: list[int] = []
        settled: set[int] = set()

        async def _walk(node: int) -> None:
            if node in settled:
                return
            if node in path:
                cycle = "→".join(str(i) for i in [*path[path.index(node) :], node])
                raise BusinessException(ResultCode.PARAM_ERROR, f"子 Agent 绑定存在环: {cycle}")
            path.append(node)
            if node == agent_id:
                children = child_ids
            else:
                children = [
                    link.subagent_agent_id
                    for link in await self.ai_agent_repository.list_subagents(db, node)
                ]
            for child in children:
                await _walk(child)
            path.pop()
            settled.add(node)

        await _walk(agent_id)

    # ── 版本快照读取（契约）──────────────────────────────

    async def get_version_detail(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        version_no: int,
    ) -> tuple[SysAiAgentVersion, dict]:
        """取版本元数据与发布快照，版本不存在抛 A0401（版本详情端点用）"""
        version = await self.ai_agent_version_repository.get_by_agent_and_version(
            db, agent_id, version_no
        )
        if not version:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
        snapshot = await self.get_published_snapshot(db, redis, agent_id, version_no)
        return version, snapshot

    async def get_published_snapshot(
        self,
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
                published = await self.ai_agent_version_repository.get_latest_published(
                    db, agent_id
                )
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
                return self.ai_agent_version_repository.resolve_snapshot(cached_snapshot)
            version = await self.ai_agent_version_repository.get_by_agent_and_version(
                db, agent_id, version_no
            )
            if not version:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
            snapshot = version.snapshot or {}
            await cache.set_json(version_key, snapshot, _AGENT_VERSION_SNAPSHOT_TTL)
            return self.ai_agent_version_repository.resolve_snapshot(snapshot)

        version_key = _AGENT_VERSION_SNAPSHOT_KEY.format(agent_id=agent_id, version_no=version_no)
        cached = await cache.get_json(version_key)
        if cached is not None:
            return self.ai_agent_version_repository.resolve_snapshot(cached)
        version = await self.ai_agent_version_repository.get_by_agent_and_version(
            db, agent_id, version_no
        )
        if not version:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "版本快照不存在")
        snapshot = version.snapshot or {}
        await cache.set_json(version_key, snapshot, _AGENT_VERSION_SNAPSHOT_TTL)
        return self.ai_agent_version_repository.resolve_snapshot(snapshot)

    async def test_agent(self, db: AsyncSession, redis: Redis, agent_id: int, message: str) -> dict:
        """测试预览：构建独立会话运行当前已发布版本，返回 final_response + usage。

        与评测执行器同机制：独立线程上下文，不落库、不污染生产会话。
        """
        import uuid

        snapshot = await self.get_published_snapshot(db, redis, agent_id)
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
        run_config: RunnableConfig = {
            "configurable": {"thread_id": f"test:{agent_id}:{uuid.uuid4()}"}
        }
        result = await graph.ainvoke(initial_state, config=run_config)
        return {
            "final_response": result.get("final_response", ""),
            "usage": result.get("usage") or {},
        }


agent_service = AgentService()
