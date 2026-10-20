"""智能体版本管理服务：草稿快照、发布（回归集门禁）、回滚、版本历史"""

import logging
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import defer_after_commit, get_db_session
from app.infrastructure.cache.cache import CacheService
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.models.schema.ai_agent import AgentVersionResult
from app.models.schema.common import PageResult
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service.ai.strategies import agent_config_resolver
from app.service.ai_eval_center_service import eval_center_service
from app.service.ai_eval_service import eval_service

logger = logging.getLogger(__name__)

# 版本快照缓存 Key / TTL（后端实现 §4.3）
_AGENT_PUBLISHED_KEY = "ai:agent:{agent_id}:published"
_AGENT_PUBLISHED_TTL = 1800

# 版本状态（1:草稿;2:已发布）
_VERSION_STATUS_PUBLISHED = 2

# 取号（MAX+1）与插入非原子，并发发布撞 (agent_id, version_no) 唯一键时的重试次数
_VERSION_NO_CONFLICT_RETRY = 5


def _defer_audit(db: AsyncSession, **kwargs) -> None:
    """登记提交后写审计日志（回滚不留痕）；create_audit_async 为同步方法，需包成协程。"""

    async def _write() -> None:
        mongo_audit_log_repository.create_audit_async(**kwargs)

    defer_after_commit(db, _write)


def _defer_clear_published_cache(db: AsyncSession, redis: Redis, agent_id: int) -> None:
    """登记事务提交后失效 published 缓存。

    提交前失效会被并发请求击穿回填：新版本号尚未提交，读到的是旧版本号，脏缓存存活至 TTL。
    """
    defer_after_commit(
        db, lambda: CacheService(redis).delete(_AGENT_PUBLISHED_KEY.format(agent_id=agent_id))
    )


class AgentVersionService:
    def __init__(
        self,
        ai_agent_repository=ai_agent_repository,
        ai_agent_version_repository=ai_agent_version_repository,
    ):
        self.ai_agent_repository = ai_agent_repository
        self.ai_agent_version_repository = ai_agent_version_repository

    async def _build_snapshot(self, db: AsyncSession, redis: Redis, agent) -> dict:
        """序列化主表可编辑态为版本快照（对齐后端实现 §2.4 / 契约）。"""
        skills = await self.ai_agent_repository.list_skill_names(db, agent.id)
        mcp = await self.ai_agent_repository.list_mcp_namespaces(db, agent.id)
        subagents = [
            {"agent_id": s.subagent_agent_id, "priority": s.priority, "endpoint_id": s.endpoint_id}
            for s in await self.ai_agent_repository.list_subagents(db, agent.id)
        ]
        # resolved_config：系统默认 ← Agent 配置 两级合并（不含会话级），冻结"继承默认"语义，
        # 保证已发布版本行为可复现，不受后续 sys_dict 变更影响。
        resolved_config = await agent_config_resolver.resolve(db, redis, agent.config, None)
        return {
            "name": agent.name,
            "description": agent.description,
            "system_prompt": agent.system_prompt,
            "model_id": agent.model_id,
            "reasoning_mode": agent.reasoning_mode,
            "config": agent.config,
            "resolved_config": resolved_config,
            "permissions": agent.permissions,
            "is_subagent": agent.is_subagent,
            "is_team": agent.is_team,
            "is_exposed": agent.is_exposed,
            "skills": skills,
            "mcp_namespaces": mcp,
            "subagents": subagents,
        }

    async def _write_draft(
        self,
        db: AsyncSession,
        redis: Redis,
        agent,
        operator_id: int | None,
        change_note: str | None,
        status: int = 1,
    ) -> SysAiAgentVersion:
        """写入一条版本记录（草稿/已发布），返回新版本实体。

        版本号为 MAX+1，与插入非原子；并发发布撞 (agent_id, version_no) 唯一键时
        在 SAVEPOINT 内回滚该次插入、按冲突号递增重试（同事务内 MAX 受可重复读快照
        影响不会变大，故按冲突号推进而非重新取号）。
        """
        snapshot = await self._build_snapshot(db, redis, agent)
        version_no = await self.ai_agent_version_repository.next_version_no(db, agent.id)
        for attempt in range(_VERSION_NO_CONFLICT_RETRY):
            version = SysAiAgentVersion(
                agent_id=agent.id,
                version_no=version_no,
                snapshot=snapshot,
                status=status,
                change_note=change_note,
                operator_id=operator_id,
            )
            try:
                async with db.begin_nested():
                    db.add(version)
                    await db.flush()
            except IntegrityError:
                logger.warning(
                    "版本号 %s 冲突(agent=%s)，重试 %s",
                    version_no,
                    agent.id,
                    attempt + 1,
                )
                version_no += 1
                continue
            await db.refresh(version)
            return version
        raise BusinessException(ResultCode.DATA_EXISTS, "版本号并发冲突，请重试发布")

    async def save_draft(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        operator_id: int | None,
        change_note: str | None = None,
    ) -> AgentVersionResult:
        """保存草稿快照（更新 Agent 后调用，生成 status=1 草稿版本）。"""
        agent = await self.ai_agent_repository.get_by_id(db, agent_id)
        if not agent:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")
        version = await self._write_draft(db, redis, agent, operator_id, change_note, status=1)
        return AgentVersionResult.model_validate(version)

    async def publish(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        operator_id: int,
        change_note: str = "",
        force: bool = False,
    ) -> int:
        """发布 Agent：通过回归集门禁后，序列化可编辑态为新已发布版本。

        1) 判分漂移门禁：judge 一致性状态为 drifted 时阻断（force 豁免，
           豁免记入 change_note 可追溯）；漂移仅暂停门禁判定，不绕过回归结果；
        2) 将当前可编辑态固化为草稿并提交（门禁评测对象即最新草稿），调用
           EvalService.run_regression（trigger_type='publish'）做回归门禁；
        3) 门禁通过：旧 published 置历史，写新 version_no 已发布版本，失效
           published 缓存，返回 version_no；
        4) 门禁失败：抛业务异常（回归集无考题/评分退化/失败样本三类文案），草稿保留供继续修订。
        """
        agent = await self.ai_agent_repository.get_by_id(db, agent_id)
        if not agent:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")

        judge = await eval_center_service.judge_status(db)
        drift_exempted = False
        if judge.get("consistency_state") == "drifted":
            if not force:
                raise BusinessException(
                    ResultCode.OPERATION_NOT_ALLOW,
                    "判分模型漂移，发布门禁暂停，请联系管理员校准判分模型后再发布",
                )
            drift_exempted = True

        # 门禁评测对象为最新草稿（评测后端实现 §1.1）。save_draft 无独立 API，草稿仅由
        # 发布链路生成：发布前将当前可编辑态固化为草稿，保证"被评测配置 == 即将发布配置"，
        # 也避免无草稿历史的首发被"暂无草稿版本"阻断。
        await self._write_draft(db, redis, agent, operator_id, "发布评测草稿", status=1)
        # 回归评测是分钟级长跑，不得在请求事务内执行：先提交草稿（独立事务读不到未提交
        # 的草稿快照），再在独立 session 中跑，请求连接不随评测被长期占用。
        await db.commit()
        async with get_db_session() as eval_db:
            gate = await eval_service.run_regression(
                eval_db, redis, agent_id, trigger_type="publish"
            )
        if not gate.get("passed", False):
            # 回归集已配置但无考题：failed_samples 为空，通用文案会误导成"空列表 bug"
            if gate.get("insufficient_eval"):
                raise BusinessException(
                    ResultCode.DATA_STATE_NOT_ALLOW,
                    "回归集无考题，请先维护评测样本后再发布",
                )
            # 退化阻断时 failed_samples 同样为空，需单独给出可读文案
            if gate.get("degraded"):
                raise BusinessException(
                    ResultCode.DATA_STATE_NOT_ALLOW,
                    "发布门禁未通过：回归评分较上次完成评测退化超阈值，请检查本次配置变更或回退修改",
                )
            failed = gate.get("failed_samples") or []
            raise BusinessException(
                ResultCode.DATA_STATE_NOT_ALLOW,
                f"发布门禁未通过，失败样本：{failed}",
            )

        # 门禁通过：旧已发布版本置历史，写入新已发布版本
        if drift_exempted:
            change_note = f"[漂移豁免]{change_note}"
        await self.ai_agent_version_repository.demote_published(db, agent_id)
        version = await self._write_draft(db, redis, agent, operator_id, change_note, status=2)
        _defer_clear_published_cache(db, redis, agent_id)
        if force:
            _defer_audit(
                db,
                operator_id=operator_id,
                target_type="ai_agent",
                target_id=agent_id,
                action="publish_force",
                module="ai_agent",
                after_value={
                    "version_no": version.version_no,
                    "change_note": change_note,
                    "drift_exempted": drift_exempted,
                },
            )
        return version.version_no

    async def rollback(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        version_no: int,
        operator_id: int,
    ) -> int:
        """回滚到历史已发布版本：snapshot 覆盖主表可编辑态 + 写新已发布版本（不覆盖历史）。

        新版本 change_note 记录"回滚自 v{version_no}"。
        """
        agent = await self.ai_agent_repository.get_by_id(db, agent_id)
        if not agent:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Agent 不存在")
        target = await self.ai_agent_version_repository.get_by_agent_and_version(
            db, agent_id, version_no
        )
        if not target:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "回滚目标版本不存在")
        # 草稿版本是发布链路的评测中间态，配置未经门禁验证，不允许作为回滚目标
        if target.status != _VERSION_STATUS_PUBLISHED:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅可回滚到已发布版本")
        snapshot = target.snapshot or {}

        # snapshot 覆盖主表可编辑态
        agent.name = snapshot.get("name", agent.name)
        agent.description = snapshot.get("description", agent.description)
        agent.system_prompt = snapshot.get("system_prompt")
        agent.model_id = snapshot.get("model_id", agent.model_id)
        agent.reasoning_mode = snapshot.get("reasoning_mode", agent.reasoning_mode)
        agent.config = snapshot.get("config")
        agent.permissions = snapshot.get("permissions")
        agent.is_subagent = snapshot.get("is_subagent", agent.is_subagent)
        agent.is_team = snapshot.get("is_team", agent.is_team)
        agent.is_exposed = snapshot.get("is_exposed", agent.is_exposed)
        # 关联关系覆盖式恢复
        await self.ai_agent_repository.replace_skills(
            db, agent_id, snapshot.get("skills", []) or []
        )
        await self.ai_agent_repository.replace_mcp_namespaces(
            db, agent_id, snapshot.get("mcp_namespaces", []) or []
        )
        await self.ai_agent_repository.replace_subagents(
            db,
            agent_id,
            [
                {
                    "agent_id": s["agent_id"],
                    "priority": s.get("priority", 0),
                    "endpoint_id": s.get("endpoint_id"),
                }
                for s in (snapshot.get("subagents") or [])
            ],
        )
        await db.flush()

        # 写新已发布版本，历史不覆盖
        await self.ai_agent_version_repository.demote_published(db, agent_id)
        version = await self._write_draft(
            db, redis, agent, operator_id, f"回滚自 v{version_no}", status=2
        )
        _defer_clear_published_cache(db, redis, agent_id)
        _defer_audit(
            db,
            operator_id=operator_id,
            target_type="ai_agent",
            target_id=agent_id,
            action="rollback",
            module="ai_agent",
            after_value={"from_version_no": version_no, "to_version_no": version.version_no},
        )
        return version.version_no

    async def list_versions(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        page: int,
        size: int,
    ) -> PageResult[AgentVersionResult]:
        """版本历史列表（分页下推 SQL，按版本号倒序，不加载快照大字段）。"""
        versions, total = await self.ai_agent_version_repository.list_versions(
            db, agent_id, (page - 1) * size, size
        )
        items = [AgentVersionResult.model_validate(v) for v in versions]
        return PageResult(list=items, total=total)

    async def diff_versions(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        base_version_no: int,
        target_version_no: int,
    ) -> list[dict]:
        """版本差异对比：递归比较两个版本快照，返回差异字段列表。

        返回 [{field: 点分路径, base: 旧值, target: 新值}]；嵌套 JSON 递归展开，
        仅记录叶节点差异。base 缺失视为 target 新增，target 缺失视为 base 删除。
        """

        async def _load(version_no: int) -> dict:
            version = await self.ai_agent_version_repository.get_by_agent_and_version(
                db, agent_id, version_no
            )
            if not version:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"版本 {version_no} 不存在")
            return version.snapshot or {}

        base = await _load(base_version_no)
        target = await _load(target_version_no)

        def _diff(base_val: Any, target_val: Any, prefix: str, acc: list) -> None:
            # 两侧均为 dict：递归到叶节点
            if isinstance(base_val, dict) and isinstance(target_val, dict):
                keys = set(base_val) | set(target_val)
                for key in sorted(keys):
                    _diff(
                        base_val.get(key),
                        target_val.get(key),
                        f"{prefix}.{key}" if prefix else key,
                        acc,
                    )
                return
            if isinstance(base_val, list) and isinstance(target_val, list):
                # 列表视为整体，整体比较（子 Agent/Skills 顺序敏感）
                if base_val != target_val:
                    acc.append({"field": prefix, "base": base_val, "target": target_val})
                return
            if base_val != target_val:
                acc.append({"field": prefix, "base": base_val, "target": target_val})

        diffs: list[dict] = []
        _diff(base, target, "", diffs)
        return diffs


agent_version_service = AgentVersionService()
