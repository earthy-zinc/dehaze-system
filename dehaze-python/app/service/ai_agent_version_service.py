"""智能体版本管理服务：草稿快照、发布（回归集门禁）、回滚、版本历史"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CacheService
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.models.schema.ai_agent import AgentVersionResult
from app.models.schema.common import PageResult
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.service.ai.strategies import agent_config_resolver
from app.service.ai_eval_center_service import eval_center_service
from app.service.ai_eval_service import eval_service

# 版本快照缓存 Key / TTL（后端实现 §4.3）
_AGENT_PUBLISHED_KEY = "ai:agent:{agent_id}:published"
_AGENT_PUBLISHED_TTL = 1800


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
        """写入一条版本记录（草稿/已发布），返回新版本实体。"""
        snapshot = await self._build_snapshot(db, redis, agent)
        version_no = await self.ai_agent_version_repository.next_version_no(db, agent.id)
        version = SysAiAgentVersion(
            agent_id=agent.id,
            version_no=version_no,
            snapshot=snapshot,
            status=status,
            change_note=change_note,
            operator_id=operator_id,
        )
        db.add(version)
        await db.flush()
        await db.refresh(version)
        return version

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
        version = await self._write_draft(
            db, redis, agent, operator_id, change_note, status=1
        )
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
        2) 调用 EvalService.run_regression（trigger_type='publish'）做回归门禁；
        3) 门禁通过：旧 published 置历史，写新 version_no 已发布版本，失效
           published 缓存，返回 version_no；
        4) 门禁失败：抛业务异常（含失败样本明细）。
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

        gate = await eval_service.run_regression(db, redis, agent_id, trigger_type="publish")
        if not gate.get("passed", False):
            failed = gate.get("failed_samples") or []
            raise BusinessException(
                ResultCode.DATA_STATE_NOT_ALLOW,
                f"发布门禁未通过，失败样本：{failed}",
            )

        # 门禁通过：旧已发布版本置历史，写入新已发布版本
        if drift_exempted:
            change_note = f"[漂移豁免]{change_note}"
        await self.ai_agent_version_repository.demote_published(db, agent_id)
        version = await self._write_draft(
            db, redis, agent, operator_id, change_note, status=2
        )
        await CacheService(redis).delete(_AGENT_PUBLISHED_KEY.format(agent_id=agent_id))
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
        await self.ai_agent_repository.replace_skills(db, agent_id, snapshot.get("skills", []) or [])
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
        await CacheService(redis).delete(_AGENT_PUBLISHED_KEY.format(agent_id=agent_id))
        return version.version_no

    async def list_versions(
        self,
        db: AsyncSession,
        redis: Redis,
        agent_id: int,
        page: int,
        size: int,
    ) -> PageResult[AgentVersionResult]:
        """版本历史列表（分页，按版本号倒序）。"""
        versions = await self.ai_agent_version_repository.list_versions(db, agent_id)
        total = len(versions)
        items = [
            AgentVersionResult.model_validate(v) for v in versions[(page - 1) * size : page * size]
        ]
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
