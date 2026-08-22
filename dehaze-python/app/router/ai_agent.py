from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_agent import (
    AgentCopyForm,
    AgentCreate,
    AgentDetail,
    AgentListItem,
    AgentMcpForm,
    AgentPageQuery,
    AgentPublishForm,
    AgentSkillsForm,
    AgentStatusForm,
    AgentSubAgentsForm,
    AgentTestForm,
    AgentUpdate,
    AgentVersionDetail,
    AgentVersionResult,
)
from app.models.schema.common import PageResult
from app.service.ai_agent_service import agent_service
from app.service.ai_agent_version_service import agent_version_service

router = APIRouter(prefix="/api/v1/ai/agents", tags=["AI对话"])

_MANAGE_PERMISSION = "ai:agent:manage"


def _is_manager(user: UserContext) -> bool:
    return user.is_root or _MANAGE_PERMISSION in user.permissions


@router.get("", response_model=Result[PageResult[AgentListItem]], summary="Agent 列表")
async def list_agents(
    query: AgentPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    if _is_manager(user):
        result = await agent_service.list_agents(
            db, redis, query.pageNum, query.pageSize, query.keyword, query.status
        )
    else:
        items = await agent_service.list_enabled(db, redis)
        result = PageResult(list=items, total=len(items))
    return success(result)


@router.get("/enabled", response_model=Result[list[AgentListItem]], summary="可选 Agent 列表")
async def list_enabled_agents(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.list_enabled(db, redis)
    return success(result)


@router.post("", response_model=Result[AgentDetail], summary="创建 Agent")
@require_permission(_MANAGE_PERMISSION)
async def create_agent(
    form: AgentCreate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.create_agent(db, redis, form)
    return success(result)


@router.get("/{agent_id}", response_model=Result[AgentDetail], summary="Agent 详情")
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.get_detail(db, redis, agent_id)
    return success(result)


@router.put("/{agent_id}", response_model=Result[AgentDetail], summary="更新 Agent")
@require_permission(_MANAGE_PERMISSION)
async def update_agent(
    agent_id: int,
    form: AgentUpdate,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.update_agent(db, redis, agent_id, form)
    return success(result)


@router.delete("/{agent_id}", response_model=Result[None], summary="删除 Agent")
@require_permission(_MANAGE_PERMISSION)
async def delete_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await agent_service.delete_agent(db, redis, agent_id)
    return success(msg="一切ok")


@router.patch("/{agent_id}/status", response_model=Result[None], summary="启停 Agent")
@require_permission(_MANAGE_PERMISSION)
async def set_agent_status(
    agent_id: int,
    form: AgentStatusForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await agent_service.set_status(db, redis, agent_id, form.status)
    return success(msg="一切ok")


@router.post("/{agent_id}/test", response_model=Result[dict], summary="Agent 测试预览")
@require_permission(_MANAGE_PERMISSION)
async def test_agent(
    agent_id: int,
    form: AgentTestForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.test_agent(db, redis, agent_id, form.message)
    return success(result)


@router.post("/{agent_id}/copy", response_model=Result[AgentDetail], summary="复制 Agent")
@require_permission(_MANAGE_PERMISSION)
async def copy_agent(
    agent_id: int,
    form: AgentCopyForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_service.copy_agent(db, redis, agent_id, form.agent_code)
    return success(result)


@router.put("/{agent_id}/skills", response_model=Result[None], summary="设置 Skills（覆盖式）")
@require_permission(_MANAGE_PERMISSION)
async def set_agent_skills(
    agent_id: int,
    form: AgentSkillsForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await agent_service.set_skills(db, redis, agent_id, form.skills)
    return success(msg="一切ok")


@router.put("/{agent_id}/mcps", response_model=Result[None], summary="设置 MCP 命名空间（覆盖式）")
@require_permission(_MANAGE_PERMISSION)
async def set_agent_mcps(
    agent_id: int,
    form: AgentMcpForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await agent_service.set_mcp(db, redis, agent_id, form.mcp_namespaces)
    return success(msg="一切ok")


@router.put("/{agent_id}/subagents", response_model=Result[None], summary="设置子 Agent（覆盖式）")
@require_permission(_MANAGE_PERMISSION)
async def set_agent_subagents(
    agent_id: int,
    form: AgentSubAgentsForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await agent_service.set_subagents(db, redis, agent_id, form)
    return success(msg="一切ok")


@router.post("/{agent_id}/publish", response_model=Result[dict], summary="发布 Agent")
@require_permission(_MANAGE_PERMISSION)
async def publish_agent(
    agent_id: int,
    form: AgentPublishForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    version_no = await agent_version_service.publish(db, redis, agent_id, user.id, form.change_note)
    return success({"version_no": version_no})


@router.get(
    "/{agent_id}/versions",
    response_model=Result[PageResult[AgentVersionResult]],
    summary="版本历史",
)
async def list_versions(
    agent_id: int,
    query: AgentPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_version_service.list_versions(
        db, redis, agent_id, query.pageNum, query.pageSize
    )
    return success(result)


@router.get("/{agent_id}/versions/diff", response_model=Result[list[dict]], summary="版本差异对比")
async def diff_versions(
    agent_id: int,
    base: int,
    target: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await agent_version_service.diff_versions(db, redis, agent_id, base, target)
    return success(result)


@router.get(
    "/{agent_id}/versions/{version_no}",
    response_model=Result[AgentVersionDetail],
    summary="版本快照详情",
)
async def get_version_detail(
    agent_id: int,
    version_no: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    version, snapshot = await agent_service.get_version_detail(db, redis, agent_id, version_no)
    return success(
        AgentVersionDetail.model_validate(
            {
                "id": version.id,
                "agent_id": agent_id,
                "version_no": version_no,
                "status": version.status,
                "change_note": version.change_note,
                "operator_id": version.operator_id,
                "create_time": version.create_time,
                "snapshot": snapshot,
            }
        )
    )


@router.post(
    "/{agent_id}/versions/{version_no}/rollback",
    response_model=Result[dict],
    summary="回滚到历史版本",
)
@require_permission(_MANAGE_PERMISSION)
async def rollback_agent(
    agent_id: int,
    version_no: int,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    new_version_no = await agent_version_service.rollback(db, redis, agent_id, version_no, user.id)
    return success({"version_no": new_version_no})
