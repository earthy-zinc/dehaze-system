"""Skills 管理路由（F-M08-006 Skills 管理）。

对齐 API 契约（API接口.md Skills 管理接口）：
- GET  /api/v1/ai/skills        列表（管理员全量含禁用；普通用户仅启用）
- GET  /api/v1/ai/skills/{id}   详情（含指令全文）
- POST /api/v1/ai/skills        创建（ai:skill:manage）
- PUT  /api/v1/ai/skills/{id}   更新（ai:skill:manage）
- PATCH /api/v1/ai/skills/{id}/status  启停（ai:skill:manage）
- DELETE /api/v1/ai/skills/{id} 软删（ai:skill:manage）
- POST /api/v1/ai/skills/{id}/test    试运行（ai:skill:manage）
- GET  /api/v1/ai/skills/market 市场目录
- POST /api/v1/ai/skills/market 共享至市场（ai:skill:manage）

管理员权限判定沿用 ai_agent 的 _is_manager 模式（is_root 或权限标识命中）。
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_skill import (
    SkillCreate,
    SkillListItem,
    SkillMarketVO,
    SkillPageQuery,
    SkillResult,
    SkillShareForm,
    SkillStatusForm,
    SkillTestForm,
    SkillUpdate,
)
from app.models.schema.common import PageResult
from app.service.ai_skill_service import skill_manage_service

router = APIRouter(prefix="/api/v1/ai/skills", tags=["AI对话"])

_MANAGE_PERMISSION = "ai:skill:manage"


def _is_manager(user: UserContext) -> bool:
    return user.is_root or _MANAGE_PERMISSION in user.permissions


@router.get("", response_model=Result[PageResult[SkillListItem]], summary="Skills 列表")
async def list_skills(
    query: SkillPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """Skills 列表：管理员全量含禁用（分页+名称模糊）；普通用户仅返回启用项。"""
    enabled_only = not _is_manager(user)
    result = await skill_manage_service.list_skills(
        db,
        enabled_only=enabled_only,
        page=query.pageNum,
        size=query.pageSize,
        keyword=query.keyword,
    )
    return success(result)


@router.get("/market", response_model=Result[list[SkillMarketVO]], summary="Skill 市场目录")
async def list_market(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """Skill 市场目录：返回已共享（market_shared=1）的启用项。"""
    result = await skill_manage_service.list_market(db)
    return success(result)


@router.get("/{skill_id}", response_model=Result[SkillResult], summary="Skill 详情")
async def get_skill(
    skill_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.get_skill(db, skill_id)
    return success(result)


@router.post("", response_model=Result[SkillResult], summary="创建 Skill")
@require_permission(_MANAGE_PERMISSION)
async def create_skill(
    form: SkillCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.create_skill(db, form)
    return success(result)


@router.put("/{skill_id}", response_model=Result[SkillResult], summary="更新 Skill")
@require_permission(_MANAGE_PERMISSION)
async def update_skill(
    skill_id: int,
    form: SkillUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.update_skill(db, skill_id, form)
    return success(result)


@router.post("/{skill_id}/test", response_model=Result[dict], summary="试运行 Skill")
@require_permission(_MANAGE_PERMISSION)
async def test_skill(
    skill_id: int,
    form: SkillTestForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.test_skill(db, skill_id, form)
    return success(result)


@router.post("/market", response_model=Result[SkillResult], summary="共享 Skill 至市场")
@require_permission(_MANAGE_PERMISSION)
async def share_to_market(
    form: SkillShareForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.share_to_market(db, form.skillId)
    return success(result)


@router.patch("/{skill_id}/status", response_model=Result[SkillResult], summary="启停 Skill")
@require_permission(_MANAGE_PERMISSION)
async def set_skill_status(
    skill_id: int,
    form: SkillStatusForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await skill_manage_service.set_status(db, skill_id, enabled=form.status == 1)
    return success(result)


@router.delete("/{skill_id}", response_model=Result[None], summary="删除 Skill")
@require_permission(_MANAGE_PERMISSION)
async def delete_skill(
    skill_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await skill_manage_service.delete_skill(db, skill_id)
    return success(msg="一切ok")
