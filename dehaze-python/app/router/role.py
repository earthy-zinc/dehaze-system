from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.common import PageResult
from app.models.schema.role import (
    MenuIdsBody,
    RoleForm,
    RoleFormVO,
    RoleOptionVO,
    RolePageQuery,
    RolePageVO,
)
from app.service.role_service import RoleService
from app.utils.datetime_utils import format_time

router = APIRouter(
    prefix="/api/v1/roles", tags=["角色管理"], dependencies=[Depends(get_current_user)]
)


DATA_SCOPE_LABELS = {
    0: "全部数据",
    1: "部门及子部门数据",
    2: "本部门数据",
    3: "本人数据",
}


@router.get("/page", response_model=Result[PageResult[RolePageVO]], summary="获取角色分页列表")
async def get_role_page(
    query: RolePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
):
    roles, total = await RoleService.get_role_list(
        db, query.pageNum, query.pageSize, query.keywords
    )

    role_list = []
    for role in roles:
        role_list.append(
            {
                "id": role.id,
                "name": role.name,
                "code": role.code,
                "sort": role.sort,
                "status": role.status,
                "dataScope": role.data_scope,
                "dataScopeLabel": DATA_SCOPE_LABELS.get(
                    role.data_scope if role.data_scope is not None else 0, ""
                ),
                "createTime": format_time(role.create_time),
            }
        )

    return success(
        {
            "list": role_list,
            "total": total,
        }
    )


@router.get("/options", response_model=Result[list[RoleOptionVO]], summary="获取角色下拉列表")
async def list_role_options(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    options = await RoleService.get_role_options(db, is_root=user.is_root)
    return success(options)


@router.post("", response_model=Result[dict[str, int]], summary="新增角色")
@require_permission("sys:role:add")
async def add_role(
    body: RoleForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await RoleService.create_role(db, redis, body.model_dump(exclude_none=True))

    return success(msg="创建成功")


@router.get("/{role_id}/form", response_model=Result[RoleFormVO], summary="获取角色表单数据")
async def get_role_form(
    role_id: int = Path(..., description="角色ID"),
    db: AsyncSession = Depends(get_db),
):
    role = await RoleService.get_role_by_id(db, role_id)

    if not role:
        return success(None)

    return success(
        {
            "id": role.id,
            "name": role.name,
            "code": role.code,
            "sort": role.sort,
            "status": role.status,
            "dataScope": role.data_scope,
        }
    )


@router.put("/{role_id}", response_model=Result[None], summary="修改角色")
@require_permission("sys:role:edit")
async def update_role(
    role_id: int = Path(..., description="角色ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    body: RoleForm = Body(...),
    user: UserContext = Depends(get_current_user),
):
    await RoleService.update_role(db, redis, role_id, body.model_dump(exclude_none=True))

    return success(msg="更新成功")


@router.delete("/{ids}", response_model=Result[None], summary="删除角色")
@require_permission("sys:role:delete")
async def delete_roles(
    ids: str = Path(..., description="角色ID，多个以英文逗号(,)分隔"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await RoleService.delete_roles(db, redis, ids)

    return success(msg="删除成功")


@router.patch("/{role_id}/status", response_model=Result[None], summary="修改角色状态")
@require_permission("sys:role:edit")
async def update_role_status(
    role_id: int = Path(..., description="角色ID"),
    status: int = Query(..., ge=0, le=1, description="状态(1-启用；0-停用)"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await RoleService.update_role_status(db, redis, role_id, status)

    return success(msg="更新成功")


@router.get("/{role_id}/menuIds", response_model=Result[list[int]], summary="获取角色的菜单ID集合")
async def get_role_menu_ids(
    role_id: int = Path(..., description="角色ID"),
    db: AsyncSession = Depends(get_db),
):
    role = await RoleService.get_role_by_id(db, role_id)

    if not role:
        return success([])

    menu_ids = await RoleService.get_role_menu_ids(db, role_id)
    return success(menu_ids)


@router.patch("/{role_id}/menus", response_model=Result[None], summary="分配菜单给角色")
@require_permission("sys:role:edit")
async def assign_menus_to_role(
    role_id: int = Path(..., description="角色ID"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    body: MenuIdsBody = Body(...),
    user: UserContext = Depends(get_current_user),
):
    # RootModel 使用 .root 访问实际的列表数据
    menu_ids: list[int] = body.root
    await RoleService.assign_menus_to_role(db, redis, role_id, menu_ids)

    return success(msg="分配成功")
