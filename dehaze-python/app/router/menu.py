from typing import Optional

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.menu import (MenuForm, MenuFormVO, MenuOptionVO,
                                    MenuVisibleBody, MenuVO, RouteVO)
from app.service.menu_service import MenuService
from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/menus",
                   tags=["菜单管理"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=Result[list[MenuVO]], summary="获取菜单列表")
async def list_menus(
    keywords: Optional[str] = Query(default=None, description="关键词(菜单名称)"),
    db: AsyncSession = Depends(get_db),
):
    menu_list = await MenuService.list_menus(db, keywords)
    return success(menu_list)


@router.get("/options", response_model=Result[list[MenuOptionVO]], summary="获取菜单下拉选项")
async def list_menu_options(
    db: AsyncSession = Depends(get_db),
):
    options = await MenuService.list_menu_options(db)
    return success(options)


@router.get(
    "/routes",
    response_model=Result[list[RouteVO]],
    summary="获取路由列表",
    description="用于前端路由注册，带缓存",
)
async def list_routes(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    route_list = await MenuService.list_routes(db, redis)
    return success(route_list)


@router.get("/{menu_id}/form", response_model=Result[MenuFormVO], summary="获取菜单表单数据")
async def get_menu_form(
    menu_id: int = Path(..., description="菜单ID"),
    db: AsyncSession = Depends(get_db),
):
    menu_form = await MenuService.get_menu_form(db, menu_id)
    return success(menu_form)


@router.post("", response_model=Result[dict[str, int]], summary="新增菜单")
@require_permission("sys:menu:add")
async def add_menu(
    body: MenuForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await MenuService.save_menu(db, redis, body.model_dump(exclude_none=True))
    return success(msg="保存成功")


@router.put("/{menu_id}", response_model=Result[dict[str, int]], summary="修改菜单")
@require_permission("sys:menu:edit")
async def update_menu(
    menu_id: int = Path(..., description="菜单ID"),
    body: MenuForm = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    data = body.model_dump(exclude_none=True)
    data["id"] = menu_id
    await MenuService.save_menu(db, redis, data)
    return success(msg="保存成功")


@router.delete("/{ids}", response_model=Result[None], summary="删除菜单", description="级联删除子菜单和角色关联，支持批量删除")
@require_permission("sys:menu:delete")
async def delete_menu(
    ids: str = Path(..., description="菜单ID，多个以英文逗号(,)分割"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    menu_ids = [int(i) for i in ids.split(",")]
    await MenuService.delete_menu(db, redis, menu_ids)
    return success(msg="删除成功")


@router.patch("/{menu_id}", response_model=Result[None], summary="修改菜单显示状态")
@require_permission("sys:menu:edit")
async def update_menu_visible(
    menu_id: int = Path(..., description="菜单ID"),
    body: MenuVisibleBody = Body(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await MenuService.update_menu_visible(db, redis, menu_id, body.visible)
    return success(msg="更新成功")
