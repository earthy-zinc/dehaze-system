from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.promotion import PromotionForm, PromotionPackageForm
from app.service.promotion_service import promotion_service

router = APIRouter(
    prefix="/api/v1/packages/promotions",
    tags=["套餐管理-促销活动"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/page", summary="分页查询促销活动")
async def page_promotions(
    name: str | None = Query(default=None),
    type: str | None = Query(default=None),
    status: int | None = Query(default=None),
    startTime: str | None = Query(default=None),
    endTime: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await promotion_service.get_page(
        db,
        page=page,
        size=size,
        name=name,
        type=type,
        status=status,
        start_time=startTime,
        end_time=endTime,
    )
    return success(data)


@router.post("", summary="创建促销活动")
@require_permission("package:promotion:add")
async def add_promotion(
    body: PromotionForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await promotion_service.create(db, body)
    return success(data)


@router.put("/{promotion_id}", summary="修改促销活动")
@require_permission("package:promotion:edit")
async def update_promotion(
    promotion_id: int = Path(...),
    body: PromotionForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await promotion_service.update(db, promotion_id, body)
    return success(data)


@router.put("/{promotion_id}/status", summary="上架/下架促销活动")
@require_permission("package:promotion:edit")
async def update_promotion_status(
    promotion_id: int = Path(...),
    status: int = Query(..., ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await promotion_service.update_status(db, promotion_id, status)
    return success(data)


@router.delete("/{promotion_id}", summary="删除促销活动")
@require_permission("package:promotion:delete")
async def delete_promotion(
    promotion_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await promotion_service.delete(db, promotion_id)
    return success(None)


@router.put("/{promotion_id}/packages", summary="绑定促销活动套餐")
@require_permission("package:promotion:edit")
async def bind_promotion_packages(
    promotion_id: int = Path(...),
    body: PromotionPackageForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await promotion_service.bind_packages(db, promotion_id, body)
    return success(None)
