from typing import Optional

from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.package import (CouponBatchDistributeForm, CouponForm,
                                        CouponQuery, PackageForm, PackageQuery)
from app.service.coupon_service import CouponService
from app.service.package_service import PackageService

router = APIRouter(
    prefix="/api/v1/packages",
    tags=["套餐管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", summary="在售套餐列表")
async def list_on_sale(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.list_on_sale(db)
    return success(data)


@router.post("", summary="新增套餐")
@require_permission("package:add")
async def add_package(
    body: PackageForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await PackageService.create(db, body.model_dump(exclude_none=True))
    return success()


@router.get("/page", summary="套餐分页列表")
async def get_package_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    name: Optional[str] = Query(default=None),
    levelCode: Optional[str] = Query(default=None),
    period: Optional[str] = Query(default=None),
    status: Optional[int] = Query(default=None, ge=0, le=1),
    startTime: Optional[str] = Query(default=None),
    endTime: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.get_page(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "name": name,
            "levelCode": levelCode,
            "period": period,
            "status": status,
            "startTime": startTime,
            "endTime": endTime,
        },
    )
    return success(data)


@router.get("/calculate-price", summary="价格计算")
async def calculate_price(
    packageId: int = Query(...),
    userCouponId: Optional[int] = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.calculate_price(db, packageId, userCouponId, user.id)
    return success(data)


@router.get("/sales/stats", summary="销售统计")
async def get_sales_stats(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.get_sales_stats(db)
    return success(data)


@router.get("/coupons/my", summary="我的优惠券列表")
async def list_my_coupons(
    status: Optional[int] = Query(default=None, ge=1, le=4),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await CouponService.list_my(db, user.id, status)
    return success(data)


@router.post("/coupons", summary="创建优惠券")
@require_permission("package:coupon:add")
async def add_coupon(
    body: CouponForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await CouponService.create(db, body.model_dump(exclude_none=True))
    return success(data)


@router.post("/coupons/batch", summary="批量发放优惠券")
@require_permission("package:coupon:distribute")
async def batch_distribute_coupon(
    body: CouponBatchDistributeForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await CouponService.batch_distribute(db, body.model_dump(exclude_none=True))
    return success(data)


@router.get("/coupons/page", summary="优惠券分页列表")
async def get_coupon_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    name: Optional[str] = Query(default=None),
    type: Optional[str] = Query(default=None),
    status: Optional[int] = Query(default=None, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await CouponService.get_page(
        db,
        {
            "pageNum": pageNum,
            "pageSize": pageSize,
            "name": name,
            "type": type,
            "status": status,
        },
    )
    return success(data)


@router.post("/coupons/{coupon_id}/receive", summary="领取优惠券")
async def receive_coupon(
    coupon_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await CouponService.receive(db, coupon_id, user.id)
    return success(data)


@router.put("/coupons/{coupon_id}", summary="修改优惠券")
@require_permission("package:coupon:edit")
async def update_coupon(
    coupon_id: int = Path(...),
    body: CouponForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await CouponService.update(db, coupon_id, body.model_dump(exclude_none=True))
    return success()


@router.delete("/coupons/{ids}", summary="删除优惠券")
@require_permission("package:coupon:delete")
async def delete_coupons(
    ids: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    id_list = [int(i) for i in ids.split(",") if i.strip()]
    await CouponService.delete_by_ids(db, id_list)
    return success()


@router.get("/{package_id}", summary="套餐详情")
async def get_package_detail(
    package_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.get_detail(db, package_id)
    return success(data)


@router.put("/{package_id}", summary="修改套餐")
@require_permission("package:edit")
async def update_package(
    package_id: int = Path(...),
    body: PackageForm = Body(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await PackageService.update(db, package_id, body.model_dump(exclude_none=True))
    return success()


@router.get("/{package_id}/form", summary="获取套餐表单数据")
async def get_package_form(
    package_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    data = await PackageService.get_form(db, package_id)
    return success(data)


@router.put("/{package_id}/status", summary="上架/下架")
@require_permission("package:edit")
async def update_package_status(
    package_id: int = Path(...),
    status: int = Query(..., ge=0, le=1),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await PackageService.update_status(db, package_id, status)
    return success()


@router.delete("/{ids}", summary="删除套餐")
@require_permission("package:delete")
async def delete_packages(
    ids: str = Path(...),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    id_list = [int(i) for i in ids.split(",") if i.strip()]
    await PackageService.delete_by_ids(db, id_list)
    return success()
