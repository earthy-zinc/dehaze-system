import json
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_promotion import SysPromotion, SysPromotionPackage
from app.models.schema.promotion import PromotionForm, PromotionPackageForm
from app.repository.promotion_repository import promotion_repository
from app.service.package_service import _format_dt

ONSALE_KEYS = ["package:onsale:all", "package:onsale:vip", "package:onsale:credit"]
DETAIL_KEY_PREFIX = "package:detail:"
DT_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, DT_FORMAT)


def _promotion_to_vo(promotion: SysPromotion) -> dict:
    return {
        "id": promotion.id,
        "name": promotion.name,
        "type": promotion.type,
        "description": promotion.description,
        "startTime": _format_dt(promotion.start_time),
        "endTime": _format_dt(promotion.end_time),
        "activityRules": promotion.activity_rules,
        "newUserOnly": promotion.new_user_only,
        "status": promotion.status,
        "createTime": _format_dt(promotion.create_time),
    }


async def _invalidate_package_cache(package_ids: list[int]) -> None:
    keys = ONSALE_KEYS + [f"{DETAIL_KEY_PREFIX}{pid}" for pid in package_ids]

    async def _del():
        redis = await get_redis_client()
        await redis.delete(*keys)

    await redis_operation_with_fallback(
        _del, default=None, operation_name="promotion_package_cache_invalidate"
    )


class PromotionService:
    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        name: str | None = None,
        type: str | None = None,
        status: int | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict:
        rows, total = await promotion_repository.get_page(
            db,
            page=page,
            size=size,
            name=name,
            type=type,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )
        return {
            "list": [_promotion_to_vo(p) for p in rows],
            "total": total,
        }

    async def create(self, db: AsyncSession, form: PromotionForm) -> dict:
        promotion = SysPromotion(
            name=form.name,
            type=form.type,
            description=form.description,
            start_time=_parse_dt(form.startTime),
            end_time=_parse_dt(form.endTime),
            activity_rules=form.activityRules,
            new_user_only=form.newUserOnly,
            status=form.status if getattr(form, "status", None) is not None else 0,
        )
        created = await promotion_repository.create(db, promotion)
        return _promotion_to_vo(created)

    async def update(
        self, db: AsyncSession, promotion_id: int, form: PromotionForm
    ) -> dict:
        existing = await promotion_repository.get_by_id(db, promotion_id)
        if not existing:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在")
        data = {}
        if getattr(form, "startTime", None) is not None:
            data["start_time"] = _parse_dt(form.startTime)
        if getattr(form, "endTime", None) is not None:
            data["end_time"] = _parse_dt(form.endTime)
        for field, attr in (
            ("name", "name"),
            ("type", "type"),
            ("description", "description"),
            ("activityRules", "activity_rules"),
            ("newUserOnly", "new_user_only"),
        ):
            value = getattr(form, field, None)
            if value is not None:
                data[attr] = value
        await promotion_repository.update(db, promotion_id, data)
        return _promotion_to_vo(existing)

    async def update_status(
        self, db: AsyncSession, promotion_id: int, status: int
    ) -> dict:
        existing = await promotion_repository.get_by_id(db, promotion_id)
        if not existing:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在")
        await promotion_repository.update(db, promotion_id, {"status": status})
        package_ids = await promotion_repository.list_package_ids_by_promotion(
            db, promotion_id
        )
        await _invalidate_package_cache(package_ids)
        return _promotion_to_vo(existing)

    async def delete(self, db: AsyncSession, promotion_id: int) -> None:
        existing = await promotion_repository.get_by_id(db, promotion_id)
        if not existing:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在")
        package_ids = await promotion_repository.list_package_ids_by_promotion(
            db, promotion_id
        )
        await promotion_repository.soft_delete(db, promotion_id)
        await promotion_repository.delete_packages_by_promotion(db, promotion_id)
        await _invalidate_package_cache(package_ids)

    async def bind_packages(
        self,
        db: AsyncSession,
        promotion_id: int,
        form: PromotionPackageForm,
    ) -> None:
        existing = await promotion_repository.get_by_id(db, promotion_id)
        if not existing:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "促销活动不存在")
        rules = existing.activity_rules or {}
        discount_type = rules.get("discount_type")
        if discount_type not in ("percent", "fixed", "full_reduction"):
            discount_type = "full_reduction" if existing.type == "full_reduction" else "percent"
        discount_value = rules.get("discount_value", 0)
        packages = [
            SysPromotionPackage(
                promotion_id=promotion_id,
                package_id=pid,
                discount_type=discount_type,
                discount_value=discount_value,
            )
            for pid in form.packageIds
        ]
        await promotion_repository.bind_packages(db, promotion_id, packages)
        await _invalidate_package_cache(form.packageIds)


promotion_service = PromotionService()
