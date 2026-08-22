"""参数预设服务"""

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import async_session_factory
from app.models.entity.sys_preset import SysPreset
from app.models.schema.common import PageResult
from app.models.schema.preset import PresetForm, PresetVO
from app.repository.preset_repository import preset_repository

logger = logging.getLogger(__name__)

TYPE_SYSTEM = "system"
TYPE_CUSTOM = "custom"

# 系统预设种子数据
_SYSTEM_PRESET_SEEDS = [
    {
        "name": "默认去雾",
        "algorithm_id": 13,
        "params": {"gamma": 1.0, "strength": "medium"},
        "is_default": 1,
    },
    {
        "name": "轻度去雾",
        "algorithm_id": 13,
        "params": {"gamma": 0.8, "strength": "light"},
        "is_default": 0,
    },
    {
        "name": "深度去雾",
        "algorithm_id": 13,
        "params": {"gamma": 1.5, "strength": "strong"},
        "is_default": 0,
    },
]


def _to_vo(entity: SysPreset) -> PresetVO:
    return PresetVO(
        id=entity.id,
        name=entity.name,
        type=entity.type,
        algorithmId=entity.algorithm_id,
        params=entity.params,
        userId=entity.user_id,
        isDefault=entity.is_default,
        createTime=entity.create_time,
    )


class PresetService:
    @staticmethod
    async def seed_system_presets() -> None:
        """初始化系统预设种子数据（幂等：已有数据则跳过）"""
        try:
            async with async_session_factory() as db:
                count = await preset_repository.count_system_presets(db)
                if count > 0:
                    logger.debug("系统预设已存在 (%d 条)，跳过种子初始化", count)
                    return

                for seed in _SYSTEM_PRESET_SEEDS:
                    preset = SysPreset(
                        name=seed["name"],
                        type=TYPE_SYSTEM,
                        algorithm_id=seed["algorithm_id"],
                        params=seed["params"],
                        user_id=None,
                        is_default=seed["is_default"],
                    )
                    db.add(preset)

                await db.commit()
                logger.info("系统预设种子数据初始化完成 (%d 条)", len(_SYSTEM_PRESET_SEEDS))
        except Exception as e:
            logger.warning("系统预设种子初始化失败（非致命）: %s", e)

    @staticmethod
    async def list_presets(
        db: AsyncSession,
        user_id: int,
        algorithm_id: int | None = None,
        is_system: bool | None = None,
        page: int = 1,
        size: int = 10,
    ) -> PageResult[PresetVO]:
        presets, total = await preset_repository.list_presets(
            db,
            user_id,
            algorithm_id,
            is_system=is_system,
            page=page,
            size=size,
        )
        items = [_to_vo(p) for p in presets]
        return PageResult(list=items, total=total)

    @staticmethod
    async def create_preset(db: AsyncSession, user_id: int, form: PresetForm) -> PresetVO:
        preset = SysPreset(
            name=form.name,
            type=TYPE_CUSTOM,
            algorithm_id=form.algorithmId,
            params=form.params,
            user_id=user_id,
            is_default=form.isDefault or 0,
        )
        try:
            preset = await preset_repository.create(db, preset)
        except IntegrityError:
            # 唯一键 uk_user_name 冲突（同名预设）→ 业务错误 A0501，不落库为 C0300
            raise BusinessException(ResultCode.DATA_EXISTS, "预设名称已存在") from None
        return _to_vo(preset)

    @staticmethod
    async def update_preset(
        db: AsyncSession, preset_id: int, user_id: int, form: PresetForm
    ) -> PresetVO:
        preset = await preset_repository.get_by_id(db, preset_id)
        if not preset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预设不存在")
        if preset.type == TYPE_SYSTEM:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "系统预设不可修改")
        if preset.user_id != user_id:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "不能操作他人的预设")

        preset.name = form.name
        preset.algorithm_id = form.algorithmId
        preset.params = form.params
        if form.isDefault is not None:
            preset.is_default = form.isDefault
        await db.flush()
        await db.refresh(preset)
        return _to_vo(preset)

    @staticmethod
    async def delete_preset(db: AsyncSession, preset_id: int, user_id: int) -> None:
        preset = await preset_repository.get_by_id(db, preset_id)
        if not preset:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预设不存在")
        if preset.type == TYPE_SYSTEM:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "系统预设不可删除")
        if preset.user_id != user_id:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "不能操作他人的预设")

        await db.delete(preset)
        await db.flush()


preset_service = PresetService()
