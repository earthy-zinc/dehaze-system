"""
字典服务

提供字典项和字典类型的 CRUD 功能，包含唯一性校验、删除约束检查和缓存支持

设计参照: dehaze-doc/docs/03-模块设计/基础模块/字典管理/后端实现.md
"""

import logging
from typing import Any

from redis.asyncio import Redis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_dict import SysDict, SysDictType
from app.repository.dict_repository import (dict_repository,
                                            dict_type_repository)
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 缓存 Key 前缀
DICT_OPTIONS_CACHE_PREFIX = "dict:options:"
# 缓存过期时间（秒）
DICT_OPTIONS_CACHE_TTL = CACHE_TTL_HOUR


class DictService:
    """字典服务（异步版本）"""

    @staticmethod
    async def get_dict_page(
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
        type_code: str | None = None,
    ) -> tuple[list, int]:
        """获取字典分页列表"""
        return await dict_repository.get_page(db, page, page_size, keywords, type_code)

    @staticmethod
    async def get_dict_form(db: AsyncSession, dict_id: int) -> dict[str, Any] | None:
        """获取字典表单数据"""
        return await dict_repository.get_form_by_id(db, dict_id)

    @staticmethod
    async def create_dict(db: AsyncSession, redis: Redis, data: dict[str, Any]) -> SysDict:
        """
        创建字典项

        业务规则:
        1. 检查类型编码是否存在
        2. 检查同一类型下键值是否唯一
        3. 创建成功后清除缓存
        """
        type_code = data.get("typeCode")
        value = data.get("value")

        if not type_code:
            raise BusinessException("字典类型编码不能为空")
        if not value:
            raise BusinessException("字典值不能为空")

        # 检查类型编码是否存在
        dict_type = await dict_type_repository.get_by_code(db, type_code)
        if not dict_type:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在")

        # 检查同一类型下键值唯一性
        existing = await dict_repository.get_by_type_code_and_value(db, type_code, value)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "该类型下字典值已存在")

        # 创建字典项
        result = await dict_repository.create_dict(db, data)

        # 清除缓存
        await CacheService(redis).delete(f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}")

        return result

    @staticmethod
    async def update_dict(db: AsyncSession, redis: Redis, dict_id: int, data: dict[str, Any]) -> bool:
        """
        更新字典项

        业务规则:
        1. 检查字典是否存在
        2. typeCode 只读，不可修改
        3. 如果修改了 value，检查唯一性
        4. 更新成功后清除相关缓存
        """
        # 获取原字典数据
        old_dict = await dict_repository.get_by_id(db, dict_id)
        if not old_dict:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典不存在")

        # typeCode 只读，移除 form 传入的 typeCode
        data.pop("typeCode", None)
        new_value = data.get("value", old_dict.value)

        # 如果修改了 value，检查唯一性（同类型下，排除自身）
        if new_value != old_dict.value:
            existing = await dict_repository.get_by_type_code_and_value(db, old_dict.type_code, new_value)
            if existing and existing.id != dict_id:
                raise BusinessException(ResultCode.DATA_EXISTS, "该类型下字典值已存在")

        # 更新字典
        result = await dict_repository.update_by_id(db, dict_id, data)

        # 清除缓存（typeCode 不变，只需清除一个）
        if old_dict.type_code:
            await CacheService(redis).delete(f"{DICT_OPTIONS_CACHE_PREFIX}{old_dict.type_code}")

        return result

    @staticmethod
    async def delete_dict(db: AsyncSession, redis: Redis, dict_ids: list[int]) -> bool:
        """
        删除字典项

        业务规则:
        1. 校验字典数据项是否存在
        2. 删除成功后清除相关缓存
        """
        # 校验字典数据项是否存在
        exist_count = await dict_repository.count_by_ids(db, dict_ids)
        if exist_count == 0:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        # 获取要删除的字典的 type_code 列表
        type_codes = await dict_repository.get_type_codes_by_ids(db, dict_ids)

        result = await dict_repository.delete_by_ids(db, dict_ids)

        # 清除相关缓存
        cache = CacheService(redis)
        for type_code in type_codes:
            if type_code:  # 确保 type_code 不为 None
                await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}")

        return result > 0  # 返回 bool 表示是否删除成功

    @staticmethod
    async def list_dict_options(db: AsyncSession, redis: Redis, type_code: str) -> list[dict[str, Any]]:
        """
        获取字典下拉列表

        使用缓存策略:
        1. 先查缓存
        2. 缓存未命中则查数据库
        3. 写入缓存
        """
        cache = CacheService(redis)
        cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}"

        # 尝试从缓存获取
        cached = await cache.get_json(cache_key)
        if cached is not None:
            return cached

        # 从数据库查询
        options = await dict_repository.list_options_by_type(db, type_code)

        # 写入缓存
        await cache.set_json(cache_key, options, DICT_OPTIONS_CACHE_TTL)

        return options


class DictTypeService:
    """字典类型服务（异步版本）"""

    @staticmethod
    async def get_dict_type_page(
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
    ) -> tuple[list, int]:
        """获取字典类型分页列表"""
        return await dict_type_repository.get_page(db, page, page_size, keywords)

    @staticmethod
    async def get_dict_type_form(db: AsyncSession, type_id: int) -> dict[str, Any] | None:
        """获取字典类型表单数据"""
        return await dict_type_repository.get_form_by_id(db, type_id)

    @staticmethod
    async def create_dict_type(db: AsyncSession, data: dict[str, Any]) -> SysDictType:
        """
        创建字典类型

        业务规则:
        1. 检查编码唯一性
        """
        code = data.get("code")

        if not code:
            raise BusinessException("字典类型编码不能为空")

        # 检查编码唯一性
        existing = await dict_type_repository.get_by_code(db, code)
        if existing:
            raise BusinessException(ResultCode.DATA_EXISTS, "字典类型编码已存在")

        result = await dict_type_repository.create_type(db, data)
        return result

    @staticmethod
    async def update_dict_type(db: AsyncSession, redis: Redis, type_id: int, data: dict[str, Any]) -> bool:
        """
        更新字典类型

        业务规则:
        1. 检查字典类型是否存在
        2. 如果修改了 code，检查唯一性
        3. code 变更时级联更新 sys_dict.type_code 并清除缓存
        """
        # 获取原类型
        old_type = await dict_type_repository.get_by_id(db, type_id)
        if not old_type:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在")

        new_code = data.get("code")

        # 如果修改了 code，检查唯一性
        if new_code and new_code != old_type.code:
            existing = await dict_type_repository.get_by_code(db, new_code)
            if existing:
                raise BusinessException(ResultCode.DATA_EXISTS, "字典类型编码已存在")

        # 更新字典类型
        result = await dict_type_repository.update_by_id(db, type_id, data)

        # code 变更时级联更新 sys_dict.type_code
        if result and new_code and new_code != old_type.code:
            await dict_repository.update_type_code(db, old_type.code, new_code)

        # 清除缓存
        if result and new_code and new_code != old_type.code:
            cache = CacheService(redis)
            await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{old_type.code}")
            await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{new_code}")

        return result

    @staticmethod
    async def delete_dict_types(db: AsyncSession, redis: Redis, type_ids: list[int], force: bool = False) -> bool:
        """
        删除字典类型

        业务规则:
        1. 校验字典类型是否存在
        2. force=False 时检查是否存在关联的字典数据，存在则禁止删除
        3. force=True 时级联删除关联的字典数据
        """
        # 校验字典类型是否存在
        exist_count = await dict_type_repository.count_by_ids(db, type_ids)
        if exist_count == 0:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND)

        # 批量查询类型编码，批量检查关联数据（避免 N+1）
        dict_types = await dict_type_repository.get_by_ids(db, type_ids)
        type_codes = [dt.code for dt in dict_types if dt.code]
        if type_codes:
            if force:
                await dict_repository.delete_by_type_codes(db, type_codes)
                cache = CacheService(redis)
                for code in type_codes:
                    await cache.delete(f"{DICT_OPTIONS_CACHE_PREFIX}{code}")
            else:
                counts = await dict_repository.count_by_type_codes(db, type_codes)
                for dt in dict_types:
                    if counts.get(dt.code, 0) > 0:
                        raise BusinessException(
                            ResultCode.DATA_BIND_EXISTS,
                            "存在关联的字典数据，无法删除")

        result = await dict_type_repository.delete_by_ids(db, type_ids)
        return result > 0
