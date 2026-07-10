"""
字典服务

提供字典项和字典类型的 CRUD 功能，包含唯一性校验、删除约束检查和缓存支持

设计参照: dehaze-doc/docs/03-模块设计/基础模块/字典管理/后端实现.md
"""

import logging
from typing import Any

from app.core.exceptions import BusinessException
from app.models.entity.sys_dict import SysDict, SysDictType
from app.repository.dict_repository import (dict_repository,
                                            dict_type_repository)
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# 缓存 Key 前缀
DICT_OPTIONS_CACHE_PREFIX = "dict:options:"
# 缓存过期时间（秒）
DICT_OPTIONS_CACHE_TTL = 3600  # 1小时


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
    async def create_dict(db: AsyncSession, data: dict[str, Any]) -> SysDict:
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
            raise BusinessException("字典类型不存在")

        # 检查同一类型下键值唯一性
        existing = await dict_repository.get_by_type_code_and_value(db, type_code, value)
        if existing:
            raise BusinessException("该类型下字典值已存在")

        # 创建字典项
        result = await dict_repository.create_dict(db, data)

        # 清除缓存
        await DictService._invalidate_options_cache(type_code)

        return result

    @staticmethod
    async def update_dict(db: AsyncSession, dict_id: int, data: dict[str, Any]) -> bool:
        """
        更新字典项

        业务规则:
        1. 检查字典是否存在
        2. 如果修改了 typeCode 或 value，检查唯一性
        3. 更新成功后清除相关缓存
        """
        # 获取原字典数据
        old_dict = await dict_repository.get_by_id(db, dict_id)
        if not old_dict:
            raise BusinessException("字典不存在")

        new_type_code = data.get("typeCode", old_dict.type_code)
        new_value = data.get("value", old_dict.value)

        # 如果修改了 typeCode 或 value，检查唯一性
        if new_type_code != old_dict.type_code or new_value != old_dict.value:
            existing = await dict_repository.get_by_type_code_and_value(db, new_type_code, new_value)
            if existing and existing.id != dict_id:
                raise BusinessException("该类型下字典值已存在")

        # 更新字典
        result = await dict_repository.update_by_id(db, dict_id, data)

        # 清除缓存（新旧 typeCode 都需要清除）
        if old_dict.type_code:
            await DictService._invalidate_options_cache(old_dict.type_code)
        if new_type_code and new_type_code != old_dict.type_code:
            await DictService._invalidate_options_cache(new_type_code)

        return result

    @staticmethod
    async def delete_dict(db: AsyncSession, dict_ids: list[int]) -> bool:
        """
        删除字典项

        删除成功后清除相关缓存
        """
        # 获取要删除的字典的 type_code 列表
        type_codes = await dict_repository.get_type_codes_by_ids(db, dict_ids)

        result = await dict_repository.delete_by_ids(db, dict_ids)

        # 清除相关缓存
        for type_code in type_codes:
            if type_code:  # 确保 type_code 不为 None
                await DictService._invalidate_options_cache(type_code)

        return result > 0  # 返回 bool 表示是否删除成功

    @staticmethod
    async def list_dict_options(db: AsyncSession, type_code: str) -> list[dict[str, Any]]:
        """
        获取字典下拉列表

        使用缓存策略:
        1. 先查缓存
        2. 缓存未命中则查数据库
        3. 写入缓存
        """
        # 尝试从缓存获取
        cached = await DictService._get_options_from_cache(type_code)
        if cached is not None:
            return cached

        # 从数据库查询
        options = await dict_repository.list_options_by_type(db, type_code)

        # 写入缓存
        await DictService._set_options_to_cache(type_code, options)

        return options

    @staticmethod
    async def _get_options_from_cache(type_code: str) -> list[dict] | None:
        """从缓存获取字典选项"""
        try:
            from app.dependencies.redis import get_redis_client
            redis = await get_redis_client()
            if redis:
                import json
                cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}"
                cached = await redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"获取字典缓存失败: {e}")
        return None

    @staticmethod
    async def _set_options_to_cache(type_code: str, options: list[dict]) -> None:
        """写入字典选项缓存"""
        try:
            from app.dependencies.redis import get_redis_client
            redis = await get_redis_client()
            if redis:
                import json
                cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}"
                await redis.setex(cache_key, DICT_OPTIONS_CACHE_TTL, json.dumps(options, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"写入字典缓存失败: {e}")

    @staticmethod
    async def _invalidate_options_cache(type_code: str) -> None:
        """清除字典选项缓存"""
        try:
            from app.dependencies.redis import get_redis_client
            redis = await get_redis_client()
            if redis:
                cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{type_code}"
                await redis.delete(cache_key)
        except Exception as e:
            logger.warning(f"清除字典缓存失败: {e}")


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
            raise BusinessException("字典类型编码已存在")

        return await dict_type_repository.create_type(db, data)

    @staticmethod
    async def update_dict_type(db: AsyncSession, type_id: int, data: dict[str, Any]) -> bool:
        """
        更新字典类型

        业务规则:
        1. 检查字典类型是否存在
        2. 如果修改了 code，检查唯一性
        """
        # 获取原类型
        old_type = await dict_type_repository.get_by_id(db, type_id)
        if not old_type:
            raise BusinessException("字典类型不存在")

        new_code = data.get("code")

        # 如果修改了 code，检查唯一性
        if new_code and new_code != old_type.code:
            existing = await dict_type_repository.get_by_code(db, new_code)
            if existing:
                raise BusinessException("字典类型编码已存在")

        return await dict_type_repository.update_by_id(db, type_id, data)

    @staticmethod
    async def delete_dict_types(db: AsyncSession, type_ids: list[int]) -> bool:
        """
        删除字典类型

        业务规则:
        1. 检查是否存在关联的字典数据
        2. 存在关联数据则禁止删除
        """
        # 检查每个类型是否存在关联数据
        for type_id in type_ids:
            dict_type = await dict_type_repository.get_by_id(db, type_id)
            if dict_type:
                count = await dict_repository.count_by_type_code(db, dict_type.code)
                if count > 0:
                    raise BusinessException(
                        f"字典类型【{dict_type.name}】存在 {count} 条关联数据，无法删除")

        return await dict_type_repository.delete_by_ids(db, type_ids)
