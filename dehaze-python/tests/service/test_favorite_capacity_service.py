"""
收藏容量（favorite_capacity）字典化改造单元测试（真实 MySQL 测试库 + SAVEPOINT 回滚）。

覆盖：等级→字典键映射读取、字典键生效（运营调整后立即生效）、缺键回退设计默认值。
"""

import pytest

from app.repository.member_repository import member_repository
from app.service.dict_service import _invalidate_dict_value_cache
from app.service.favorite_service import CAPACITY_DEFAULTS, favorite_service

pytestmark = pytest.mark.requires_db

USER_ID = 1005001


async def _setup_member(db, *, level_code: str):
    member = await member_repository.get_or_init_member(db, USER_ID)
    member.level_code = level_code
    await db.flush()
    return member


async def _capacity(db, user_id: int) -> int:
    return await favorite_service._get_capacity(db, user_id)


class TestCapacityByLevel:
    async def test_level0_default_capacity(self, db):
        await _setup_member(db, level_code="level_0")
        assert await _capacity(db, USER_ID) == 200

    async def test_level1_vip1_capacity(self, db):
        await _setup_member(db, level_code="level_1")
        assert await _capacity(db, USER_ID) == 500

    async def test_level2_vip2_capacity(self, db):
        await _setup_member(db, level_code="level_2")
        assert await _capacity(db, USER_ID) == 1000

    async def test_level3_svip_capacity(self, db):
        await _setup_member(db, level_code="level_3")
        assert await _capacity(db, USER_ID) == 3000


class TestCapacityDictOverrides:
    async def test_dict_value_override_takes_effect(self, db, mock_redis):
        """运营调整 sys_dict 后容量即时生效（直接更新库中种子值）。"""
        from app.repository.dict_repository import dict_repository

        await _setup_member(db, level_code="level_1")
        item = await dict_repository.get_by_type_code_and_name(db, "favorite_capacity", "vip1")
        item.value = "888"
        await db.flush()
        # 模拟生产：运营更新字典后失效 dict:value 缓存（测试绕过 DictService 直改 DB）
        await _invalidate_dict_value_cache(mock_redis, "favorite_capacity")
        assert await _capacity(db, USER_ID) == 888

    async def test_missing_key_falls_back_to_default(self, db, mock_redis):
        """删除某级容量字典项后回退设计默认值，不抛异常。"""
        await _setup_member(db, level_code="level_2")
        from app.repository.dict_repository import dict_repository

        # 删除 vip2 键（SAVEPOINT 回滚，不影响其它测试）
        item = await dict_repository.get_by_type_code_and_name(db, "favorite_capacity", "vip2")
        await db.delete(item)
        await db.flush()
        await _invalidate_dict_value_cache(mock_redis, "favorite_capacity")
        assert await _capacity(db, USER_ID) == CAPACITY_DEFAULTS["vip2"]

    async def test_unknown_level_falls_back_to_default(self, db, mock_redis):
        """未知/缺失会员等级回落 default 键容量。"""
        await _setup_member(db, level_code="level_9")
        assert await _capacity(db, USER_ID) == 200


class TestCapacityDictSeedTypePreset:
    async def test_capacity_and_growth_type_are_preset(self, db):
        """收藏容量/成长值规则为预置字典类型（不可删除）。"""
        from app.repository.dict_repository import dict_type_repository

        assert await dict_type_repository.get_by_code(db, "favorite_capacity") is not None
        assert await dict_type_repository.get_by_code(db, "member_growth_rules") is not None
