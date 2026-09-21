"""字典服务层测试（真实 MySQL 测试库 + fakeredis）。

对应字典管理测试用例.md：T-DM-007/008/012（类型新增与编码唯一/状态默认）、
T-DM-015/017（code 只读、编辑不存在）、T-DM-020~025（删除与 force 级联、预置保护）、
T-DM-032/034/040（字典新增、同类型值唯一、defaulted 默认）、T-DM-042~044（字典编辑）、
T-DM-054/056/060/062（下拉可见性与类型/数据状态联动）、T-DM-064（变更后缓存立即失效）、
§8 缓存 TTL=1h、§9.2 唯一性含软删行。
另覆盖 ensure_system_dict_defaults 种子幂等与 get_dict_int 读路径缓存。

缓存失效经 defer_after_commit 登记，测试中以 run_after_commit_callbacks 模拟事务提交。
"""

import uuid

import pytest
from sqlalchemy import func, select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import run_after_commit_callbacks
from app.models.entity.sys_dict import SysDict, SysDictType
from app.repository.dict_repository import dict_repository, dict_type_repository
from app.service.dict_service import (
    DICT_OPTIONS_CACHE_PREFIX,
    DICT_VALUE_CACHE_PREFIX,
    SYSTEM_PRESET_DICT_TYPE_CODES,
    dict_service,
    dict_type_service,
    ensure_system_dict_defaults,
    get_dict_int,
)

pytestmark = pytest.mark.requires_db


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


def _type_data(**overrides) -> dict:
    data = {"name": "svc测试类型", "code": _uid("SVC_DT"), "status": 1, "remark": "svc备注"}
    data.update(overrides)
    return data


def _dict_data(type_code: str, **overrides) -> dict:
    data = {
        "typeCode": type_code,
        "name": _uid("svc字典"),
        "value": _uid("svc_val"),
        "sort": 1,
        "status": 1,
        "defaulted": 0,
        "remark": "svc字典备注",
    }
    data.update(overrides)
    return data


async def _create_type(db, mock_redis, **overrides):
    return await dict_type_service.create_dict_type(db, mock_redis, _type_data(**overrides))


# ===== 字典类型：新增 / 编辑 / 删除 =====


async def test_create_type_persists_fields(db, mock_redis):
    data = _type_data(remark="持久化备注")
    created = await dict_type_service.create_dict_type(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    assert created.id > 0
    form = await dict_type_service.get_dict_type_form(db, created.id)
    assert form is not None
    assert form["name"] == data["name"]
    assert form["code"] == data["code"]
    assert form["status"] == 1
    assert form["remark"] == "持久化备注"
    assert form["isPreset"] is False


async def test_preset_type_form_marks_is_preset(db, mock_redis):
    preset = await dict_type_repository.get_by_code(db, "gender")
    assert preset is not None
    form = await dict_type_service.get_dict_type_form(db, preset.id)
    assert form is not None
    assert form["isPreset"] is True


async def test_create_type_duplicate_code_rejected(db, mock_redis):
    data = _type_data()
    await dict_type_service.create_dict_type(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.create_dict_type(db, mock_redis, _type_data(code=data["code"]))
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_type_recreate_same_code_after_delete(db, mock_redis):
    """唯一键含 deleted：软删类型不占键位，删除后可重建同 code"""
    created = await _create_type(db, mock_redis, code=_uid("SOFT_CODE"))
    await dict_type_service.delete_dict_types(db, mock_redis, [created.id])
    await run_after_commit_callbacks(db)

    recreated = await _create_type(db, mock_redis, code=created.code)
    assert recreated.id != created.id
    assert recreated.deleted == 0


async def test_create_type_requires_code(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.create_dict_type(db, mock_redis, {"name": "无编码类型", "code": ""})
    assert "编码不能为空" in ei.value.message


async def test_update_type_persists_and_keeps_same_code(db, mock_redis):
    created = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    ok = await dict_type_service.update_dict_type(
        db,
        mock_redis,
        created.id,
        {"name": "更新名称", "code": created.code, "status": 0, "remark": "更新备注"},
    )
    await run_after_commit_callbacks(db)
    assert ok is True
    form = await dict_type_service.get_dict_type_form(db, created.id)
    assert form is not None
    assert form["name"] == "更新名称"
    assert form["status"] == 0
    assert form["code"] == created.code


async def test_update_type_rejects_code_change(db, mock_redis):
    created = await _create_type(db, mock_redis)
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.update_dict_type(
            db, mock_redis, created.id, {"name": created.name, "code": _uid("CHANGED")}
        )
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW


async def test_update_type_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.update_dict_type(db, mock_redis, 99999999, {"name": "任意"})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_delete_types_without_data(db, mock_redis):
    t1 = await _create_type(db, mock_redis)
    t2 = await _create_type(db, mock_redis)
    assert await dict_type_service.delete_dict_types(db, mock_redis, [t1.id, t2.id]) is True
    await run_after_commit_callbacks(db)
    assert await dict_type_service.get_dict_type_form(db, t1.id) is None
    assert await dict_type_service.get_dict_type_form(db, t2.id) is None


async def test_delete_types_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.delete_dict_types(db, mock_redis, [99999999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_delete_type_with_data_blocked_without_force(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.delete_dict_types(db, mock_redis, [created_type.id])
    assert ei.value.code == ResultCode.DATA_BIND_EXISTS
    assert await dict_type_service.get_dict_type_form(db, created_type.id) is not None


async def test_delete_type_force_cascades_dict_data(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    created = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)

    assert (
        await dict_type_service.delete_dict_types(db, mock_redis, [created_type.id], force=True)
        is True
    )
    await run_after_commit_callbacks(db)
    assert await dict_type_service.get_dict_type_form(db, created_type.id) is None
    assert await dict_service.get_dict_form(db, created.id) is None


async def test_delete_preset_type_rejected_even_with_force(db, mock_redis):
    """T-DM-025：系统预置类型（种子 gender）force=true 也不可删"""
    assert "gender" in SYSTEM_PRESET_DICT_TYPE_CODES
    preset = await dict_type_repository.get_by_code(db, "gender")
    assert preset is not None
    with pytest.raises(BusinessException) as ei:
        await dict_type_service.delete_dict_types(db, mock_redis, [preset.id], force=True)
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert await dict_type_service.get_dict_type_form(db, preset.id) is not None


# ===== 字典数据：新增 / 编辑 / 删除 =====


async def test_create_dict_persists_fields(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    data = _dict_data(created_type.code, sort=5, defaulted=1, status=1, remark="完整字段")
    created = await dict_service.create_dict(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    assert created.id > 0
    form = await dict_service.get_dict_form(db, created.id)
    assert form is not None
    assert form["typeCode"] == created_type.code
    assert form["value"] == data["value"]
    assert form["sort"] == 5
    assert form["defaulted"] == 1
    assert form["remark"] == "完整字段"


async def test_create_dict_type_not_found_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_service.create_dict(db, mock_redis, _dict_data(_uid("NO_TYPE")))
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_create_dict_duplicate_value_same_type_rejected(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    data = _dict_data(created_type.code)
    await dict_service.create_dict(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    with pytest.raises(BusinessException) as ei:
        await dict_service.create_dict(
            db, mock_redis, _dict_data(created_type.code, value=data["value"])
        )
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_dict_duplicate_name_same_type_rejected(db, mock_redis):
    """同类型下 name 重复（value 不同）→ A0501（uk_type_name 含软删行，服务层查重转业务异常）"""
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    data = _dict_data(created_type.code)
    await dict_service.create_dict(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    with pytest.raises(BusinessException) as ei:
        await dict_service.create_dict(
            db,
            mock_redis,
            _dict_data(created_type.code, name=data["name"], value=_uid("other_val")),
        )
    assert ei.value.code == ResultCode.DATA_EXISTS
    assert "名称已存在" in ei.value.message


async def test_create_dict_same_value_different_types_allowed(db, mock_redis):
    t1 = await _create_type(db, mock_redis)
    t2 = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    value = _uid("shared_val")
    d1 = await dict_service.create_dict(db, mock_redis, _dict_data(t1.code, value=value))
    d2 = await dict_service.create_dict(db, mock_redis, _dict_data(t2.code, value=value))
    assert d1.id != d2.id


async def test_update_dict_persists_and_typecode_readonly(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    created = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    ok = await dict_service.update_dict(
        db,
        mock_redis,
        created.id,
        {
            "typeCode": _uid("TRY_CHANGE"),
            "name": "改名后",
            "value": created.value,
            "sort": 9,
            "status": 0,
            "defaulted": 1,
        },
    )
    await run_after_commit_callbacks(db)
    assert ok is True
    form = await dict_service.get_dict_form(db, created.id)
    assert form is not None
    assert form["typeCode"] == created_type.code
    assert form["name"] == "改名后"
    assert form["sort"] == 9
    assert form["status"] == 0
    assert form["defaulted"] == 1


async def test_update_dict_value_conflict_rejected(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    d1 = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    d2 = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    with pytest.raises(BusinessException) as ei:
        await dict_service.update_dict(db, mock_redis, d2.id, {"value": d1.value, "name": d2.name})
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_update_dict_name_conflict_rejected(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    d1 = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    d2 = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    with pytest.raises(BusinessException) as ei:
        await dict_service.update_dict(db, mock_redis, d2.id, {"name": d1.name, "value": d2.value})
    assert ei.value.code == ResultCode.DATA_EXISTS
    assert "名称已存在" in ei.value.message


async def test_update_dict_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_service.update_dict(db, mock_redis, 99999999, {"name": "任意"})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_delete_dict_soft_and_unique_semantics(db, mock_redis):
    """逻辑删除口径：唯一键含 deleted，删除后同 type_code+name 可重建，value 可复用"""
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    data = _dict_data(created_type.code)
    created = await dict_service.create_dict(db, mock_redis, data)
    await run_after_commit_callbacks(db)

    assert await dict_service.delete_dict(db, mock_redis, [created.id]) is True
    await run_after_commit_callbacks(db)
    assert await dict_service.get_dict_form(db, created.id) is None

    recreated = await dict_service.create_dict(db, mock_redis, data)
    await run_after_commit_callbacks(db)
    assert recreated.id != created.id
    assert recreated.deleted == 0


async def test_delete_dict_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await dict_service.delete_dict(db, mock_redis, [99999999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 下拉选项可见性与排序（T-DM-054/056/060/062） =====


async def test_options_only_enabled_sorted_by_sort_asc(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    items = [
        _dict_data(created_type.code, name="低优先", value="v_low", sort=3),
        _dict_data(created_type.code, name="高优先", value="v_high", sort=1),
        _dict_data(created_type.code, name="中优先", value="v_mid", sort=2),
        _dict_data(created_type.code, name="禁用项", value="v_off", sort=0, status=0),
    ]
    for item in items:
        await dict_service.create_dict(db, mock_redis, item)
    await run_after_commit_callbacks(db)

    options = await dict_service.list_dict_options(db, mock_redis, created_type.code)
    assert [o["value"] for o in options] == ["v_high", "v_mid", "v_low"]
    assert {o["label"] for o in options} == {"高优先", "中优先", "低优先"}


async def test_options_exclude_disabled_type(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    assert len(await dict_service.list_dict_options(db, mock_redis, created_type.code)) == 1

    await dict_type_service.update_dict_type(
        db,
        mock_redis,
        created_type.id,
        {"name": created_type.name, "code": created_type.code, "status": 0},
    )
    await run_after_commit_callbacks(db)
    assert await dict_service.list_dict_options(db, mock_redis, created_type.code) == []


# ===== 缓存失效链路（T-DM-064 + 后端实现.md §5/§8） =====


async def test_options_cache_invalidated_on_dict_update(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    created = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{created_type.code}"

    first = await dict_service.list_dict_options(db, mock_redis, created_type.code)
    assert first
    assert await mock_redis.get(cache_key) is not None

    await dict_service.update_dict(db, mock_redis, created.id, {"name": "改名后"})
    await run_after_commit_callbacks(db)
    assert await mock_redis.get(cache_key) is None

    options = await dict_service.list_dict_options(db, mock_redis, created_type.code)
    assert [o["label"] for o in options] == ["改名后"]


async def test_options_cache_invalidated_on_dict_delete(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    created = await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{created_type.code}"
    await dict_service.list_dict_options(db, mock_redis, created_type.code)

    await dict_service.delete_dict(db, mock_redis, [created.id])
    await run_after_commit_callbacks(db)
    assert await mock_redis.get(cache_key) is None
    assert await dict_service.list_dict_options(db, mock_redis, created_type.code) == []


async def test_options_cache_invalidated_on_type_status_change(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    cache_key = f"{DICT_OPTIONS_CACHE_PREFIX}{created_type.code}"
    await dict_service.list_dict_options(db, mock_redis, created_type.code)
    assert await mock_redis.get(cache_key) is not None

    await dict_type_service.update_dict_type(
        db,
        mock_redis,
        created_type.id,
        {"name": created_type.name, "code": created_type.code, "status": 0},
    )
    await run_after_commit_callbacks(db)
    assert await mock_redis.get(cache_key) is None


async def test_options_cache_ttl_is_one_hour(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    await dict_service.create_dict(db, mock_redis, _dict_data(created_type.code))
    await run_after_commit_callbacks(db)
    await dict_service.list_dict_options(db, mock_redis, created_type.code)
    ttl = await mock_redis.ttl(f"{DICT_OPTIONS_CACHE_PREFIX}{created_type.code}")
    assert 3500 < ttl <= 3600


async def test_dict_value_cache_invalidated_on_update(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    created = await dict_service.create_dict(
        db, mock_redis, _dict_data(created_type.code, name="sign_in_value", value="3")
    )
    await run_after_commit_callbacks(db)
    cache_key = f"{DICT_VALUE_CACHE_PREFIX}{created_type.code}:sign_in_value"
    assert await get_dict_int(db, created_type.code, "sign_in_value", 99) == 3
    assert await mock_redis.get(cache_key) is not None

    await dict_service.update_dict(db, mock_redis, created.id, {"value": "8"})
    await run_after_commit_callbacks(db)
    assert await mock_redis.get(cache_key) is None
    assert await get_dict_int(db, created_type.code, "sign_in_value", 99) == 8


async def test_get_dict_int_missing_key_falls_back_to_default(db, mock_redis):
    assert await get_dict_int(db, _uid("NO_TYPE"), "any_key", 42) == 42


async def test_get_dict_int_non_numeric_value_falls_back_to_default(db, mock_redis):
    created_type = await _create_type(db, mock_redis)
    await run_after_commit_callbacks(db)
    await dict_service.create_dict(
        db, mock_redis, _dict_data(created_type.code, name="bad_num", value="not_a_number")
    )
    await run_after_commit_callbacks(db)
    assert await get_dict_int(db, created_type.code, "bad_num", 7) == 7


# ===== 种子幂等（ensure_system_dict_defaults） =====


async def _counts(db) -> tuple[int, int]:
    type_count = (await db.execute(select(func.count()).select_from(SysDictType))).scalar()
    item_count = (await db.execute(select(func.count()).select_from(SysDict))).scalar()
    return int(type_count), int(item_count)


async def test_ensure_system_dict_defaults_idempotent(db, mock_redis):
    await ensure_system_dict_defaults(db, mock_redis)
    type_count, item_count = await _counts(db)
    await ensure_system_dict_defaults(db, mock_redis)
    assert await _counts(db) == (type_count, item_count)


async def test_ensure_system_dict_defaults_fills_missing_item(db, mock_redis):
    existing = await dict_repository.get_by_type_code_and_name(db, "favorite_capacity", "svip")
    assert existing is not None
    await db.delete(existing)
    await db.flush()

    await ensure_system_dict_defaults(db, mock_redis)
    refilled = await dict_repository.get_by_type_code_and_name(db, "favorite_capacity", "svip")
    assert refilled is not None
    assert refilled.value == "3000"
