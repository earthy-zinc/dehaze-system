import pytest
from sqlalchemy import select

from app.dependencies.auth import UserContext
from app.models.entity.sys_user import SysUser
from app.repository.data_scope import apply_data_scope


def _make_user(data_scope=None, roles=None, dept_id=10, user_id=100):
    return UserContext(
        id=user_id,
        username="tester",
        dept_id=dept_id,
        data_scope=data_scope,
        roles=roles or [],
        permissions=[],
    )


async def _compile_where_params(stmt):
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    return str(compiled)


async def test_root_user_skips_filter(monkeypatch):
    user = _make_user(data_scope=0, roles=["ROOT"])
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )

    assert result is stmt


async def test_data_scope_0_all_data():
    user = _make_user(data_scope=0)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )

    assert result is stmt


async def test_data_scope_3_self():
    user = _make_user(data_scope=3, user_id=100)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )
    sql = await _compile_where_params(result)

    assert "create_by" in sql
    assert "100" in sql


async def test_data_scope_2_dept():
    user = _make_user(data_scope=2, dept_id=10)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )
    sql = await _compile_where_params(result)

    assert "dept_id" in sql
    assert "10" in sql


async def test_data_scope_1_dept_tree():
    user = _make_user(data_scope=1, dept_id=10)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt,
        user,
        db=None,
        dept_field=SysUser.dept_id,
        creator_field=SysUser.create_by,
        children_ids=[10, 11, 12],
    )
    sql = await _compile_where_params(result)

    assert "dept_id" in sql
    assert "10" in sql
    assert "11" in sql
    assert "12" in sql


async def test_data_scope_1_without_children_ids_returns_empty():
    user = _make_user(data_scope=1, dept_id=10)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt,
        user,
        db=None,
        dept_field=SysUser.dept_id,
        creator_field=SysUser.create_by,
        children_ids=None,
    )
    sql = await _compile_where_params(result)

    assert "false" in sql.lower() or "0 = 1" in sql.lower() or "0 = false" in sql.lower()


async def test_data_scope_3_without_creator_field_raises():
    user = _make_user(data_scope=3)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    with pytest.raises(ValueError, match="creator_field"):
        await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=None)


async def test_data_scope_2_without_dept_field_raises():
    user = _make_user(data_scope=2)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    with pytest.raises(ValueError, match="dept_field"):
        await apply_data_scope(
            stmt, user, db=None, dept_field=None, creator_field=SysUser.create_by
        )


async def test_data_scope_none_treated_as_all():
    user = _make_user(data_scope=None)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )
    assert result is stmt


async def test_data_scope_2_no_dept_returns_empty():
    user = _make_user(data_scope=2, dept_id=None)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(
        stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by
    )
    sql = await _compile_where_params(result)

    assert "false" in sql.lower() or "0 = 1" in sql.lower() or "0 = false" in sql.lower()
