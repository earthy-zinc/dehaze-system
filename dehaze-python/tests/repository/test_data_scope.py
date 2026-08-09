"""
行级数据权限过滤助手单元测试

覆盖 5 种 data_scope 取值的过滤行为，与验收标准 §2.5 对齐：
- 0 全部数据 → 原样返回
- 1 部门及子部门 → WHERE dept_id IN (本部门及子部门)
- 2 本部门 → WHERE dept_id == 本部门
- 3 本人 → WHERE create_by == 当前用户
- ROOT 用户 → 跳过过滤
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import UserContext
from app.models.entity.sys_user import SysUser
from app.repository.data_scope import apply_data_scope


def _make_user(data_scope=None, roles=None, dept_id=10, user_id=100):
    """构造测试用 UserContext"""
    return UserContext(
        id=user_id,
        username="tester",
        dept_id=dept_id,
        data_scope=data_scope,
        roles=roles or [],
        permissions=[],
    )


async def _compile_where_params(stmt):
    """提取查询语句的 SQL 字符串（内联绑定参数），用于断言过滤条件"""
    compiled = stmt.compile(compile_kwargs={"literal_binds": True})
    return str(compiled)


@pytest.mark.asyncio
async def test_root_user_skips_filter(monkeypatch):
    """ROOT 用户跳过过滤"""
    user = _make_user(data_scope=0, roles=["ROOT"])
    stmt = select(SysUser).where(SysUser.deleted == 0)

    # dept_repository 不应被调用
    called = False

    async def _no_call(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("app.repository.data_scope.dept_repository.get_children_ids", _no_call)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)

    assert result is stmt
    assert not called


@pytest.mark.asyncio
async def test_data_scope_0_all_data(monkeypatch):
    """data_scope=0 全部数据，原样返回"""
    user = _make_user(data_scope=0)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)

    # 不追加任何条件
    assert result is stmt


@pytest.mark.asyncio
async def test_data_scope_3_self(monkeypatch):
    """data_scope=3 本人数据，追加 create_by == user_id"""
    user = _make_user(data_scope=3, user_id=100)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)
    sql = await _compile_where_params(result)

    assert "create_by" in sql
    assert "100" in sql


@pytest.mark.asyncio
async def test_data_scope_2_dept(monkeypatch):
    """data_scope=2 本部门数据，追加 dept_id == user.dept_id"""
    user = _make_user(data_scope=2, dept_id=10)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)
    sql = await _compile_where_params(result)

    assert "dept_id" in sql
    assert "10" in sql


@pytest.mark.asyncio
async def test_data_scope_1_dept_tree(monkeypatch):
    """data_scope=1 部门及子部门，通过 dept_repository 查子部门并 IN 过滤"""
    user = _make_user(data_scope=1, dept_id=10)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    async def _mock_children(db, dept_id):
        return [10, 11, 12]

    monkeypatch.setattr("app.repository.data_scope.dept_repository.get_children_ids", _mock_children)

    result = await apply_data_scope(stmt, user, db=object(), dept_field=SysUser.dept_id, creator_field=SysUser.create_by)
    sql = await _compile_where_params(result)

    assert "dept_id" in sql
    assert "10" in sql
    assert "11" in sql
    assert "12" in sql


@pytest.mark.asyncio
async def test_data_scope_3_without_creator_field_raises():
    """data_scope=3 但未提供 creator_field 时抛 ValueError"""
    user = _make_user(data_scope=3)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    with pytest.raises(ValueError, match="creator_field"):
        await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=None)


@pytest.mark.asyncio
async def test_data_scope_2_without_dept_field_raises():
    """data_scope=2 但未提供 dept_field 时抛 ValueError"""
    user = _make_user(data_scope=2)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    with pytest.raises(ValueError, match="dept_field"):
        await apply_data_scope(stmt, user, db=None, dept_field=None, creator_field=SysUser.create_by)


@pytest.mark.asyncio
async def test_data_scope_none_treated_as_all():
    """data_scope=None（未设置）视为全部数据"""
    user = _make_user(data_scope=None)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)
    assert result is stmt


@pytest.mark.asyncio
async def test_data_scope_2_no_dept_returns_empty(monkeypatch):
    """data_scope=2 但用户无部门，返回空集（WHERE false）"""
    user = _make_user(data_scope=2, dept_id=None)
    stmt = select(SysUser).where(SysUser.deleted == 0)

    result = await apply_data_scope(stmt, user, db=None, dept_field=SysUser.dept_id, creator_field=SysUser.create_by)
    sql = await _compile_where_params(result)

    # false 条件使结果为空
    assert "false" in sql.lower() or "0 = 1" in sql.lower() or "0 = false" in sql.lower()
