from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import UserContext
from app.router.file import check_file, get_file_info

pytestmark = pytest.mark.api


def _admin() -> UserContext:
    """管理员用户上下文（is_admin 由 roles 派生，与生产 UserContext 一致）。"""
    return UserContext(id=1, username="admin", roles=["ADMIN"])


def _file(**overrides):
    base = {
        "id": 1,
        "name": "test.jpg",
        "type": "jpg",
        "size": "2.44MB",
        "size_bytes": 2560000,
        "object_name": "upload/20240101/abc.jpg",
        "storage": "minio",
        "md5": "a" * 32,
        "create_time": None,
        "update_time": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "bad_md5",
    [
        "invalid",
        "abc",
        "a" * 31,
        "a" * 33,
        "",
    ],
)
async def test_check_file_invalid_md5_rejected(bad_md5):
    with pytest.raises(BusinessException) as ei:
        await check_file(md5=bad_md5, db=AsyncSession())
    assert ei.value.code == ResultCode.FILE_MD5_INVALID
    assert "MD5格式无效" in ei.value.message


@pytest.mark.parametrize(
    "valid_md5",
    [
        "a" * 32,
        "0123456789abcdef" * 2,
        "ABCDEF0123456789abcdef0123456789",
    ],
)
async def test_check_file_valid_md5_returns_file(monkeypatch, valid_md5):
    async def _found(db, md5):
        return _file(md5=md5)

    monkeypatch.setattr("app.router.file.file_service.get_file_by_md5", _found)
    resp = await check_file(md5=valid_md5, db=AsyncSession())
    assert resp.code == ResultCode.SUCCESS.code
    assert resp.data is not None
    assert resp.data.sizeBytes == 2560000


async def test_get_file_info_not_found_rejected(monkeypatch):
    async def _none(db, file_id):
        return None

    monkeypatch.setattr("app.router.file.file_service.get_file_by_id", _none)
    admin = _admin()
    with pytest.raises(BusinessException) as ei:
        await get_file_info(file_id=999, db=AsyncSession(), user=admin)
    assert ei.value.code == ResultCode.FILE_NOT_FOUND
    assert "文件不存在" in ei.value.message


async def test_get_file_info_returns_size_bytes(monkeypatch):
    async def _found(db, file_id):
        return _file()

    monkeypatch.setattr("app.router.file.file_service.get_file_by_id", _found)
    admin = _admin()
    resp = await get_file_info(file_id=1, db=AsyncSession(), user=admin)
    assert resp.data is not None
    assert resp.data.sizeBytes == 2560000
    assert resp.data.size == "2.44MB"
