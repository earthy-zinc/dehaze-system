"""文件路由层测试：下载路径遍历防护、Content-Type 推断、上传大小校验、归属校验（B0407）"""

import io
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

import app.router.file as file_router
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.auth import UserContext
from app.models.entity.sys_file import SysFile
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api


def _db() -> AsyncSession:
    """无绑定真实会话：DB 交互全部由打桩的 file_service 承接，此处仅满足端点 db 契约。"""
    return AsyncSession()


def _file(**overrides):
    base = {
        "id": 1,
        "name": "测试图片.png",
        "type": "png",
        "size": "1B",
        "size_bytes": 1,
        "object_name": "upload/20240101/abc.png",
        "storage": "minio",
        "md5": "a" * 32,
        "create_by": 100,
        "create_time": None,
        "update_time": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _user(user_id=100, is_admin=False) -> UserContext:
    return make_user_context(
        id=user_id, username=f"u{user_id}", roles=["ADMIN"] if is_admin else []
    )


# ===== 归属校验：普通用户仅可见/可操作自己上传的文件，管理员全量（越权 B0407） =====


async def test_detail_denies_other_user_file(monkeypatch):
    async def _found(db, file_id):
        return _file(create_by=100)

    monkeypatch.setattr(file_router.file_service, "get_file_by_id", _found)
    with pytest.raises(BusinessException) as ei:
        await file_router.get_file_info(file_id=1, db=_db(), user=_user(user_id=200))
    assert ei.value.code == ResultCode.FILE_ACCESS_DENIED


async def test_detail_allows_owner_and_admin(monkeypatch):
    async def _found(db, file_id):
        return _file(create_by=100)

    monkeypatch.setattr(file_router.file_service, "get_file_by_id", _found)

    owner_resp = await file_router.get_file_info(file_id=1, db=_db(), user=_user(user_id=100))
    assert owner_resp.data is not None
    assert owner_resp.data.id == 1

    admin_resp = await file_router.get_file_info(
        file_id=1, db=_db(), user=_user(user_id=200, is_admin=True)
    )
    assert admin_resp.data is not None
    assert admin_resp.data.id == 1


async def test_delete_denies_other_user_file(monkeypatch):
    async def _found(db, file_id):
        return _file(create_by=100)

    deleted = {}

    async def _delete(db, file_id):
        deleted["id"] = file_id

    monkeypatch.setattr(file_router.file_service, "get_file_by_id", _found)
    monkeypatch.setattr(file_router.file_service, "delete_file_with_storage", _delete)

    with pytest.raises(BusinessException) as ei:
        await file_router.delete_file(fileId=1, db=_db(), user=_user(user_id=200))
    assert ei.value.code == ResultCode.FILE_ACCESS_DENIED
    assert "id" not in deleted

    # 管理员可删除他人文件
    await file_router.delete_file(fileId=1, db=_db(), user=_user(user_id=200, is_admin=True))
    assert deleted["id"] == 1


async def test_download_denies_other_user_file(monkeypatch):
    async def _found(db, object_name):
        return _file(create_by=100)

    monkeypatch.setattr(file_router.file_service, "get_file_by_object_name", _found)
    with pytest.raises(BusinessException) as ei:
        await file_router.download_file(
            object_name="upload/20240101/abc.png", db=_db(), user=_user(user_id=200)
        )
    assert ei.value.code == ResultCode.FILE_ACCESS_DENIED


async def test_page_filters_by_owner_for_plain_user(db):
    """普通用户分页仅返回自己上传的文件；管理员全量"""
    mine = SysFile(
        type="txt",
        name="mine_page.txt",
        object_name=f"upload/test/{uuid.uuid4().hex}.txt",
        storage="minio",
        size="1B",
        size_bytes=1,
        md5=uuid.uuid4().hex,
        create_by=100,
    )
    others = SysFile(
        type="txt",
        name="other_page.txt",
        object_name=f"upload/test/{uuid.uuid4().hex}.txt",
        storage="minio",
        size="1B",
        size_bytes=1,
        md5=uuid.uuid4().hex,
        create_by=999,
    )
    db.add_all([mine, others])
    await db.flush()

    # 普通用户视角（owner_id=100）：仅见自己上传的文件
    items, _ = await file_router.file_service.get_file_page(db, 1, 50, "page", owner_id=100)
    ids = {f.id for f in items}
    assert mine.id in ids
    assert others.id not in ids

    # 管理员视角（owner_id=None）全量可见
    items, _ = await file_router.file_service.get_file_page(db, 1, 50, "page", owner_id=None)
    ids = {f.id for f in items}
    assert mine.id in ids
    assert others.id in ids


async def test_check_file_has_no_ownership_check(monkeypatch):
    """check_file 秒传预检不做归属校验：持有相同内容（md5 命中）即可获知文件已存在，无保密性泄露"""

    async def _found(db, md5):
        return _file(create_by=999)

    monkeypatch.setattr(file_router.file_service, "get_file_by_md5", _found)
    resp = await file_router.check_file(md5="a" * 32, db=_db())
    assert resp.data is not None
    assert resp.data.id == 1


# ===== 下载：路径遍历防护 =====


@pytest.mark.parametrize(
    "bad_path",
    [
        "../etc/passwd",
        "upload/../../etc/passwd",
        "/absolute/path",
        "\\windows\\path",
        "upload/..\\secret",
    ],
)
async def test_download_rejects_path_traversal(bad_path):
    with pytest.raises(HTTPException) as ei:
        await file_router.download_file(object_name=bad_path, db=_db())
    assert ei.value.status_code == 400


async def test_download_not_found_raises_b0401(monkeypatch):
    async def _none(db, object_name):
        return None

    monkeypatch.setattr(file_router.file_service, "get_file_by_object_name", _none)
    with pytest.raises(HTTPException) as ei:
        await file_router.download_file(object_name="upload/x/missing.txt", db=_db())
    assert ei.value.status_code == 404


# ===== 下载：按扩展名推断 Content-Type + 中文文件名 RFC 5987 编码 =====


async def test_download_sets_media_type_by_extension(monkeypatch):
    async def _found(db, object_name):
        return _file(name="photo.png", type="png")

    async def _stream(object_name, storage="minio"):
        yield b"binary"

    monkeypatch.setattr(file_router.file_service, "get_file_by_object_name", _found)
    monkeypatch.setattr(file_router.file_service, "download_file_stream", _stream)

    resp = await file_router.download_file(
        object_name="upload/20240101/abc.png", db=_db(), user=_user(user_id=100)
    )
    assert resp.media_type == "image/png"
    disposition = resp.headers["content-disposition"]
    assert "attachment" in disposition
    # RFC 5987：中文文件名以 filename*=UTF-8'' 编码携带
    assert "filename*=UTF-8''" in disposition


async def test_download_unknown_extension_falls_back_octet_stream(monkeypatch):
    async def _found(db, object_name):
        return _file(name="blob.xyzunknown", type="xyzunknown")

    async def _stream(object_name, storage="minio"):
        yield b"binary"

    monkeypatch.setattr(file_router.file_service, "get_file_by_object_name", _found)
    monkeypatch.setattr(file_router.file_service, "download_file_stream", _stream)

    resp = await file_router.download_file(
        object_name="upload/x/a.xyzunknown", db=_db(), user=_user(user_id=100)
    )
    assert resp.media_type == "application/octet-stream"


# ===== 上传：大小校验（B0402）=====


def _upload_file(content: bytes, filename: str) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(file_router.settings, "MAX_UPLOAD_SIZE", 10)
    with pytest.raises(BusinessException) as ei:
        await file_router.upload_file(file=_upload_file(b"x" * 20, "big.txt"), db=_db())
    assert ei.value.code == ResultCode.FILE_TOO_LARGE


async def test_upload_rejects_missing_filename():
    with pytest.raises(BusinessException) as ei:
        await file_router.upload_file(file=_upload_file(b"data", filename=""), db=_db())
    assert ei.value.code == ResultCode.PARAM_ERROR
