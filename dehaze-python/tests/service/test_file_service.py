"""文件服务单元测试（对抗性文件名语料 + 本地存储真实读写 + MD5 去重/复活 + 孤儿清理阈值）"""

import os
import time
import uuid

import pytest
from sqlalchemy import select

import app.service.file_service as fs
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_file import SysFile
from app.service.storage.local_storage import LocalStorageService

pytestmark = pytest.mark.requires_db


@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    """本地存储后端注入（tmp_path 真实读写）：上传走 get_storage_service，
    下载/删除按 sys_file.storage 走 get_storage_by_name，两条路径都需指向 tmp 实例"""
    storage = LocalStorageService(base_dir=str(tmp_path))
    monkeypatch.setattr(fs, "get_storage_service", lambda: storage)
    monkeypatch.setattr(fs, "get_storage_by_name", lambda name: storage)
    return storage


# ===== sanitize_filename 对抗性文件名校验 =====


@pytest.mark.parametrize(
    "bad_name",
    [
        "file\x00.txt",  # 空字节
        "file\n.txt",  # CRLF/控制字符
        "file\r.txt",
        'file"quote.txt',  # 双引号
        "file<js>.txt",  # 尖括号
        "file|pipe.txt",  # 管道符
        "a" * 101 + ".txt",  # 超过 name 列宽 100
        "file." + "x" * 21,  # 扩展名超过 20
    ],
)
def test_sanitize_filename_rejects_unsafe(bad_name):
    with pytest.raises(BusinessException) as ei:
        fs.sanitize_filename(bad_name)
    assert ei.value.code == ResultCode.PARAM_ERROR


@pytest.mark.parametrize(
    "bad_name",
    ["", "   ", None],
)
def test_sanitize_filename_rejects_empty(bad_name):
    with pytest.raises(BusinessException) as ei:
        fs.sanitize_filename(bad_name)
    assert ei.value.code == ResultCode.PARAM_ERROR


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        # 路径遍历：先 basename 化，只保留文件名部分（name 仅作元数据，不参与存储定位）
        ("../../etc/passwd", "passwd"),
        ("..\\..\\win.ini", "win.ini"),
        ("dir/sub/report.txt", "report.txt"),
        # 零宽字符/emoji/中文：合法字符，允许保存
        ("隐\u200b藏.txt", "隐\u200b藏.txt"),
        ("🚀rocket.txt", "🚀rocket.txt"),
        ("报告（终稿）.txt", "报告（终稿）.txt"),
    ],
)
def test_sanitize_filename_accepts_and_strips_path(raw_name, expected):
    assert fs.sanitize_filename(raw_name) == expected


# ===== 图片 magic bytes 校验：伪装扩展名拒绝，真实图片通过 =====


@pytest.mark.parametrize(
    ("ext", "magic"),
    [
        ("jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16),
        ("png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 10),
        ("gif", b"GIF89a" + b"\x00" * 13),
        ("bmp", b"BM" + b"\x00" * 17),
        ("webp", b"RIFF" + b"\x00" * 4 + b"WEBP"),
    ],
)
def test_image_magic_bytes_accepts_real_header(ext, magic):
    fs.validate_image_magic_bytes(ext, magic)


@pytest.mark.parametrize(
    ("ext", "content"),
    [
        ("jpg", b"plain text disguised as jpg"),  # 文本伪装 .jpg
        ("png", b"\xff\xd8\xff corrupted"),  # jpg 头伪装 .png
        ("gif", b""),
        ("bmp", b"MZ\x90\x00"),  # PE 可执行文件头伪装 .bmp
        ("webp", b"RIFF" + b"\x00" * 4 + b"WEBQ"),
    ],
)
def test_image_magic_bytes_rejects_forged_content(ext, content):
    with pytest.raises(BusinessException) as ei:
        fs.validate_image_magic_bytes(ext, content)
    assert ei.value.code == ResultCode.FILE_TYPE_NOT_SUPPORTED


def test_image_magic_bytes_skips_non_image():
    # 非图片类型不做内容校验
    fs.validate_image_magic_bytes("txt", b"anything")
    fs.validate_image_magic_bytes("bin", b"")


async def test_upload_file_rejects_forged_image(db, local_storage):
    with pytest.raises(BusinessException) as ei:
        await fs.file_service.upload_file(
            db, filename="fake.jpg", content=b"plain text not a jpeg", content_type="image/jpeg"
        )
    assert ei.value.code == ResultCode.FILE_TYPE_NOT_SUPPORTED


async def test_upload_file_accepts_real_png(db, local_storage, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    created = await fs.file_service.upload_file(
        db, filename="real.png", content=png, content_type="image/png"
    )
    assert created.id > 0
    assert created.type == "png"


# ===== upload_file：本地存储真实读写 + MD5 去重（秒传） + 软删复活 =====


async def test_upload_file_stores_content_and_metadata(db, local_storage, tmp_path):
    content = b"hello dehaze file"
    created = await fs.file_service.upload_file(
        db, filename="hello.txt", content=content, content_type="text/plain"
    )

    assert created.id > 0
    assert created.name == "hello.txt"
    assert created.type == "txt"
    assert created.md5
    assert created.size_bytes == len(content)
    assert created.deleted == 0
    # object_name 符合 upload/{yyyyMMdd}/{md5}.{ext} 规则
    assert created.object_name.startswith("upload/")
    assert created.object_name.endswith(f"/{created.md5}.txt")
    # 物理文件真实写入存储
    assert local_storage.exists("dehaze", created.object_name) is True


async def test_upload_file_md5_dedup_reuses_record(db, local_storage):
    content = b"dedup-content-1"
    first = await fs.file_service.upload_file(
        db, filename="a.txt", content=content, content_type="text/plain"
    )
    second = await fs.file_service.upload_file(
        db, filename="renamed.txt", content=content, content_type="text/plain"
    )
    # MD5 唯一索引：同内容命中秒传，复用同一记录
    assert second.id == first.id


async def test_upload_file_revives_soft_deleted_record(db, local_storage):
    content = b"revive-content-1"
    created = await fs.file_service.upload_file(
        db, filename="revive.txt", content=content, content_type="text/plain"
    )
    await fs.file_service.delete_file_with_storage(db, created.id)

    # 软删后重新上传同内容：唯一键含 deleted，软删行不占键位，插入新活跃记录
    await fs.file_service.upload_file(
        db, filename="revive.txt", content=content, content_type="text/plain"
    )
    result = await db.execute(
        select(SysFile.id, SysFile.deleted).where(SysFile.md5 == created.md5, SysFile.deleted == 0)
    )
    row = result.one()
    assert row.id != created.id
    assert row.deleted == 0


async def test_upload_file_zero_byte(db, local_storage):
    created = await fs.file_service.upload_file(
        db, filename="empty.bin", content=b"", content_type="application/octet-stream"
    )
    assert created.id > 0
    assert created.size_bytes == 0


# ===== delete_file_with_storage：物理删除 + 软删 =====


async def test_delete_file_removes_physical_and_soft_deletes(db, local_storage):
    content = b"delete-me-1"
    created = await fs.file_service.upload_file(
        db, filename="del.txt", content=content, content_type="text/plain"
    )
    await fs.file_service.delete_file_with_storage(db, created.id)

    assert local_storage.exists("dehaze", created.object_name) is False
    # （单会话事务模式下实体属性可能过期，用列查询读库验证；查软删行需绕过全局过滤）
    row = await db.execute(
        select(SysFile.deleted)
        .where(SysFile.id == created.id)
        .execution_options(include_deleted=True)
    )
    assert (row.scalar_one()) != 0


async def test_delete_file_not_found_raises_b0401(db):
    with pytest.raises(BusinessException) as ei:
        await fs.file_service.delete_file_with_storage(db, 999999999)
    assert ei.value.code == ResultCode.FILE_NOT_FOUND


# ===== get_page：关键字匹配文件名与类型（对齐 Java/Go 双列匹配） =====


async def _insert_file(db, name: str, ftype: str) -> SysFile:
    f = SysFile(
        type=ftype,
        name=name,
        object_name=f"upload/test/{uuid.uuid4().hex}.{ftype}",
        storage="minio",
        size="1B",
        size_bytes=1,
        md5=uuid.uuid4().hex,
    )
    db.add(f)
    await db.flush()
    return f


async def test_get_page_keywords_match_name_and_type(db):
    by_name = await _insert_file(db, "unique_name_marker_a.txt", "png")
    by_type = await _insert_file(db, "plain_file_b.txt", "markerpng")
    await _insert_file(db, "unrelated.txt", "pdf")

    items, _ = await fs.file_service.get_file_page(db, 1, 50, "marker")
    ids = {f.id for f in items}
    assert by_name.id in ids
    assert by_type.id in ids
    assert all("marker" not in (f.name or "") or f.id in ids for f in items)
    assert by_name in items
    assert by_type in items


async def test_get_page_excludes_deleted(db):
    created = await _insert_file(db, "tobedeleted_page.txt", "txt")
    await fs.file_service.delete_file_with_storage(db, created.id)

    items, _ = await fs.file_service.get_file_page(db, 1, 50, "tobedeleted_page")
    assert created.id not in {f.id for f in items}


# ===== 孤儿文件清理：仅删除超过保留阈值的无引用文件 =====


async def test_cleanup_orphan_files_respects_retention_threshold(db, tmp_path, monkeypatch):
    from app.infrastructure.job import handlers
    from app.service.storage import factory

    storage = LocalStorageService(base_dir=str(tmp_path))
    monkeypatch.setattr(factory, "get_storage_service", lambda: storage)

    stale = "upload/20260101/stale_orphan.bin"
    fresh = "upload/20260101/fresh_orphan.bin"
    storage.upload("dehaze", stale, b"stale", "application/octet-stream")
    storage.upload("dehaze", fresh, b"fresh", "application/octet-stream")

    # 陈旧孤儿改为 72 小时前（超过 48h 阈值）；新孤儿保持当前时间
    stale_path = tmp_path / "dehaze" / stale
    old_ts = time.time() - 72 * 3600
    os.utime(stale_path, (old_ts, old_ts))

    msg = await handlers.cleanup_orphan_files()

    assert storage.exists("dehaze", stale) is False
    assert storage.exists("dehaze", fresh) is True
    assert "已删除=1" in msg
