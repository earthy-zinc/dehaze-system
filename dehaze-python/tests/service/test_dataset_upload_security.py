"""数据项图片上传安全校验测试。

覆盖三端契约要求（需求规格 §3.3：系统校验文件格式、大小）：
- 非图片扩展名（html/svg 等）拒绝入库，防止经文件 URL 渲染的存储型 XSS
- 扩展名伪装但内容非真实图片（魔数/解析不符）拒绝
- 大小超限拒绝（原实现整包读入内存且无上限）
- type 枚举与 SDK ItemFileUploadForm 对齐
"""

import io
from typing import Any

import PIL.Image
import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.dataset import DatasetAddForm, DatasetItemCreateForm
from app.service.dataset import item_file_service as item_file_module
from app.service.dataset._shared import validate_image_content
from app.service.dataset.dataset_item_service import dataset_item_service
from app.service.dataset.item_file_service import item_file_service


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PIL.Image.new("RGB", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


class _UploadFile:
    def __init__(self, filename: str, content: bytes, content_type="image/png"):
        self.filename = filename
        self.content = content
        self.content_type = content_type

    async def read(self):
        return self.content


class _StubDb(AsyncSession):
    """AsyncSession 假件：仅实现 add/flush/refresh 与 id 发号（上传链路只用到这些）。

    继承真实 AsyncSession 以满足端点 db 契约；仓储调用全部由打桩的 service 承接，
    故不会触达真实 SQL。__init__ 不绑定引擎。
    """

    def __init__(self):
        super().__init__()
        self.added: Any = None

    def add(self, instance, _warn: bool = True) -> None:
        self.added = instance

    async def flush(self, objects=None) -> None:
        assert self.added is not None
        self.added.id = 123

    async def refresh(self, instance, attribute_names=None, with_for_update=None) -> None:
        pass


def _item():
    from types import SimpleNamespace

    return SimpleNamespace(id=7, dataset_id=1, name="item")


async def _no_evict(redis):
    pass


# ===== validate_image_content =====


def test_reject_html_extension():
    with pytest.raises(BusinessException) as exc:
        validate_image_content("payload.html", b"<h1>x</h1>")
    assert exc.value.code == ResultCode.PARAM_ERROR


def test_reject_svg_extension():
    with pytest.raises(BusinessException) as exc:
        validate_image_content("payload.svg", b"<svg onload='alert(1)'/>")
    assert exc.value.code == ResultCode.PARAM_ERROR


def test_reject_missing_extension():
    with pytest.raises(BusinessException) as exc:
        validate_image_content("noext", _png_bytes())
    assert exc.value.code == ResultCode.PARAM_ERROR


def test_reject_fake_png_magic():
    """扩展名 png 但内容非图片（文本伪装）"""
    with pytest.raises(BusinessException) as exc:
        validate_image_content("evil.png", b"#!/bin/sh\nrm -rf /")
    assert exc.value.code == ResultCode.PARAM_ERROR


def test_accept_real_png():
    validate_image_content("ok.png", _png_bytes())


# ===== item_file_service.upload_item_file =====


async def test_upload_item_file_rejects_unknown_type(monkeypatch, mock_redis):
    async def _get_item(db, item_id):
        return _item()

    monkeypatch.setattr(item_file_module.dataset_repository, "get_item_by_id", _get_item)
    with pytest.raises(BusinessException) as exc:
        await item_file_service.upload_item_file(
            db=_StubDb(),
            redis=mock_redis,
            item_id=7,
            image_type="banner",
            scene_type="",
            haze_level="",
            description="",
            file=_UploadFile("a.png", _png_bytes()),
        )
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_upload_item_file_rejects_oversize(monkeypatch, mock_redis):
    async def _get_item(db, item_id):
        return _item()

    monkeypatch.setattr(item_file_module.dataset_repository, "get_item_by_id", _get_item)
    monkeypatch.setattr(item_file_module.dataset_service, "_evict_all_cache", _no_evict)
    original = settings.MAX_UPLOAD_SIZE
    settings.MAX_UPLOAD_SIZE = 16
    try:
        with pytest.raises(BusinessException) as exc:
            await item_file_service.upload_item_file(
                db=_StubDb(),
                redis=mock_redis,
                item_id=7,
                image_type="hazy",
                scene_type="",
                haze_level="",
                description="",
                file=_UploadFile("big.png", _png_bytes()),
            )
        assert exc.value.code == ResultCode.FILE_TOO_LARGE
    finally:
        settings.MAX_UPLOAD_SIZE = original


async def test_upload_item_file_rejects_non_image(monkeypatch, mock_redis):
    async def _get_item(db, item_id):
        return _item()

    monkeypatch.setattr(item_file_module.dataset_repository, "get_item_by_id", _get_item)
    monkeypatch.setattr(item_file_module.dataset_service, "_evict_all_cache", _no_evict)
    with pytest.raises(BusinessException) as exc:
        await item_file_service.upload_item_file(
            db=_StubDb(),
            redis=mock_redis,
            item_id=7,
            image_type="hazy",
            scene_type="",
            haze_level="",
            description="",
            file=_UploadFile("evil.html", b"<script>alert(1)</script>"),
        )
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_upload_item_file_accepts_real_image(monkeypatch, mock_redis):
    async def _get_item(db, item_id):
        return _item()

    monkeypatch.setattr(item_file_module.dataset_repository, "get_item_by_id", _get_item)
    monkeypatch.setattr(item_file_module.dataset_service, "_evict_all_cache", _no_evict)
    called = {}

    async def _fake_upload_file(db, filename, content, content_type):
        called["filename"] = filename
        from types import SimpleNamespace

        return SimpleNamespace(id=99)

    monkeypatch.setattr(item_file_module.file_service, "upload_file", _fake_upload_file)

    def _fake_vo(item_file, file_obj):
        return {"id": 123, "itemId": item_file.item_id, "type": item_file.type}

    monkeypatch.setattr(item_file_module, "_build_file_vo", _fake_vo)
    vo = await item_file_service.upload_item_file(
        db=_StubDb(),
        redis=mock_redis,
        item_id=7,
        image_type="hazy",
        scene_type="street",
        haze_level="light",
        description="",
        file=_UploadFile("real.png", _png_bytes()),
    )
    assert called["filename"] == "real.png"
    assert vo["type"] == "hazy"


# ===== dataset_item_service.upload_dataset_item_with_images（仅雾图路径此前无校验）=====


async def test_paired_upload_rejects_non_image_hazy_without_clear(mock_redis):
    """仅有雾图（无清晰图）上传时，有雾图也必须校验为真实图片"""
    with pytest.raises(BusinessException) as exc:
        await dataset_item_service.upload_dataset_item_with_images(
            db=_StubDb(),
            redis=mock_redis,
            dataset_id=1,
            clear_file_content=None,
            hazy_files_data=[
                {"filename": "01_hazy.html", "content": b"<h1>x</h1>", "contentType": "text/html"}
            ],
        )
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_paired_upload_rejects_fake_image_hazy_without_clear(mock_redis):
    with pytest.raises(BusinessException) as exc:
        await dataset_item_service.upload_dataset_item_with_images(
            db=_StubDb(),
            redis=mock_redis,
            dataset_id=1,
            clear_file_content=None,
            hazy_files_data=[
                {"filename": "01_hazy.png", "content": b"not an image", "contentType": "image/png"}
            ],
        )
    assert exc.value.code == ResultCode.PARAM_ERROR


async def test_paired_upload_rejects_oversize_hazy(monkeypatch, mock_redis):
    monkeypatch.setattr(item_file_module.dataset_service, "_evict_all_cache", _no_evict)
    original = settings.MAX_UPLOAD_SIZE
    settings.MAX_UPLOAD_SIZE = 16
    try:
        with pytest.raises(BusinessException) as exc:
            await dataset_item_service.upload_dataset_item_with_images(
                db=_StubDb(),
                redis=mock_redis,
                dataset_id=1,
                clear_file_content=None,
                hazy_files_data=[
                    {
                        "filename": "01_hazy.png",
                        "content": _png_bytes(),
                        "contentType": "image/png",
                    }
                ],
            )
        assert exc.value.code == ResultCode.FILE_TOO_LARGE
    finally:
        settings.MAX_UPLOAD_SIZE = original


# ===== 表单长度与 DB 列宽一致（varchar(64)，超长应 A0400 而非 DB 500）=====


def test_dataset_name_over_64_rejected():
    with pytest.raises(ValidationError):
        DatasetAddForm(parentId=0, name="x" * 65)


def test_dataset_name_64_accepted():
    form = DatasetAddForm(parentId=0, name="x" * 64)
    assert form.name == "x" * 64


def test_item_name_over_64_rejected():
    with pytest.raises(ValidationError):
        DatasetItemCreateForm(datasetId=1, name="y" * 65)
