import logging
import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.schema.file import FilePageVO, FileUploadResultVO, FileVO
from app.service.file_service import file_service
from app.service.storage.factory import get_storage_by_name

logger = logging.getLogger(__name__)

# MD5 格式：32 位十六进制（T-FM-034/035：无效 MD5 返回 B0404"MD5格式无效"）
_MD5_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")

router = APIRouter(
    prefix="/api/v1/files", tags=["文件管理"], dependencies=[Depends(get_current_user)]
)


def _validate_file(file: UploadFile) -> None:
    """校验上传文件，不合法时抛出 BusinessException"""
    if not file.filename:
        raise BusinessException(ResultCode.PARAM_ERROR, "请选择文件")

    # 检查文件大小（通过 content-length 头，如果有的话）
    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE // 1024 // 1024
        raise BusinessException(ResultCode.FILE_TOO_LARGE, f"文件大小超过限制 ({max_mb}MB)")


def _build_file_url(file_info) -> str | None:
    """根据文件 storage 标识和 object_name 运行时拼接完整 URL"""
    if not file_info or not file_info.object_name:
        return None
    storage_service = get_storage_by_name(file_info.storage)
    return storage_service.get_url(file_info.object_name)


@router.post(
    "",
    summary="文件上传",
    description="上传文件到存储服务，支持文件去重（根据MD5）",
    response_model=Result[FileUploadResultVO],
)
async def upload_file(
    file: UploadFile = File(..., description="要上传的文件"),
    modelId: int | None = Form(default=None, description="模型ID"),
    db: AsyncSession = Depends(get_db),
) -> Result[FileUploadResultVO]:
    _validate_file(file)

    # 读取文件内容
    content = await file.read()

    # 校验实际文件大小
    if len(content) > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE // 1024 // 1024
        raise BusinessException(ResultCode.FILE_TOO_LARGE, f"文件大小超过限制 ({max_mb}MB)")

    # 上传文件
    file_info = await file_service.upload_file(
        db=db,
        filename=file.filename,
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )

    return success(
        data=FileUploadResultVO(
            id=file_info.id,
            name=file_info.name,
            type=file_info.type,
            size=file_info.size,
            sizeBytes=file_info.size_bytes,
            objectName=file_info.object_name,
            storage=file_info.storage,
            url=_build_file_url(file_info),
            md5=file_info.md5,
            createTime=file_info.create_time,
        ),
        msg="文件上传成功",
    )


@router.get(
    "/check",
    summary="文件校验",
    description="根据MD5值校验文件是否已存在，存在则返回文件信息",
    response_model=Result[FileVO],
)
async def check_file(
    md5: str = Query(..., description="文件MD5值"),
    db: AsyncSession = Depends(get_db),
) -> Result[FileVO]:
    # MD5 格式校验：32 位十六进制（T-FM-034/035：无效 MD5 返回 B0404"MD5格式无效"）
    if not md5 or not _MD5_PATTERN.fullmatch(md5):
        raise BusinessException(ResultCode.FILE_MD5_INVALID)
    file_info = await file_service.get_file_by_md5(db, md5)
    if not file_info:
        return success(data=None)
    return success(
        data=FileVO(
            id=file_info.id,
            name=file_info.name,
            type=file_info.type,
            size=file_info.size,
            sizeBytes=file_info.size_bytes,
            objectName=file_info.object_name,
            storage=file_info.storage,
            url=_build_file_url(file_info),
            md5=file_info.md5,
            createTime=file_info.create_time,
        )
    )


@router.get(
    "/page",
    summary="文件分页查询",
    description="分页查询文件列表，支持关键词搜索",
    response_model=Result[FilePageVO],
)
async def get_file_page(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    keywords: str | None = Query(default=None, description="搜索关键词"),
    db: AsyncSession = Depends(get_db),
) -> Result[FilePageVO]:
    items, total = await file_service.get_file_page(db, pageNum, pageSize, keywords)

    file_list = [
        FileVO(
            id=f.id,
            name=f.name,
            type=f.type,
            size=f.size,
            sizeBytes=f.size_bytes,
            objectName=f.object_name,
            storage=f.storage,
            url=_build_file_url(f),
            md5=f.md5,
            createTime=f.create_time,
        )
        for f in items
    ]

    return success(data=FilePageVO(list=file_list, total=total))


@router.get(
    "/download/{object_name:path}",
    summary="文件下载",
    description="根据对象名称从对应存储后端下载文件",
)
async def download_file(
    object_name: str,
    db: AsyncSession = Depends(get_db),
):
    # 防止路径遍历攻击
    if ".." in object_name or object_name.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="无效的文件路径")

    # 获取文件信息
    file_info = await file_service.get_file_by_object_name(db, object_name)
    if not file_info:
        raise HTTPException(status_code=404, detail=ResultCode.FILE_NOT_FOUND.msg)

    # 按 storage 选后端流式读取（统一无分支，不再前缀判断 / 302 跳转）
    # 构造 Content-Disposition（RFC 5987 编码中文文件名）
    filename = file_info.name
    ascii_filename = filename.encode("ascii", "ignore").decode("ascii") or "download"
    encoded_filename = quote(filename)
    content_disposition = (
        f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"
    )

    headers = {"Content-Disposition": content_disposition}

    return StreamingResponse(
        file_service.download_file_stream(object_name, storage=file_info.storage),
        media_type="application/octet-stream",
        headers=headers,
    )


@router.delete(
    "",
    summary="文件删除",
    description="根据文件ID删除文件（包括物理文件和数据库记录）",
    response_model=Result[None],
)
async def delete_file(
    fileId: int = Query(..., description="文件ID"),
    db: AsyncSession = Depends(get_db),
) -> Result[None]:
    await file_service.delete_file_with_storage(db, fileId)
    return success(msg="文件删除成功")


@router.get(
    "/{file_id}",
    summary="获取文件信息",
    description="根据文件ID获取文件详细信息",
    response_model=Result[FileVO],
)
async def get_file_info(
    file_id: int,
    db: AsyncSession = Depends(get_db),
) -> Result[FileVO]:
    file_info = await file_service.get_file_by_id(db, file_id)

    if not file_info:
        # T-FM-044：文件不存在返回 B0401"文件不存在"（对齐文档与 Java 端行为）
        raise BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在")

    return success(
        data=FileVO(
            id=file_info.id,
            name=file_info.name,
            type=file_info.type,
            size=file_info.size,
            sizeBytes=file_info.size_bytes,
            objectName=file_info.object_name,
            storage=file_info.storage,
            url=_build_file_url(file_info),
            md5=file_info.md5,
            createTime=file_info.create_time,
            updateTime=file_info.update_time,
        )
    )
