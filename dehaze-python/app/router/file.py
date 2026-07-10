import logging
from typing import Optional
from urllib.parse import quote

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, error, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.schema.file import FilePageVO, FileUploadResultVO, FileVO
from app.service.file_service import FileService, validate_md5_format
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files",
                   tags=["文件管理"], dependencies=[Depends(get_current_user)])


def _validate_file(file: UploadFile) -> tuple[bool, ResultCode | None, str]:
    """
    校验上传文件

    Returns:
        (是否有效, 错误码, 错误消息)
    """
    if not file.filename:
        return False, ResultCode.PARAM_ERROR, "请选择文件"

    # 检查文件扩展名
    ext = file.filename.rsplit(
        ".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False, ResultCode.FILE_TYPE_NOT_SUPPORTED, f"不支持的文件类型: .{ext}"

    # 检查文件大小（通过 content-length 头，如果有的话）
    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE // 1024 // 1024
        return False, ResultCode.FILE_TOO_LARGE, f"文件大小超过限制 ({max_mb}MB)"

    return True, None, ""


@router.post(
    "",
    summary="文件上传",
    description="上传文件到存储服务，支持文件去重（根据MD5）",
    response_model=Result[FileUploadResultVO],
)
async def upload_file(
    file: UploadFile = File(..., description="要上传的文件"),
    db: AsyncSession = Depends(get_db),
) -> Result[FileUploadResultVO]:
    # 校验文件
    is_valid, error_code, error_msg = _validate_file(file)
    if not is_valid:
        assert error_code is not None
        return error(error_msg, error_code.code)

    # 读取文件内容
    content = await file.read()

    # 校验实际文件大小
    if len(content) > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE // 1024 // 1024
        return error(f"文件大小超过限制 ({max_mb}MB)", ResultCode.FILE_TOO_LARGE.code)

    try:
        # 上传文件
        if file.filename is None:
            raise ValueError("文件名不能为空")
        file_info = await FileService.upload_file(
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
                url=file_info.url,
                path=file_info.path,
                objectName=file_info.object_name,
                md5=file_info.md5,
                createTime=file_info.create_time,
            ),
            msg="文件上传成功",
        )
    except BusinessException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}", exc_info=True)
        return error("文件上传失败", ResultCode.FILE_STORAGE_ERROR.code)


@router.get(
    "/check",
    summary="文件校验",
    description="根据MD5值校验文件是否已存在",
    response_model=Result[bool],
)
async def check_file(
    md5: str = Query(..., min_length=32, max_length=32, description="文件MD5值"),
    db: AsyncSession = Depends(get_db),
) -> Result[bool]:
    # MD5 格式校验
    if not validate_md5_format(md5):
        return error("MD5格式无效，必须为32位十六进制字符串", ResultCode.FILE_MD5_INVALID.code)

    exists = await FileService.check_file_exists(db, md5)
    return success(data=exists)


@router.get(
    "/page",
    summary="文件分页查询",
    description="分页查询文件列表，支持关键词搜索",
    response_model=Result[FilePageVO],
)
async def get_file_page(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    keywords: Optional[str] = Query(default=None, description="搜索关键词"),
    db: AsyncSession = Depends(get_db),
) -> Result[FilePageVO]:
    items, total = await FileService.get_file_page(db, pageNum, pageSize, keywords)

    file_list = [
        FileVO(
            id=f.id,
            name=f.name,
            type=f.type,
            size=f.size,
            url=f.url,
            path=f.path,
            objectName=f.object_name,
            md5=f.md5,
            createTime=f.create_time,
        )
        for f in items
    ]

    return success(data=FilePageVO(list=file_list, total=total))


@router.get(
    "/download/{object_name:path}",
    summary="文件下载",
    description="根据对象名称从存储服务下载文件",
)
async def download_file(
    object_name: str,
    db: AsyncSession = Depends(get_db),
):
    # 防止路径遍历攻击
    if ".." in object_name or object_name.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="无效的文件路径")

    # 获取文件信息
    file_info = await FileService.get_file_by_object_name(db, object_name)
    if not file_info:
        raise HTTPException(
            status_code=404, detail=ResultCode.FILE_NOT_FOUND.msg)

    # 构造 Content-Disposition（RFC 5987 编码中文文件名）
    filename = file_info.name
    ascii_filename = filename.encode(
        "ascii", "ignore").decode("ascii") or "download"
    encoded_filename = quote(filename)
    content_disposition = (
        f"attachment; filename=\"{ascii_filename}\"; "
        f"filename*=UTF-8''{encoded_filename}"
    )

    # 构造响应头
    headers = {"Content-Disposition": content_disposition}

    # 返回流式响应
    return StreamingResponse(
        FileService.download_file_stream(object_name),
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
    try:
        await FileService.delete_file_with_storage(db, fileId)
        return success(msg="文件删除成功")
    except BusinessException:
        raise
    except Exception as e:
        logger.error(f"文件删除失败: {e}", exc_info=True)
        return error("文件删除失败", ResultCode.FILE_STORAGE_ERROR.code)


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
    file_info = await FileService.get_file_by_id(db, file_id)

    if not file_info:
        return error(ResultCode.FILE_NOT_FOUND.msg, ResultCode.FILE_NOT_FOUND.code)

    return success(
        data=FileVO(
            id=file_info.id,
            name=file_info.name,
            type=file_info.type,
            size=file_info.size,
            url=file_info.url,
            path=file_info.path,
            objectName=file_info.object_name,
            md5=file_info.md5,
            createTime=file_info.create_time,
            updateTime=file_info.update_time,
        )
    )
