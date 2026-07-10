"""
文件模块 Schema 模型
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# ==================== 查询参数模型 ====================


class FileIdQuery(BaseModel):
    """文件ID查询参数"""
    fileId: int = Field(..., description="文件ID")


class FileCheckQuery(BaseModel):
    """文件校验查询参数"""
    md5: str = Field(..., min_length=32, max_length=32,
                     pattern=r'^[a-fA-F0-9]{32}$', description="文件MD5值")


class FilePageQuery(BaseModel):
    """文件分页查询参数"""
    pageNum: int = Field(default=1, ge=1, description="页码")
    pageSize: int = Field(default=10, ge=1, le=100, description="每页数量")
    keywords: Optional[str] = Field(default=None, description="搜索关键词")


# ==================== 表单参数模型 ====================

class FileUploadForm(BaseModel):
    """文件上传表单参数"""
    modelId: Optional[int] = Field(default=None, description="模型ID")


# ==================== 响应模型 ====================

class FileUploadResultVO(BaseModel):
    """文件上传结果VO"""
    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    type: Optional[str] = Field(default=None, description="文件类型")
    size: str = Field(description="文件大小(格式化)")
    sizeBytes: int = Field(description="文件大小(字节)")
    url: Optional[str] = Field(default=None, description="文件URL")
    path: Optional[str] = Field(default=None, description="文件路径")
    objectName: str = Field(description="对象存储名称")
    md5: str = Field(description="文件MD5值")
    createTime: Optional[datetime] = Field(default=None, description="创建时间")


class FileCheckResultVO(BaseModel):
    """文件校验结果VO"""
    exists: bool = Field(description="文件是否存在")
    fileId: Optional[int] = Field(default=None, description="已存在文件的ID")
    url: Optional[str] = Field(default=None, description="已存在文件的URL")


class FileVO(BaseModel):
    """文件信息VO"""
    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    type: Optional[str] = Field(default=None, description="文件类型")
    size: Optional[str] = Field(default=None, description="文件大小(格式化)")
    sizeBytes: Optional[int] = Field(default=None, description="文件大小(字节)")
    url: Optional[str] = Field(default=None, description="文件URL")
    path: Optional[str] = Field(default=None, description="文件路径")
    objectName: Optional[str] = Field(default=None, description="对象存储名称")
    md5: Optional[str] = Field(default=None, description="文件MD5值")
    createTime: Optional[datetime] = Field(default=None, description="创建时间")


class FilePageVO(BaseModel):
    """文件分页结果VO"""
    list: List[FileVO] = Field(default_factory=list, description="文件列表")
    total: int = Field(description="总记录数")
