"""
文件模块 Schema 模型

注意：url 字段为运行时动态生成（storage.baseUrl + object_name），不落库。
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class FileUploadResultVO(BaseModel):
    """文件上传结果VO"""

    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    type: str | None = Field(default=None, description="文件类型")
    size: str = Field(description="文件大小(格式化)")
    sizeBytes: int | None = Field(default=None, description="文件大小(原始字节数)")
    objectName: str = Field(description="对象存储名称")
    storage: str = Field(description="存储后端标识")
    url: str | None = Field(default=None, description="文件URL（运行时动态生成）")
    md5: str = Field(description="文件MD5值")
    createTime: datetime | None = Field(default=None, description="创建时间")


class FileVO(BaseModel):
    """文件信息VO"""

    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    type: str | None = Field(default=None, description="文件类型")
    size: str | None = Field(default=None, description="文件大小(格式化)")
    sizeBytes: int | None = Field(default=None, description="文件大小(原始字节数)")
    objectName: str | None = Field(default=None, description="对象存储名称")
    storage: str | None = Field(default=None, description="存储后端标识")
    url: str | None = Field(default=None, description="文件URL（运行时动态生成）")
    md5: str | None = Field(default=None, description="文件MD5值")
    createTime: datetime | None = Field(default=None, description="创建时间")
    updateTime: datetime | None = Field(default=None, description="更新时间")


class FilePageVO(BaseModel):
    """文件分页结果VO"""

    list: List[FileVO] = Field(default_factory=list, description="文件列表")
    total: int = Field(description="总记录数")
