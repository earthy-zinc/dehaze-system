"""
文件模块 Schema 模型
"""
from typing import Optional
from pydantic import BaseModel, Field


# ==================== 查询参数模型 ====================

class FileIdQuery(BaseModel):
    """文件ID查询参数"""
    fileId: int = Field(..., description="文件ID")


class FileCheckQuery(BaseModel):
    """文件校验查询参数"""
    md5: str = Field(..., min_length=1, description="文件MD5值")


# ==================== 表单参数模型 ====================

class FileUploadForm(BaseModel):
    """文件上传表单参数"""
    modelId: Optional[int] = Field(default=None, description="模型ID")


# ==================== 响应模型 ====================

class FileUploadResultVO(BaseModel):
    """文件上传结果VO"""
    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    url: str = Field(description="文件URL")
    size: int = Field(description="文件大小(字节)")
    md5: str = Field(description="文件MD5值")


class FileVO(BaseModel):
    """文件信息VO"""
    id: int = Field(description="文件ID")
    name: str = Field(description="文件名称")
    url: Optional[str] = Field(default=None, description="文件URL")
    size: Optional[int] = Field(default=None, description="文件大小(字节)")
    md5: Optional[str] = Field(default=None, description="文件MD5值")
    objectName: Optional[str] = Field(default=None, description="对象存储名称")
