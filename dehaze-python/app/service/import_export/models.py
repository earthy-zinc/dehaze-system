"""
导入导出通用框架数据模型
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ExportFieldConfig(BaseModel):
    field: str = Field(description="字段名（实体属性名）")
    label: str = Field(description="表头显示名称")
    order: int = Field(default=0, description="排序")
    date_format: Optional[str] = Field(default=None, description="日期格式")
    dict_type: Optional[str] = Field(default=None, description="字典类型编码，用于值-标签转换")
    hidden: bool = Field(default=False, description="是否隐藏（不导出）")


class ImportFieldConfig(BaseModel):
    field: str = Field(description="字段名（实体属性名）")
    label: str = Field(description="表头显示名称")
    required: bool = Field(default=False, description="是否必填")
    date_format: Optional[str] = Field(default=None, description="日期格式")
    dict_type: Optional[str] = Field(default=None, description="字典类型编码")
    regex: Optional[str] = Field(default=None, description="正则校验")
    max_length: Optional[int] = Field(default=None, description="最大长度")
    default_value: Optional[str] = Field(default=None, description="默认值")


class ExportContext(BaseModel):
    task_id: str = Field(description="任务ID")
    module: str = Field(description="模块名")
    format: str = Field(default="excel", description="导出格式: excel/csv")
    selected_fields: Optional[list[str]] = Field(default=None, description="选定导出字段")
    query_params: dict[str, Any] = Field(default_factory=dict, description="查询参数")
    total_count: int = Field(default=0, description="预估总行数")


class ImportOptions(BaseModel):
    mode: str = Field(default="all", description="导入模式: all(全量) / partial(部分)")
    extra: dict[str, Any] = Field(default_factory=dict, description="模块特定额外参数")


class ImportError(BaseModel):
    row: int = Field(description="行号")
    field: Optional[str] = Field(default=None, description="字段名")
    message: str = Field(description="错误信息")


class ImportResult(BaseModel):
    total_rows: int = Field(default=0, description="总行数")
    success_count: int = Field(default=0, description="成功数")
    failure_count: int = Field(default=0, description="失败数")
    skipped_count: int = Field(default=0, description="跳过数")
    errors: list[ImportError] = Field(default_factory=list, description="错误明细")
    error_report_object_name: Optional[str] = Field(default=None, description="错误报告对象名")


class ProgressCallback:
    """进度回调协议（结构化类型，便于实现）"""

    async def on_progress(self, processed: int, total: int) -> None: ...


class CancelChecker:
    """取消检测协议"""

    async def is_cancelled(self) -> bool: ...
