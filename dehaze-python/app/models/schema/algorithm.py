"""
算法模块 Schema 模型
"""

import re

from pydantic import BaseModel, Field, field_validator


class AlgorithmForm(BaseModel):
    """算法表单 (对齐 Java AlgorithmForm)"""

    id: int | None = Field(default=None, description="算法ID")
    parentId: int = Field(default=0, ge=0, alias="parentId", description="父级ID")
    type: str = Field(..., min_length=1, description="算法类型")
    name: str = Field(..., min_length=1, max_length=100, description="算法名称")
    path: str | None = Field(default="", description="模型路径")
    importPath: str | None = Field(default="", alias="importPath", description="导入路径")
    description: str | None = Field(default="", description="算法描述")
    status: int | None = Field(default=None, description="算法状态")
    version: str | None = Field(default=None, description="算法版本号")

    model_config = {"populate_by_name": True}


class AlgorithmAuditForm(BaseModel):
    """算法审核表单 (对齐 Java AlgorithmAuditForm)"""

    approved: bool = Field(..., description="是否通过")
    remark: str | None = Field(default=None, description="审核备注（驳回时必填）")


class AlgorithmVersionForm(BaseModel):
    """新增版本表单 (对齐 Java AlgorithmVersionForm)"""

    version: str = Field(..., description="版本号 vX.Y.Z")
    changeLog: str | None = Field(default=None, alias="changeLog", description="变更日志")
    status: int | None = Field(default=None, description="该版本时的状态")
    configJson: str | None = Field(
        default=None, alias="configJson", description="该版本时的配置JSON"
    )
    modelFileId: int | None = Field(default=None, alias="modelFileId", description="模型文件ID")
    isActive: int | None = Field(
        default=0, ge=0, le=1, alias="isActive", description="是否当前活跃版本"
    )

    model_config = {"populate_by_name": True}

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not re.match(r"^v\d+\.\d+\.\d+$", v):
            raise ValueError("版本号格式必须为 vX.Y.Z")
        return v


class AlgorithmVO(BaseModel):
    """算法VO（对齐 Java AlgorithmVO 字段名 camelCase）"""

    id: int = Field(description="算法ID")
    parentId: int = Field(
        validation_alias="parent_id", serialization_alias="parentId", description="父级ID"
    )
    type: str | None = Field(default=None, description="算法类型")
    name: str = Field(description="算法名称")
    path: str | None = Field(default=None, description="模型路径")
    size: str | None = Field(default=None, description="模型大小")
    img: str | None = Field(default=None, description="示例图片")
    params: str | None = Field(default=None, description="参数量")
    flops: str | None = Field(default=None, description="计算量")
    importPath: str | None = Field(
        default=None,
        validation_alias="import_path",
        serialization_alias="importPath",
        description="导入路径",
    )
    description: str | None = Field(default=None, description="算法描述")
    status: int = Field(description="状态(1:草稿;2:测试中;3:待审核;4:已发布;5:已停用;6:已归档)")
    version: str | None = Field(default=None, description="版本号")
    auditBy: int | None = Field(
        default=None,
        validation_alias="audit_by",
        serialization_alias="auditBy",
        description="审核人ID",
    )
    auditTime: str | None = Field(
        default=None,
        validation_alias="audit_time",
        serialization_alias="auditTime",
        description="审核时间",
    )
    auditRemark: str | None = Field(
        default=None,
        validation_alias="audit_remark",
        serialization_alias="auditRemark",
        description="审核备注",
    )
    createTime: str | None = Field(
        default=None,
        validation_alias="create_time",
        serialization_alias="createTime",
        description="创建时间",
    )
    updateTime: str | None = Field(
        default=None,
        validation_alias="update_time",
        serialization_alias="updateTime",
        description="更新时间",
    )
    children: list["AlgorithmVO"] | None = Field(default=None, description="子算法列表")

    model_config = {"populate_by_name": True}


class AlgorithmOptionVO(BaseModel):
    """算法下拉选项VO"""

    value: int = Field(description="选项值(算法ID)")
    label: str = Field(description="选项标签(算法名称)")
    children: list["AlgorithmOptionVO"] | None = Field(default=None, description="子选项列表")


class AlgorithmVersionVO(BaseModel):
    """算法版本历史VO (对齐 Java AlgorithmVersionVO)"""

    id: int = Field(description="版本ID")
    algorithmId: int = Field(
        validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID"
    )
    version: str = Field(description="版本号")
    changeLog: str | None = Field(
        default=None,
        validation_alias="change_log",
        serialization_alias="changeLog",
        description="变更日志",
    )
    status: int | None = Field(default=None, description="该版本时的状态")
    configJson: str | None = Field(
        default=None,
        validation_alias="config_json",
        serialization_alias="configJson",
        description="该版本时的配置JSON",
    )
    modelFileId: int | None = Field(
        default=None,
        validation_alias="model_file_id",
        serialization_alias="modelFileId",
        description="模型文件ID",
    )
    isActive: int | None = Field(
        default=0,
        validation_alias="is_active",
        serialization_alias="isActive",
        description="是否当前活跃版本",
    )
    createTime: str | None = Field(
        default=None,
        validation_alias="create_time",
        serialization_alias="createTime",
        description="创建时间",
    )
    updateTime: str | None = Field(
        default=None,
        validation_alias="update_time",
        serialization_alias="updateTime",
        description="更新时间",
    )

    model_config = {"populate_by_name": True}


class AlgorithmMonitorVO(BaseModel):
    """算法监控数据VO (对齐 Java AlgorithmMonitorVO)"""

    callCount: int = Field(default=0, description="调用次数")
    avgTime: float = Field(default=0, description="平均处理时间(毫秒)")
    successRate: float = Field(default=0, description="成功率(0-100)")
    todayCallCount: int = Field(default=0, description="今日调用次数")


AlgorithmVO.model_rebuild()
AlgorithmOptionVO.model_rebuild()
