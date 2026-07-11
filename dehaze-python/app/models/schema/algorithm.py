"""
算法模块 Schema 模型
"""
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# ==================== 查询参数模型 ====================


class AlgorithmQuery(BaseModel):
    """算法查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键词(算法名称)")


class AlgorithmIdsQuery(BaseModel):
    """批量删除查询参数"""
    ids: str = Field(..., description="算法ID，多个以英文逗号(,)分隔")


# ==================== 路径参数模型 ====================

class AlgorithmIdPath(BaseModel):
    """算法ID路径参数"""
    algorithm_id: int = Field(..., description="算法ID")


# ==================== 请求体模型 ====================

class AlgorithmForm(BaseModel):
    """算法表单 (对齐 Java AlgorithmForm)"""
    id: Optional[int] = Field(default=None, description="算法ID")
    parentId: int = Field(
        default=0, ge=0, alias="parentId", description="父级ID")
    type: Optional[str] = Field(default="", description="算法类型")
    name: str = Field(..., min_length=1, max_length=100, description="算法名称")
    path: Optional[str] = Field(default="", description="模型路径")
    importPath: Optional[str] = Field(
        default="", alias="importPath", description="导入路径")
    description: Optional[str] = Field(default="", description="算法描述")
    status: int = Field(default=0, ge=0, le=5, description="状态(0:草稿;1:测试中;2:待审核;3:已发布;4:已停用;5:已归档)")
    version: Optional[str] = Field(default=None, description="算法版本号")

    model_config = {"populate_by_name": True}


class AlgorithmStatusForm(BaseModel):
    """算法状态修改表单"""
    status: int = Field(..., ge=0, le=5, description="目标状态(0:草稿;1:测试中;2:待审核;3:已发布;4:已停用;5:已归档)")


class AlgorithmAuditForm(BaseModel):
    """算法审核表单"""
    # passed=True 通过，False 驳回
    passed: bool = Field(..., description="是否通过")
    remark: Optional[str] = Field(default=None, description="审核备注（驳回时必填）")


class AlgorithmVersionForm(BaseModel):
    """新增版本表单 (对齐 Java AlgorithmVersionForm)"""
    version: str = Field(..., description="版本号 vX.Y.Z")
    changeLog: Optional[str] = Field(default=None, alias="changeLog", description="变更日志")
    status: Optional[int] = Field(default=None, description="该版本时的状态")
    configJson: Optional[str] = Field(default=None, alias="configJson", description="该版本时的配置JSON")
    modelFileId: Optional[int] = Field(default=None, alias="modelFileId", description="模型文件ID")
    isActive: Optional[int] = Field(default=0, ge=0, le=1, alias="isActive", description="是否当前活跃版本")

    model_config = {"populate_by_name": True}

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        if not re.match(r"^v\d+\.\d+\.\d+$", v):
            raise ValueError("版本号格式必须为 vX.Y.Z")
        return v


class AlgorithmRollbackForm(BaseModel):
    """版本回滚表单"""
    versionId: int = Field(..., alias="versionId", description="目标版本ID")

    model_config = {"populate_by_name": True}


# ==================== 响应模型 ====================

class AlgorithmVO(BaseModel):
    """算法VO（对齐 Java AlgorithmVO 字段名 camelCase）"""
    id: int = Field(description="算法ID")
    parentId: int = Field(validation_alias="parent_id", serialization_alias="parentId", description="父级ID")
    type: Optional[str] = Field(default=None, description="算法类型")
    name: str = Field(description="算法名称")
    path: Optional[str] = Field(default=None, description="模型路径")
    size: Optional[str] = Field(default=None, description="模型大小")
    img: Optional[str] = Field(default=None, description="示例图片")
    params: Optional[str] = Field(default=None, description="参数量")
    flops: Optional[str] = Field(default=None, description="计算量")
    importPath: Optional[str] = Field(default=None, validation_alias="import_path", serialization_alias="importPath", description="导入路径")
    description: Optional[str] = Field(default=None, description="算法描述")
    status: int = Field(description="状态(0:草稿;1:测试中;2:待审核;3:已发布;4:已停用;5:已归档)")
    version: Optional[str] = Field(default=None, description="版本号")
    auditBy: Optional[int] = Field(default=None, validation_alias="audit_by", serialization_alias="auditBy", description="审核人ID")
    auditTime: Optional[str] = Field(default=None, validation_alias="audit_time", serialization_alias="auditTime", description="审核时间")
    auditRemark: Optional[str] = Field(default=None, validation_alias="audit_remark", serialization_alias="auditRemark", description="审核备注")
    createTime: Optional[str] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="创建时间")
    updateTime: Optional[str] = Field(default=None, validation_alias="update_time", serialization_alias="updateTime", description="更新时间")
    children: Optional[List["AlgorithmVO"]] = Field(
        default=None, description="子算法列表")

    model_config = {"populate_by_name": True}


class AlgorithmOptionVO(BaseModel):
    """算法下拉选项VO"""
    value: int = Field(description="选项值(算法ID)")
    label: str = Field(description="选项标签(算法名称)")
    children: Optional[List["AlgorithmOptionVO"]] = Field(
        default=None, description="子选项列表")


class AlgorithmIdVO(BaseModel):
    """算法ID响应VO"""
    id: int = Field(description="算法ID")


class AlgorithmDeleteResultVO(BaseModel):
    """算法删除结果VO"""
    count: int = Field(description="删除数量")


class AlgorithmVersionVO(BaseModel):
    """算法版本历史VO (对齐 Java AlgorithmVersionVO)"""
    id: int = Field(description="版本ID")
    algorithmId: int = Field(validation_alias="algorithm_id", serialization_alias="algorithmId", description="算法ID")
    version: str = Field(description="版本号")
    changeLog: Optional[str] = Field(default=None, validation_alias="change_log", serialization_alias="changeLog", description="变更日志")
    status: Optional[int] = Field(default=None, description="该版本时的状态")
    configJson: Optional[str] = Field(default=None, validation_alias="config_json", serialization_alias="configJson", description="该版本时的配置JSON")
    modelFileId: Optional[int] = Field(default=None, validation_alias="model_file_id", serialization_alias="modelFileId", description="模型文件ID")
    isActive: Optional[int] = Field(default=0, validation_alias="is_active", serialization_alias="isActive", description="是否当前活跃版本")
    createTime: Optional[str] = Field(default=None, validation_alias="create_time", serialization_alias="createTime", description="创建时间")
    updateTime: Optional[str] = Field(default=None, validation_alias="update_time", serialization_alias="updateTime", description="更新时间")

    model_config = {"populate_by_name": True}


class AlgorithmMonitorVO(BaseModel):
    """算法监控数据VO"""
    algorithmId: int = Field(description="算法ID")
    algorithmName: Optional[str] = Field(default=None, description="算法名称")
    # 实时指标
    totalCalls: int = Field(default=0, description="总调用次数")
    avgTime: float = Field(default=0, description="平均耗时(毫秒)")
    successRate: float = Field(default=0, description="成功率(0-1)")
    # 最近调用
    lastCallTime: Optional[str] = Field(default=None, description="最近调用时间")


class AlgorithmMonitorStatsVO(BaseModel):
    """算法监控统计报表VO"""
    algorithmId: int = Field(description="算法ID")
    # 时间序列
    timeSeries: List[dict] = Field(default_factory=list, description="时间序列数据")
    # 汇总
    totalCalls: int = Field(default=0, description="总调用次数")
    avgTime: float = Field(default=0, description="平均耗时")
    maxTime: int = Field(default=0, description="最大耗时")
    minTime: int = Field(default=0, description="最小耗时")
    successRate: float = Field(default=0, description="成功率")


class AlgorithmImportResultVO(BaseModel):
    """算法导入结果VO"""
    success: bool = Field(description="是否成功")
    algorithmId: Optional[int] = Field(default=None, description="导入后的算法ID")
    message: str = Field(default="", description="消息")


# 解决自引用
AlgorithmVO.model_rebuild()
AlgorithmOptionVO.model_rebuild()
