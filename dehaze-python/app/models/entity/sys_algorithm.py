"""
算法模型实体
对齐 dehaze-java SysAlgorithm.java + BaseEntity
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel, SoftDeleteMixin


class SysAlgorithm(BaseModel, SoftDeleteMixin):
    """算法模型表 (对齐 Java SysAlgorithm + BaseEntity)"""

    __tablename__ = "sys_algorithm"
    __table_args__ = (
        Index("idx_algorithm_parent_id", "parent_id"),
        Index("idx_algorithm_status", "status"),
        Index("idx_algorithm_name", "name"),
        {"comment": "算法模型表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="模型id"
    )
    parent_id: Mapped[int | None] = mapped_column(BigInteger, default=0, comment="模型的父id")
    type: Mapped[str] = mapped_column(String(100), default="", comment="模型类型")
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="模型名称")
    # 版本号 (对齐 Java schema/sys_algorithm.sql: varchar(50))
    version: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="算法版本号")
    img: Mapped[str | None] = mapped_column(Text, comment="模型图片")
    path: Mapped[str] = mapped_column(String(255), default="", comment="模型存储路径")
    size: Mapped[str | None] = mapped_column(String(100), comment="模型大小")
    params: Mapped[str | None] = mapped_column(String(255), comment="模型参数")
    flops: Mapped[str | None] = mapped_column(String(255), comment="模型浮点运算次数")
    import_path: Mapped[str | None] = mapped_column(String(255), comment="模型代码导入路径")
    description: Mapped[str | None] = mapped_column(String(2048), comment="针对该模型的详细描述")
    # 状态机：1=草稿 2=测试中 3=待审核 4=已发布 5=已停用 6=已归档
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT,
        default=1,
        comment="状态(1:草稿;2:测试中;3:待审核;4:已发布;5:已停用;6:已归档)",
    )
    # 审核字段 (对齐 Java schema/sys_algorithm.sql: audit_by bigint, audit_remark varchar(500))
    audit_by: Mapped[int | None] = mapped_column(BigInteger, comment="审核人ID")
    audit_time: Mapped[datetime | None] = mapped_column(DateTime, comment="审核时间")
    audit_remark: Mapped[str | None] = mapped_column(String(500), comment="审核备注")


class SysAlgorithmVersion(BaseModel, SoftDeleteMixin):
    """算法版本历史表 (对齐 Java SysAlgorithmVersion.java + BaseEntity)

    字段完全对齐 config/sql/schema/sys_algorithm_version.sql 中表定义.
    """

    __tablename__ = "sys_algorithm_version"
    __table_args__ = (
        UniqueConstraint("algorithm_id", "version", name="uk_algo_version"),
        Index("idx_algo_version_algo_id", "algorithm_id"),
        {"comment": "算法版本历史表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="关联算法ID")
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment="版本号")
    change_log: Mapped[str | None] = mapped_column(Text, comment="变更日志")
    status: Mapped[int | None] = mapped_column(Integer, comment="该版本时的状态")
    config_json: Mapped[str | None] = mapped_column(Text, comment="该版本时的配置JSON")
    model_file_id: Mapped[int | None] = mapped_column(BigInteger, comment="模型文件ID")
    is_active: Mapped[int | None] = mapped_column(
        mysql_types.TINYINT(1), default=0, comment="是否当前活跃版本"
    )
