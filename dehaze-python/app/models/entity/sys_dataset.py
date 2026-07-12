"""
数据集相关实体模型
"""

from app.database import Base
from app.models.base import BaseModel
from sqlalchemy import BigInteger, Boolean, Index, Numeric, String, Text
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysDataset(BaseModel):
    __tablename__ = 'sys_dataset'
    __table_args__ = {'comment': '数据集表'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='数据集ID')
    parent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment='父数据集ID')
    tree_path: Mapped[str] = mapped_column(
        String(255), default='', comment='父节点ID路径')
    type: Mapped[str] = mapped_column(
        String(64), nullable=False, default='', comment='数据集类型')
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, default='', comment='数据集名称')
    img: Mapped[str | None] = mapped_column(Text, comment='数据集样例图片')
    description: Mapped[str] = mapped_column(
        String(2048), default='', comment='数据集描述')
    path: Mapped[str] = mapped_column(
        String(512), nullable=False, default='', comment='存储位置')
    size: Mapped[str] = mapped_column(
        String(100), default='', comment='占用空间大小')
    status: Mapped[int] = mapped_column(
        mysql_types.TINYINT, nullable=False, default=1, comment='状态(1:启用；0:禁用)')
    deleted: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=0, comment='逻辑删除标识(1:已删除;0:未删除)')


class SysDatasetItem(Base):
    __tablename__ = 'sys_dataset_item'
    __table_args__ = (
        Index('idx_dataset_id', 'dataset_id'),
        {'comment': '数据集与数据项关联表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment='所属数据集id')
    name: Mapped[str | None] = mapped_column(String(64), comment='数据项名称')
    create_time: Mapped[str | None] = mapped_column(
        String(32), comment='创建时间')
    update_time: Mapped[str | None] = mapped_column(
        String(32), comment='更新时间')


class SysItemFile(Base):
    __tablename__ = 'sys_item_file'
    __table_args__ = (
        Index('idx_item_id_file_id', 'item_id', 'file_id'),
        {'comment': '数据项图片关联表'}
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment='id')
    item_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment='所属数据项id')
    file_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment='文件id')
    thumbnail_file_id: Mapped[int | None] = mapped_column(
        BigInteger, comment='缩略图文件id')
    type: Mapped[str] = mapped_column(
        String(64), nullable=False, comment='图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)')
    scene_type: Mapped[str] = mapped_column(
        String(64), default='未分类', comment='场景类型')
    haze_level: Mapped[str] = mapped_column(
        String(32), default='未标注', comment='雾霾程度标识，支持多种规范：light/medium/heavy（人工分级），beta=0.5（β参数），A=0.8,beta=0.2（大气光A+β双参数），空值表示未标注或无雾')
    description: Mapped[str | None] = mapped_column(String(255), comment='描述')
