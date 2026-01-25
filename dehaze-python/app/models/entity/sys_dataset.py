"""
数据集相关实体模型
"""

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, Numeric, String, Text, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysDataset(mysql.Model):
    __tablename__ = 'sys_dataset'
    __table_args__ = {'comment': '数据集表'}

    id = Column(BigInteger, primary_key=True,
                autoincrement=True, comment='数据集ID')
    parent_id = Column(BigInteger, nullable=False, default=0, comment='父数据集ID')
    tree_path = Column(String(255), default='', comment='父节点ID路径')
    type = Column(String(64), nullable=False, default='', comment='数据集类型')
    name = Column(String(64), nullable=False, default='', comment='数据集名称')
    img = Column(Text, comment='数据集样例图片')
    description = Column(String(2048), default='', comment='数据集描述')
    path = Column(String(512), nullable=False, default='', comment='存储位置')
    size = Column(String(100), default='', comment='占用空间大小')
    status = Column(mysql_types.TINYINT, nullable=False,
                    default=1, comment='状态(1:启用；0:禁用)')
    deleted = Column(mysql_types.TINYINT, default=0,
                     comment='逻辑删除标识(1:已删除;0:未删除)')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


class SysDatasetItem(mysql.Model):
    __tablename__ = 'sys_dataset_item'
    __table_args__ = (
        Index('idx_dataset_id', 'dataset_id'),
        {'comment': '数据集与数据项关联表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    dataset_id = Column(BigInteger, nullable=False, comment='所属数据集id')
    name = Column(String(64), comment='数据项名称')


class SysItemFile(mysql.Model):
    __tablename__ = 'sys_item_file'
    __table_args__ = (
        Index('idx_item_id_file_id', 'item_id', 'file_id'),
        {'comment': '数据项图片关联表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    item_id = Column(BigInteger, nullable=False, comment='所属数据项id')
    file_id = Column(BigInteger, nullable=False, comment='文件id')
    thumbnail_file_id = Column(BigInteger, comment='缩略图文件id')
    type = Column(String(64), nullable=False, comment='图片类型（清晰图、雾霾图、分割图等）')
    scene_type = Column(String(64), default='未分类', comment='场景类型')
    haze_level = Column(String(32), default='未标注', comment='雾霾等级')
    quality_score = Column(Numeric(5, 2), nullable=True, comment='质量分数')
    dehaze_algorithm = Column(String(64), default='', comment='去雾算法')
    is_labeled = Column(Boolean, default=False, comment='是否已标注')
    description = Column(String(255), comment='描述')
