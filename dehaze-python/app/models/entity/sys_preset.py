"""参数预设实体"""
from sqlalchemy import BigInteger, Integer, JSON, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SysPreset(BaseModel):
    __tablename__ = 'sys_preset'
    __table_args__ = {'comment': '参数预设表'}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment='预设名称')
    type: Mapped[str] = mapped_column(String(16), nullable=False, default='custom', comment='预设类型(system:系统预设;custom:用户自定义)')
    algorithm_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment='关联算法ID')
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment='参数键值对(JSON)')
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment='所属用户ID(系统预设为空)')
    is_default: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False, default=0, comment='是否默认预设(0:否;1:是)')
