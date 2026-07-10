"""
菜单相关实体模型
"""

from typing import Optional

from app.database import Base
from app.models.base import BaseModel
from sqlalchemy import BigInteger, Column, Integer, String
from sqlalchemy.dialects import mysql as mysql_types
from sqlalchemy.orm import Mapped, mapped_column


class SysMenu(BaseModel):
    __tablename__ = 'sys_menu'
    __table_args__ = {'comment': '菜单管理'}

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True)
    parent_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, comment='父菜单ID')
    tree_path: Mapped[str] = mapped_column(
        String(255), default=',', comment='父节点ID路径，格式如 ",1,2,"')
    name: Mapped[str] = mapped_column(
        String(64), nullable=False, default='', comment='菜单名称')
    type: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False,
                                      comment='菜单类型(1:目录 2:菜单 3:外链 4:按钮)')
    path: Mapped[str] = mapped_column(
        String(128), default='', comment='路由路径(浏览器地址栏路径)')
    component: Mapped[Optional[str]] = mapped_column(
        String(128), comment='组件路径(vue页面完整路径，省略.vue后缀)')
    perm: Mapped[Optional[str]] = mapped_column(String(128), comment='权限标识')
    visible: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False,
                                         default=1, comment='显示状态(1:显示 0:隐藏)')
    status: Mapped[int] = mapped_column(mysql_types.TINYINT, nullable=False,
                                        default=1, comment='状态(1:启用 0:禁用)')
    sort: Mapped[int] = mapped_column(Integer, default=0, comment='排序')
    icon: Mapped[str] = mapped_column(String(64), default='', comment='菜单图标')
    redirect: Mapped[Optional[str]] = mapped_column(
        String(128), comment='跳转路径')
    always_show: Mapped[int] = mapped_column(mysql_types.TINYINT, default=0,
                                             comment='【目录】只有一个子路由是否始终显示(1:是 0:否)')
    keep_alive: Mapped[int] = mapped_column(
        mysql_types.TINYINT, default=0, comment='【菜单】是否开启页面缓存(1:是 0:否)')
    deleted: Mapped[int] = mapped_column(mysql_types.TINYINT, default=0,
                                         comment='逻辑删除标识(1:已删除;0:未删除)')


class SysRoleMenu(Base):
    """角色菜单关联表（无时间字段）"""
    __tablename__ = 'sys_role_menu'
    __table_args__ = {'comment': '角色和菜单关联表'}

    role_id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                         nullable=False, comment='角色ID')
    menu_id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                         nullable=False, comment='菜单ID')
