"""
菜单相关实体模型
"""

from sqlalchemy import Column, DateTime, Integer, String, BigInteger
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysMenu(mysql.Model):
    __tablename__ = 'sys_menu'
    __table_args__ = {'comment': '菜单管理'}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    parent_id = Column(BigInteger, nullable=False, comment='父菜单ID')
    tree_path = Column(String(255), comment='父节点ID路径')
    name = Column(String(64), nullable=False, default='', comment='菜单名称')
    type = Column(mysql_types.TINYINT, nullable=False,
                  comment='菜单类型(1:菜单 2:目录 3:外链 4:按钮)')
    path = Column(String(128), default='', comment='路由路径(浏览器地址栏路径)')
    component = Column(String(128), comment='组件路径(vue页面完整路径，省略.vue后缀)')
    perm = Column(String(128), comment='权限标识')
    visible = Column(mysql_types.TINYINT, nullable=False,
                     default=1, comment='显示状态(1-显示')
    sort = Column(Integer, default=0, comment='排序')
    icon = Column(String(64), default='', comment='菜单图标')
    redirect = Column(String(128), comment='跳转路径')
    create_time = Column(DateTime)
    update_time = Column(DateTime)
    always_show = Column(mysql_types.TINYINT,
                         comment='【目录】只有一个子路由是否始终显示(1:是 0:否)')
    keep_alive = Column(mysql_types.TINYINT, comment='【菜单】是否开启页面缓存(1:是 0:否)')


class SysRoleMenu(mysql.Model):
    __tablename__ = 'sys_role_menu'
    __table_args__ = {'comment': '角色和菜单关联表'}

    role_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='角色ID')
    menu_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='菜单ID')
