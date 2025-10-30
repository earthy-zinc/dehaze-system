from datetime import datetime, timezone

from sqlalchemy import CHAR, JSON, VARCHAR, BigInteger, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects import mysql as mysql_types

from app.extensions import mysql


class SysFile(mysql.Model):
    __tablename__ = 'sys_file'
    __table_args__ = (
        Index('md5_key', 'md5', unique=True),
        {'comment': '文件表'}
    )

    id = Column(Integer, primary_key=True, autoincrement=True, comment='文件id')
    type = Column(String(100), nullable=True, comment='文件类型')
    url = Column(Text, nullable=True, comment='文件url')
    name = Column(String(100), nullable=False, comment='文件原始名')
    object_name = Column(String(100), nullable=False, comment='文件存储名')
    size = Column(String(100), nullable=False, default='0', comment='文件大小')
    path = Column(String(255), nullable=False, comment='文件路径')
    md5 = Column(CHAR(32), nullable=False, unique=True,
                 comment='文件的MD5值，用于比对文件是否相同')
    create_time = Column(DateTime, nullable=False, comment='创建时间')
    update_time = Column(DateTime, nullable=True, comment='更新时间')


class SysAlgorithm(mysql.Model):
    __tablename__ = 'sys_algorithm'
    __table_args__ = {'comment': '算法模型表'}

    id = Column(BigInteger, primary_key=True,
                autoincrement=True, comment='模型id')
    parent_id = Column(BigInteger, default=0, comment='模型的父id')
    type = Column(String(100), default='', comment='模型类型')
    name = Column(String(64), nullable=False, comment='模型名称')
    img = Column(Text, comment='模型图片')
    path = Column(String(255), default='', comment='模型存储路径')
    size = Column(String(100), comment='模型大小')
    params = Column(String(255), comment='模型参数')
    flops = Column(String(255), comment='模型浮点运算次数')
    import_path = Column(String(255), comment='模型代码导入路径')
    description = Column(String(2048), comment='针对该模型的详细描述')
    status = Column(mysql_types.TINYINT, default=1, comment='状态(1:启用；0:禁用)')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


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


class SysDept(mysql.Model):
    __tablename__ = 'sys_dept'
    __table_args__ = {'comment': '部门表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    name = Column(String(64), nullable=False, default='', comment='部门名称')
    parent_id = Column(BigInteger, nullable=False, default=0, comment='父节点id')
    tree_path = Column(String(255), default='', comment='父节点id路径')
    sort = Column(Integer, default=0, comment='显示顺序')
    status = Column(mysql_types.TINYINT, nullable=False,
                    default=1, comment='状态(1:正常;0:禁用)')
    deleted = Column(mysql_types.TINYINT, default=0,
                     comment='逻辑删除标识(1:已删除;0:未删除)')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


class SysDict(mysql.Model):
    __tablename__ = 'sys_dict'
    __table_args__ = {'comment': '字典数据表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键')
    type_code = Column(String(64), comment='字典类型编码')
    name = Column(String(50), default='', comment='字典项名称')
    value = Column(String(50), default='', comment='字典项值')
    sort = Column(Integer, default=0, comment='排序')
    status = Column(mysql_types.TINYINT, default=0, comment='状态(1:正常;0:禁用)')
    defaulted = Column(mysql_types.TINYINT, default=0, comment='是否默认(1:是;0:否)')
    remark = Column(String(255), default='', comment='备注')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class SysDictType(mysql.Model):
    __tablename__ = 'sys_dict_type'
    __table_args__ = (
        Index('type_code', 'code', unique=True),
        {'comment': '字典类型表'}
    )

    id = Column(BigInteger, primary_key=True,
                autoincrement=True, comment='主键 ')
    name = Column(String(50), default='', comment='类型名称')
    code = Column(String(50), default='', comment='类型编码')
    status = Column(mysql_types.TINYINT, default=0, comment='状态(0:正常;1:禁用)')
    remark = Column(String(255), comment='备注')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='更新时间')


class SysEvalLog(mysql.Model):
    __tablename__ = 'sys_eval_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_pred_md5', 'pred_md5'),
        Index('idx_gt_md5', 'gt_md5'),
        {'comment': '模型预测日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    pred_file_id = Column(BigInteger, comment='预测图像文件id')
    pred_md5 = Column(CHAR(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    gt_file_id = Column(BigInteger, comment='真值图像文件id')
    gt_md5 = Column(CHAR(32), nullable=False, comment='真值图像md5值')
    gt_url = Column(Text, nullable=False, comment='真值图像url')
    time = Column(Integer, default=0, comment='评估时间（秒）')
    result = Column(JSON, comment='预测结果')
    create_time = Column(DateTime, nullable=False,
                         default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(
        timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


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
    description = Column(String(255), comment='描述')


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


class SysPredLog(mysql.Model):
    __tablename__ = 'sys_pred_log'
    __table_args__ = (
        Index('idx_algorithm_id', 'algorithm_id'),
        Index('idx_origin_md5', 'origin_md5'),
        Index('idx_pred_md5', 'pred_md5'),
        {'comment': '模型预测日志表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    origin_file_id = Column(BigInteger, comment='原始图像文件id（有雾图像）')
    origin_md5 = Column(CHAR(32), nullable=False, comment='原始图像md5值')
    origin_url = Column(Text, nullable=False, comment='原始图像url')
    pred_file_id = Column(BigInteger, comment='预测图像文件id')
    pred_md5 = Column(CHAR(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    time = Column(Integer, default=0, comment='推理时间（秒）')
    create_time = Column(DateTime, nullable=False,
                         default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(
        timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, comment='创建人ID')
    update_by = Column(BigInteger, comment='修改人ID')


class SysRole(mysql.Model):
    __tablename__ = 'sys_role'
    __table_args__ = (
        Index('idx_sys_role_name', 'name', unique=True),
        {'comment': '角色表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, comment='角色名称')
    code = Column(String(32), comment='角色编码')
    sort = Column(BigInteger, comment='显示顺序')
    status = Column(mysql_types.TINYINT, default=1,
                    comment='角色状态(1-正常；0-停用)')
    data_scope = Column(mysql_types.TINYINT,
                        comment='数据权限(0-所有数据；1-部门及子部门数据；2-本部门数据；3-本人数据)')
    deleted = Column(mysql_types.TINYINT, nullable=False,
                     default=0, comment='逻辑删除标识(0-未删除；1-已删除)')
    create_time = Column(DateTime)
    update_time = Column(DateTime)


class SysRoleMenu(mysql.Model):
    __tablename__ = 'sys_role_menu'
    __table_args__ = {'comment': '角色和菜单关联表'}

    role_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='角色ID')
    menu_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='菜单ID')


class SysUser(mysql.Model):
    __tablename__ = 'sys_user'
    __table_args__ = (
        Index('idx_sys_user_username', 'username', unique=True),
        {'comment': '用户信息表'}
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), comment='用户名')
    nickname = Column(String(64), comment='昵称')
    gender = Column(mysql_types.TINYINT, default=1, comment='性别((1:男')
    password = Column(String(100), comment='密码')
    dept_id = Column(BigInteger, comment='部门ID')
    avatar = Column(Text, comment='用户头像')
    mobile = Column(String(20), comment='联系方式')
    status = Column(mysql_types.TINYINT, default=1, comment='用户状态((1:正常')
    email = Column(String(128), comment='用户邮箱')
    deleted = Column(mysql_types.TINYINT, default=0, comment='逻辑删除标识(0:未删除')
    create_time = Column(DateTime)
    update_time = Column(DateTime)


class SysUserRole(mysql.Model):
    __tablename__ = 'sys_user_role'
    __table_args__ = {'comment': '用户和角色关联表'}

    user_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='用户ID')
    role_id = Column(BigInteger, primary_key=True,
                     nullable=False, comment='角色ID')


class SysWpxFile(mysql.Model):
    __tablename__ = 'sys_wpx_file'
    __table_args__ = {'comment': 'WPX文件表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    origin_file_id = Column(BigInteger, comment='旧文件id')
    origin_md5 = Column(CHAR(32), unique=True,
                        nullable=False, comment='旧文件的MD5值')
    origin_path = Column(VARCHAR(255), nullable=False, comment='旧文件路径')
    new_file_id = Column(BigInteger, comment='新文件id')
    new_path = Column(VARCHAR(255), nullable=False, comment='新文件路径')
    new_md5 = Column(CHAR(32), unique=True, nullable=False, comment='新文件的MD5值')
