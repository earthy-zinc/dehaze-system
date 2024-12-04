from datetime import datetime, timezone

from sqlalchemy import Column, String, BigInteger, Text, DateTime, SmallInteger, Integer, CHAR, VARCHAR, Float, func

from app.extensions import mysql

class SysFile(mysql.Model):
    __tablename__ = 'sys_file'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='文件id')
    type = Column(String(100), nullable=True, comment='文件类型')
    url = Column(String(255), nullable=True, comment='文件url')
    name = Column(String(100), nullable=False, comment='文件原始名')
    object_name = Column(String(100), nullable=False, comment='文件存储名')
    size = Column(String(100), nullable=False, default='0', comment='文件大小')
    path = Column(String(255), nullable=False, comment='文件路径')
    md5 = Column(String(32), unique=True, nullable=False, comment='文件的MD5值，用于比对文件是否相同')
    create_time = Column(DateTime, nullable=False, default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=True, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')

class SysAlgorithm(mysql.Model):
    __tablename__ = 'sys_algorithm'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='模型id')
    parent_id = Column(BigInteger, default=0, comment='模型的父id')
    type = Column(String(100), default='', comment='模型类型')
    name = Column(String(64), nullable=False, comment='模型名称')
    img = Column(Text, default=None, comment='模型图片')
    path = Column(String(255), default='', comment='模型存储路径')
    size = Column(String(100), default=None, comment='模型大小')
    params = Column(String(255), default=None, comment='模型参数')
    flops = Column(String(255), default=None, comment='模型FLOPs')
    import_path = Column(String(255), default='', comment='模型代码导入路径')
    description = Column(Text, default=None, comment='针对该模型的详细描述')
    status = Column(SmallInteger, default=1, comment='状态(1:启用；0:禁用)')
    create_time = Column(DateTime, default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, nullable=True, comment='创建人ID')
    update_by = Column(BigInteger, nullable=True, comment='修改人ID')

class SysWpxFile(mysql.Model):
    __tablename__ = 'sys_wpx_file'
    __table_args__ = {'comment': 'WPX文件表'}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    origin_file_id = Column(BigInteger, nullable=False, comment='旧文件id')
    origin_md5 = Column(CHAR(32), unique=True, nullable=False, comment='旧文件的MD5值')
    origin_path = Column(VARCHAR(255), nullable=False, comment='旧文件路径')
    new_path = Column(VARCHAR(255), nullable=False, comment='新文件路径')
    new_md5 = Column(CHAR(32), unique=True, nullable=False, comment='新文件的MD5值')
    new_file_id = Column(BigInteger, nullable=False, comment='新文件id')


class SysPredLog(mysql.Model):
    __tablename__ = 'sys_pred_log'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    origin_file_id = Column(BigInteger, nullable=True, comment='原始图像文件id（有雾图像）')
    origin_md5 = Column(String(32), nullable=False, comment='原始图像md5值')
    origin_url = Column(Text, nullable=False, comment='原始图像url')
    pred_file_id = Column(BigInteger, nullable=True, comment='预测图像文件id')
    pred_md5 = Column(String(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    time = Column(Integer, default=0, comment='推理时间（秒）')
    create_time = Column(DateTime, nullable=False, default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, nullable=True, comment='创建人ID')
    update_by = Column(BigInteger, nullable=True, comment='修改人ID')

class SysEvalLog(mysql.Model):
    __tablename__ = 'sys_eval_log'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='id')
    algorithm_id = Column(BigInteger, nullable=False, comment='算法id')
    origin_file_id = Column(BigInteger, nullable=True, comment='原始图像文件id（有雾图像）')
    origin_md5 = Column(String(32), nullable=False, comment='原始图像md5值')
    origin_url = Column(Text, nullable=False, comment='原始图像url')
    pred_file_id = Column(BigInteger, nullable=True, comment='预测图像文件id')
    pred_md5 = Column(String(32), nullable=False, comment='预测图像md5值')
    pred_url = Column(Text, nullable=False, comment='预测图像url')
    gt_file_id = Column(BigInteger, nullable=True, comment='真值图像文件id')
    gt_md5 = Column(String(32), nullable=False, comment='真值图像md5值')
    gt_url = Column(Text, nullable=False, comment='真值图像url')
    time = Column(Integer, default=0, comment='评估时间（秒）')
    psnr = Column(Float, default=0, comment='PSNR')
    ssim = Column(Float, default=0, comment='SSIM')
    niqe = Column(Float, default=0, comment='NIQE')
    nima = Column(Float, default=0, comment='NIMA')
    brisque = Column(Float, default=0, comment='BRISQUE')
    create_time = Column(DateTime, nullable=False, default=datetime.now(timezone.utc), comment='创建时间')
    update_time = Column(DateTime, nullable=False, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment='更新时间')
    create_by = Column(BigInteger, nullable=True, comment='创建人ID')
    update_by = Column(BigInteger, nullable=True, comment='修改人ID')
