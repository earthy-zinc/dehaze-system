"""
测试专用模型

使用 SQLite 兼容的数据类型，用于单元测试
"""

from datetime import datetime

from sqlalchemy import BigInteger, Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class MockBase(DeclarativeBase):
    """测试模型基类"""

    pass


class MockUser(MockBase):
    """测试用户模型（SQLite 兼容）"""

    __tablename__ = "test_user"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    nickname = Column(String(64))
    password = Column(String(100), nullable=False)
    gender = Column(Integer, default=1)
    dept_id = Column(BigInteger)
    avatar = Column(Text)
    mobile = Column(String(20))
    status = Column(Integer, default=1)
    email = Column(String(128))
    deleted = Column(Integer, default=0)
    create_time = Column(String(30), default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    update_time = Column(String(30), onupdate=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    create_by = Column(BigInteger)
    update_by = Column(BigInteger)


class MockRole(MockBase):
    """测试角色模型（SQLite 兼容）"""

    __tablename__ = "test_role"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    code = Column(String(32))
    sort = Column(BigInteger)
    status = Column(Integer, default=1)
    data_scope = Column(Integer)
    deleted = Column(Integer, default=0)
    create_time = Column(String(30), default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    update_time = Column(String(30), onupdate=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class MockUserRole(MockBase):
    """测试用户角色关联模型"""

    __tablename__ = "test_user_role"

    user_id = Column(BigInteger, primary_key=True, nullable=False)
    role_id = Column(BigInteger, primary_key=True, nullable=False)
