"""
pytest 配置和共享 fixtures
"""
import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app import create_app
from app.extensions import mysql
from app.models import SysUser, SysRole


@pytest.fixture(scope='session')
def app():
    """
    创建测试用的 Flask 应用实例（session级别）
    """
    # 设置测试环境
    os.environ['FLASK_ENV'] = 'testing'
    os.environ['TEST_DATABASE_TYPE'] = 'mysql'

    # 创建应用
    app = create_app('testing')

    # 为 PostgreSQL 创建测试数据库（如果需要）
    if 'postgresql' in app.config['SQLALCHEMY_DATABASE_URI']:
        _create_postgresql_test_db(app.config['SQLALCHEMY_DATABASE_URI'])

    yield app

    # 清理：删除 PostgreSQL 测试数据库
    if 'postgresql' in app.config['SQLALCHEMY_DATABASE_URI']:
        _drop_postgresql_test_db(app.config['SQLALCHEMY_DATABASE_URI'])


@pytest.fixture(scope='function')
def db_session(app):
    """
    为每个测试函数提供独立的数据库会话（function级别）
    在测试前创建所有表，测试后删除所有表并清理会话
    """
    with app.app_context():
        # 创建所有表
        mysql.create_all()

        # 提供数据库会话
        yield mysql.session

        # 测试后清理
        mysql.session.remove()
        mysql.drop_all()


@pytest.fixture
def client(app):
    """
    Flask 测试客户端
    """
    return app.test_client()


@pytest.fixture
def runner(app):
    """
    Flask CLI runner
    """
    return app.test_cli_runner()


@pytest.fixture
def sample_roles(db_session):
    """
    创建测试角色数据
    """
    role1 = SysRole(
        name='管理员',
        code='ADMIN',
        sort=1,
        status=1,
        data_scope=1
    )
    role2 = SysRole(
        name='普通用户',
        code='USER',
        sort=2,
        status=1,
        data_scope=2
    )

    db_session.add(role1)
    db_session.add(role2)
    db_session.commit()

    return {'admin': role1, 'user': role2}


@pytest.fixture
def sample_user(db_session):
    """
    创建测试用户
    """
    from werkzeug.security import generate_password_hash

    user = SysUser(
        username='testuser',
        nickname='Test User',
        password=generate_password_hash('password123'),
        gender=1,
        dept_id=1,
        mobile='13800138000',
        email='test@example.com',
        status=1,
        deleted=0
    )

    db_session.add(user)
    db_session.commit()

    return user


# ============ 辅助函数 ============

def _create_postgresql_test_db(db_uri):
    """
    为 PostgreSQL 创建测试数据库
    """
    try:
        # 解析数据库 URI
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(db_uri)
        db_name = parsed.path.lstrip('/')

        # 连接到默认的 postgres 数据库
        default_db_uri = urlunparse((
            parsed.scheme,
            parsed.netloc,
            '/postgres',
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        engine = create_engine(default_db_uri, isolation_level='AUTOCOMMIT')

        with engine.connect() as conn:
            # 检查数据库是否存在
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            )
            exists = result.fetchone() is not None

            if not exists:
                # 创建数据库
                conn.execute(text(f'CREATE DATABASE {db_name}'))
                print(f"✓ 已创建 PostgreSQL 测试数据库: {db_name}")

        engine.dispose()

    except OperationalError as e:
        print(f"警告: 无法创建 PostgreSQL 测试数据库: {e}")
        print("请确保 PostgreSQL 服务正在运行，并且用户有创建数据库的权限")


def _drop_postgresql_test_db(db_uri):
    """
    删除 PostgreSQL 测试数据库
    """
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(db_uri)
        db_name = parsed.path.lstrip('/')

        # 连接到默认的 postgres 数据库
        default_db_uri = urlunparse((
            parsed.scheme,
            parsed.netloc,
            '/postgres',
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        engine = create_engine(default_db_uri, isolation_level='AUTOCOMMIT')

        with engine.connect() as conn:
            # 终止所有连接到测试数据库的会话
            conn.execute(text(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """))

            # 删除数据库
            conn.execute(text(f'DROP DATABASE IF EXISTS {db_name}'))
            print(f"✓ 已删除 PostgreSQL 测试数据库: {db_name}")

        engine.dispose()

    except OperationalError as e:
        print(f"警告: 无法删除 PostgreSQL 测试数据库: {e}")
    except Exception as e:
        print(f"警告: 删除 PostgreSQL 测试数据库时发生错误: {e}")