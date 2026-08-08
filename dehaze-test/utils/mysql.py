"""调试工具库：pymysql 直连 MySQL。

对齐 dehaze-sdk-js/test/utils/mysql.ts：
- 单例连接
- 提供 query / query_one / execute / disconnect_mysql
- 默认 autocommit=True，方便调试脚本立即看到效果
- DictCursor，返回 dict 列表
"""
from __future__ import annotations

import pymysql

from . import config


_conn: pymysql.Connection | None = None


def get_conn(database: str | None = None) -> pymysql.Connection:
    """获取 MySQL 连接。

    Args:
        database: 指定数据库（默认 dehaze）；rebuild_mysql 时会切换 dehaze_test。
    """
    global _conn
    target_db = database or config.MYSQL_DATABASE
    if _conn is None or _conn.open is False:
        _conn = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=target_db,
            autocommit=True,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
            read_timeout=30,
            write_timeout=30,
        )
    return _conn


def query(sql: str, params: tuple | list | None = None, database: str | None = None) -> list[dict]:
    """执行 SELECT，返回 dict 列表。"""
    conn = get_conn(database)
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return list(cur.fetchall())


def query_one(sql: str, params: tuple | list | None = None, database: str | None = None) -> dict | None:
    """执行 SELECT，返回单条 dict（无结果返回 None）。"""
    rows = query(sql, params, database)
    return rows[0] if rows else None


def execute(sql: str, params: tuple | list | None = None, database: str | None = None) -> int:
    """执行 INSERT/UPDATE/DELETE，返回受影响行数（已 autocommit）。"""
    conn = get_conn(database)
    with conn.cursor() as cur:
        return cur.execute(sql, params or ())


def disconnect_mysql() -> None:
    """关闭 MySQL 连接。"""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# ===== 业务相关便捷查询（按需扩展） =====

def get_user_by_username(username: str) -> dict | None:
    return query_one("SELECT * FROM sys_user WHERE username = %s AND deleted = 0", (username,))


def get_unread_message_count(user_id: int) -> int:
    """直接查 MySQL 验证未读消息数。

    与后端 MessageServiceImpl.getUnreadCount() 逻辑对齐：
    - recipient_id = user_id（接收人）
    - read_status = 0（未读）
    - deleted = 0（MyBatis-Plus @TableLogic 自动加，DB 直查需显式带上）
    """
    row = query_one(
        "SELECT COUNT(*) AS cnt FROM sys_message "
        "WHERE read_status = 0 AND recipient_id = %s AND deleted = 0",
        (user_id,),
    )
    return row["cnt"] if row else 0
