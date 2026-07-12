"""
将 dehaze 数据库中的种子数据同步到 dehaze_test 数据库。
仅同步集成测试所需的数据：sys_user, sys_role, sys_user_role, sys_role_menu, sys_menu。
处理两个数据库间的列差异（dehaze_test 缺少 create_by/update_by/status 等列）。
"""
import pymysql
from datetime import datetime

DB_HOST = '127.0.0.1'
DB_PORT = 3306
DB_USER = 'root'
DB_PASS = '12345678'
SRC_DB = 'dehaze'
DST_DB = 'dehaze_test'

conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, charset='utf8mb4')
cur = conn.cursor()


def get_columns(db, table):
    cur.execute(f"DESCRIBE {db}.{table}")
    return [r[0] for r in cur.fetchall()]


def get_common_cols(src_db, dst_db, table):
    """返回两个数据库中同名表共有的列列表"""
    src_cols = set(get_columns(src_db, table))
    dst_cols = set(get_columns(dst_db, table))
    common = src_cols & dst_cols
    # 保持原始顺序
    return [c for c in get_columns(src_db, table) if c in common]


def sync_table(table, where_clause="1=1", replace=True):
    """从源数据库同步表数据到目标数据库，仅复制共有列"""
    cols = get_common_cols(SRC_DB, DST_DB, table)
    col_list = ", ".join([f"`{c}`" for c in cols])
    placeholders = ", ".join(["%s"] * len(cols))

    # 读取源数据
    cur.execute(f"SELECT {col_list} FROM {SRC_DB}.{table} WHERE {where_clause}")
    rows = cur.fetchall()
    print(f"  {table}: 读取 {len(rows)} 行 (列: {', '.join(cols)})")

    if not rows:
        return

    # 清空目标表（如果 replace=True）
    if replace:
        cur.execute(f"DELETE FROM {DST_DB}.{table}")
        print(f"  {table}: 已清空目标表")

    # 插入数据
    sql = f"INSERT INTO {DST_DB}.{table} ({col_list}) VALUES ({placeholders})"
    cur.executemany(sql, rows)
    print(f"  {table}: 已插入 {cur.rowcount} 行")


def sync_missing_rows(table, key_col, where_clause="1=1"):
    """仅同步目标表中不存在的行（按 key_col 判断）"""
    cols = get_common_cols(SRC_DB, DST_DB, table)
    col_list = ", ".join([f"`{c}`" for c in cols])
    placeholders = ", ".join(["%s"] * len(cols))

    # 读取源数据
    cur.execute(f"SELECT {col_list} FROM {SRC_DB}.{table} WHERE {where_clause}")
    rows = cur.fetchall()
    print(f"  {table}: 读取 {len(rows)} 行")

    if not rows:
        return

    # 获取目标表已有的 key 值
    cur.execute(f"SELECT {key_col} FROM {DST_DB}.{table}")
    existing_keys = set(r[0] for r in cur.fetchall())

    # 筛选需要插入的行
    key_idx = cols.index(key_col)
    new_rows = [r for r in rows if r[key_idx] not in existing_keys]
    print(f"  {table}: 需新增 {len(new_rows)} 行 (已存在 {len(existing_keys)} 行)")

    if not new_rows:
        return

    sql = f"INSERT INTO {DST_DB}.{table} ({col_list}) VALUES ({placeholders})"
    cur.executemany(sql, new_rows)
    print(f"  {table}: 已插入 {cur.rowcount} 行")


print("=" * 60)
print("开始同步种子数据: dehaze -> dehaze_test")
print("=" * 60)

try:
    # 1. sys_role: 全量替换（仅未删除的角色）
    print("\n[1/5] sys_role")
    sync_table("sys_role", where_clause="deleted=0", replace=True)

    # 2. sys_user: 全量替换（仅未删除的用户）
    print("\n[2/5] sys_user")
    sync_table("sys_user", where_clause="deleted=0", replace=True)

    # 3. sys_user_role: 全量替换
    print("\n[3/5] sys_user_role")
    sync_table("sys_user_role", replace=True)

    # 4. sys_role_menu: 全量替换
    print("\n[4/5] sys_role_menu")
    sync_table("sys_role_menu", replace=True)

    # 5. sys_menu: 仅补充缺失的行
    print("\n[5/5] sys_menu (仅补充缺失行)")
    sync_missing_rows("sys_menu", "id")

    conn.commit()
    print("\n" + "=" * 60)
    print("同步完成！数据已提交。")
    print("=" * 60)

    # 验证结果
    print("\n验证结果:")
    for t in ['sys_user', 'sys_role', 'sys_user_role', 'sys_role_menu', 'sys_menu']:
        cur.execute(f"SELECT COUNT(*) FROM {DST_DB}.{t}")
        count = cur.fetchone()[0]
        print(f"  {DST_DB}.{t}: {count} 行")

    # 验证关键用户
    print("\n关键用户:")
    cur.execute(f"SELECT id, username, nickname, status FROM {DST_DB}.sys_user WHERE username IN ('admin', 'test')")
    for r in cur.fetchall():
        print(f"  {r}")

    # 验证用户角色映射
    print("\n用户角色映射:")
    cur.execute(f"""
        SELECT u.username, r.code
        FROM {DST_DB}.sys_user u
        JOIN {DST_DB}.sys_user_role ur ON u.id = ur.user_id
        JOIN {DST_DB}.sys_role r ON ur.role_id = r.id
        WHERE u.username IN ('admin', 'test')
    """)
    for r in cur.fetchall():
        print(f"  {r}")

except Exception as e:
    conn.rollback()
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
