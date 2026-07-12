"""
数据库刷新脚本：扫描 D:\\DeepLearning\\dataset 重建 sys_file/sys_dataset_item/sys_item_file

前置条件：
1. docker-compose 已启动（MySQL + MinIO）
2. MinIO 容器已 bind mount D:/DeepLearning/dataset → /data/dehaze
3. .env 中已设置 DEHAZE_PASSWORD 和 MINIO_ACCESS_KEY=admin

执行：
    python scripts/refresh_dataset_db.py

说明：
- 清空 sys_file / sys_dataset_item / sys_item_file 三张表后重建
- 文件 URL 直连 MinIO：http://127.0.0.1:9000/dehaze/{object_name}
- object_name = {dataset_path}/{subdir}/{filename}（与 MinIO bucket 内路径一致）
- 配对策略：按文件名前导数字分组（如 01_GT.png 与 01_hazy.png 都归到 "01" 组）
- MD5 字段使用 object_name 的 MD5（保证 UNIQUE 约束，不读取文件内容以提速）
"""

import hashlib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pymysql
from dotenv import load_dotenv

# ============================== 配置 ==============================

DATASET_ROOT = Path(r"D:\DeepLearning\dataset")
MINIO_URL_BASE = "http://127.0.0.1:9000/dehaze"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_DB = "dehaze"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
BATCH_SIZE = 2000  # 批量插入大小

# 叶子数据集配置：(dataset_id, dataset_path, clear_subdir, hazy_subdir, trans_subdir)
# clear_subdir / hazy_subdir / trans_subdir 为 None 表示该数据集没有此类型子目录
# 子目录名严格按磁盘实际名称（大小写敏感）
LEAF_DATASETS = [
    (1, "Dense-Haze", "clean", "hazy", None),
    (2, "O-HAZE", "clean", "hazy", None),
    (3, "I-HAZE", "clean", "hazy", None),
    (5, "NH-HAZE-2020", "clean", "hazy", None),
    (6, "NH-HAZE-2021", "clean", "hazy", None),
    (7, "NH-HAZE-2023", "clean", "hazy", None),
    (9, "RESIDE/ITS", "clear", "hazy", "trans"),
    (10, "RESIDE/OTS", "clear", "haze", None),
    (15, "RESIDE-6k/train", "GT", "hazy", None),
    (16, "RESIDE-6k/test", "GT", "hazy", None),
    (18, "RESIDE-IN/train", "GT", "hazy", None),
    (19, "RESIDE-IN/test", "GT", "hazy", None),
    (21, "RESIDE-OUT/train", "GT", "hazy", None),
    (22, "RESIDE-OUT/test", "GT", "hazy", None),
    (24, "RSHAZE/train", "GT", "hazy", None),
    (25, "RSHAZE/test", "GT", "hazy", None),
]

# 前导数字提取正则（匹配文件名开头的数字部分）
_LEADING_NUM_RE = re.compile(r"^(\d+)")


# ============================== 工具函数 ==============================

def load_password() -> str:
    """从 .env 加载数据库密码"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    password = os.getenv("DEHAZE_PASSWORD", "")
    if not password:
        print(f"[FATAL] 未在 {env_path} 中找到 DEHAZE_PASSWORD", file=sys.stderr)
        sys.exit(1)
    return password


def leading_number(filename: str) -> Optional[str]:
    """提取文件名前导数字，用于配对分组"""
    m = _LEADING_NUM_RE.match(filename)
    return m.group(1) if m else None


def fake_md5(object_name: str) -> str:
    """基于 object_name 生成确定性 MD5（保证 UNIQUE 约束）"""
    return hashlib.md5(object_name.encode("utf-8")).hexdigest()


def format_size(size_bytes: int) -> str:
    """字节数格式化显示"""
    if size_bytes == 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = min(len(units) - 1, (size_bytes.bit_length() - 1) // 10)
    p = 1 << (i * 10)
    return f"{size_bytes / p:.2f} {units[i]}"


def scan_image_dir(dir_path: Path) -> list[Path]:
    """扫描目录下所有图片文件（大小写不敏感扩展名匹配）"""
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    result = []
    for entry in dir_path.iterdir():
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            result.append(entry)
    return result


def build_object_name(dataset_path: str, subdir: str, filename: str) -> str:
    """构造 MinIO object_name（与 bucket 内相对路径一致）"""
    return f"{dataset_path}/{subdir}/{filename}"


def build_url(object_name: str) -> str:
    """构造文件访问 URL（直连 MinIO）"""
    return f"{MINIO_URL_BASE}/{object_name}"


def parse_haze_level(filename: str) -> str:
    """
    从文件名解析雾霾程度标注

    支持的格式：
    - {id}_{idx}_{beta}.png       → "beta={beta}"（RESIDE/ITS、RSHAZE）
    - {id}_{A}_{beta}.jpg         → "A={A},beta={beta}"（RESIDE/OTS）
    - {id}_{beta}_{A}.png         → "beta={beta},A={A}"（Haze4K）
    - {id}_hazy_light.jpg         → "light"
    - {id}_hazy_medium.jpg        → "medium"
    - {id}_hazy_heavy.jpg         → "heavy"
    - {id}_hazy.jpg               → ""（未标注）
    - {id}.png                    → ""（未标注）
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    # 形如 1000_1_0.74905 → 3 段，最后一段是浮点数 → 视为 beta
    if len(parts) >= 3:
        last = parts[-1]
        mid = parts[-2]
        try:
            float(last)
            # 最后一段是浮点数，判定为物理参数
            # 进一步判定倒数第二段：若也是浮点数则为 A+β，否则为 idx+β
            try:
                float(mid)
                # 两段都是浮点数：A+β 或 β+A
                # RESIDE/OTS 命名 0025_0.8_0.2 → A=0.8, β=0.2
                # Haze4K 命名 1012_0.85_1.28 → β=0.85, A=1.28
                # 两者难以仅靠数值区分，统一记为 A={mid},beta={last}
                return f"A={mid},beta={last}"
            except ValueError:
                # 倒数第二段不是浮点数（如 idx），则最后一段是 β
                return f"beta={last}"
        except ValueError:
            pass

    # 形如 xxx_hazy_light / xxx_hazy_medium / xxx_hazy_heavy
    lower = stem.lower()
    if lower.endswith("_hazy_light") or lower.endswith("_light"):
        return "light"
    if lower.endswith("_hazy_medium") or lower.endswith("_medium"):
        return "medium"
    if lower.endswith("_hazy_heavy") or lower.endswith("_heavy"):
        return "heavy"

    return ""


# ============================== 数据库操作 ==============================

def get_connection(password: str):
    """获取 MySQL 连接"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=password,
        database=MYSQL_DB,
        charset="utf8mb4",
        autocommit=False,
    )


def truncate_tables(cursor):
    """清空三张表（按外键依赖顺序）"""
    print("[STEP] 清空 sys_item_file / sys_dataset_item / sys_file ...")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("TRUNCATE TABLE sys_item_file")
    cursor.execute("TRUNCATE TABLE sys_dataset_item")
    cursor.execute("TRUNCATE TABLE sys_file")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    print("[STEP] 三张表已清空")


def insert_batch(cursor, table: str, columns: list[str], rows: list[tuple]):
    """批量插入"""
    if not rows:
        return
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    cursor.executemany(sql, rows)


# ============================== 主流程 ==============================

def process_dataset(cursor, dataset_id: int, dataset_path: str,
                    clear_subdir: Optional[str], hazy_subdir: Optional[str],
                    trans_subdir: Optional[str]) -> dict:
    """
    处理单个数据集：扫描文件夹 → 配对 → 插入数据库

    Returns:
        统计信息 dict
    """
    root = DATASET_ROOT / dataset_path
    if not root.exists():
        print(f"  [SKIP] 数据集路径不存在: {root}")
        return {"files": 0, "items": 0, "skipped": True}

    # 扫描各类子目录
    clear_files = scan_image_dir(root / clear_subdir) if clear_subdir else []
    hazy_files = scan_image_dir(root / hazy_subdir) if hazy_subdir else []
    trans_files = scan_image_dir(root / trans_subdir) if trans_subdir else []

    print(f"  扫描完成: clear={len(clear_files)}, hazy={len(hazy_files)}, trans={len(trans_files)}")

    # 按 leading_number 建立索引
    clear_map: dict[str, Path] = {}
    for f in clear_files:
        num = leading_number(f.name) or f.stem
        clear_map[num] = f

    hazy_map: dict[str, list[Path]] = {}
    for f in hazy_files:
        num = leading_number(f.name) or f.stem
        hazy_map.setdefault(num, []).append(f)

    trans_map: dict[str, list[Path]] = {}
    for f in trans_files:
        num = leading_number(f.name) or f.stem
        trans_map.setdefault(num, []).append(f)

    # 收集所有分组键（clear 和 hazy 的并集）
    all_keys = set(clear_map.keys()) | set(hazy_map.keys())

    file_rows: list[tuple] = []
    item_rows: list[tuple] = []
    item_file_rows: list[tuple] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_count = 0
    item_count = 0

    # 由于 sys_file.id 是 AUTO_INCREMENT，先插入文件获取 ID，再插入 item 和关联
    # 为提高性能，采用：先批量插入所有文件，再查询 object_name → id 映射，最后插入 item 和关联

    # 收集所有需要插入的文件
    file_specs: list[tuple[str, str, str, str, int, str]] = []  # (object_name, url, name, ext, size_bytes, md5)

    def add_file(local_path: Path, subdir: str) -> str:
        """登记一个文件，返回其 object_name"""
        object_name = build_object_name(dataset_path, subdir, local_path.name)
        try:
            size_bytes = local_path.stat().st_size
        except OSError:
            size_bytes = 0
        file_specs.append((
            object_name,
            build_url(object_name),
            local_path.name,
            local_path.suffix.lstrip(".").lower(),
            size_bytes,
            fake_md5(object_name),
        ))
        return object_name

    # 为每个分组构造数据项
    # 分组策略：每个 leading_number 对应一个 sys_dataset_item
    # 如果只有 hazy 没有 clear（如 RTTS），每个 hazy 文件单独成项
    items_to_create: list[dict] = []  # [{name, clear_path, hazy_paths, trans_paths}]

    for key in sorted(all_keys):
        clear_path = clear_map.get(key)
        hazy_paths = hazy_map.get(key, [])
        trans_paths = trans_map.get(key, [])

        if clear_path is None and hazy_paths:
            # 无清晰图：每个 hazy 单独成项（如真实有雾数据集）
            for hp in hazy_paths:
                items_to_create.append({
                    "name": hp.stem,
                    "clear": None,
                    "hazy": [hp],
                    "trans": [],
                })
        else:
            # 有清晰图：合并为一个数据项
            items_to_create.append({
                "name": key,
                "clear": clear_path,
                "hazy": hazy_paths,
                "trans": trans_paths,
            })

    # 登记所有文件
    for item in items_to_create:
        if item["clear"]:
            add_file(item["clear"], clear_subdir)
        for hp in item["hazy"]:
            add_file(hp, hazy_subdir)
        for tp in item["trans"]:
            add_file(tp, trans_subdir)

    if not file_specs:
        print(f"  [SKIP] 无可插入的文件")
        return {"files": 0, "items": 0, "skipped": True}

    # 批量插入 sys_file
    file_columns = ["object_name", "url", "name", "type", "size_bytes", "size", "path", "md5", "create_time"]
    for i in range(0, len(file_specs), BATCH_SIZE):
        batch = file_specs[i:i + BATCH_SIZE]
        rows = [
            (spec[0], spec[1], spec[2], spec[3], spec[4], format_size(spec[4]), "", spec[5], now)
            for spec in batch
        ]
        insert_batch(cursor, "sys_file", file_columns, rows)
    file_count = len(file_specs)
    print(f"  已插入 sys_file: {file_count} 条")

    # 查询 object_name → id 映射
    # 对于大数据集（如 RESIDE-OUT/train 60w 文件），分批查询以避免内存问题
    object_name_to_id: dict[str, int] = {}
    # 仅查询本次插入的 object_name
    all_object_names = [spec[0] for spec in file_specs]
    for i in range(0, len(all_object_names), BATCH_SIZE):
        batch = all_object_names[i:i + BATCH_SIZE]
        placeholders = ",".join(["%s"] * len(batch))
        cursor.execute(
            f"SELECT id, object_name FROM sys_file WHERE object_name IN ({placeholders})",
            batch,
        )
        for row in cursor.fetchall():
            object_name_to_id[row[1]] = row[0]

    # 重建 file_specs 索引：object_name → spec
    spec_map = {spec[0]: spec for spec in file_specs}

    # 插入 sys_dataset_item 和 sys_item_file
    item_columns = ["dataset_id", "name", "create_time", "update_time"]
    item_file_columns = ["item_id", "file_id", "type", "haze_level", "create_time", "update_time"]

    pending_item_rows: list[tuple] = []
    pending_item_file_rows: list[tuple] = []

    for item in items_to_create:
        pending_item_rows.append((dataset_id, item["name"], now, now))

    # 批量插入 items
    for i in range(0, len(pending_item_rows), BATCH_SIZE):
        insert_batch(cursor, "sys_dataset_item", item_columns, pending_item_rows[i:i + BATCH_SIZE])

    # 查询刚插入的 item id（按 dataset_id + name 匹配）
    cursor.execute(
        "SELECT id, name FROM sys_dataset_item WHERE dataset_id = %s",
        (dataset_id,),
    )
    item_name_to_id = {row[1]: row[0] for row in cursor.fetchall()}

    # 构造 item_file 关联
    for item in items_to_create:
        item_id = item_name_to_id.get(item["name"])
        if item_id is None:
            continue

        # 清晰图
        if item["clear"]:
            object_name = build_object_name(dataset_path, clear_subdir, item["clear"].name)
            file_id = object_name_to_id.get(object_name)
            if file_id:
                pending_item_file_rows.append((item_id, file_id, "clear", "", now, now))

        # 有雾图
        for hp in item["hazy"]:
            object_name = build_object_name(dataset_path, hazy_subdir, hp.name)
            file_id = object_name_to_id.get(object_name)
            if file_id:
                haze_level = parse_haze_level(hp.name)
                pending_item_file_rows.append((item_id, file_id, "hazy", haze_level, now, now))

        # 透射率图
        for tp in item["trans"]:
            object_name = build_object_name(dataset_path, trans_subdir, tp.name)
            file_id = object_name_to_id.get(object_name)
            if file_id:
                pending_item_file_rows.append((item_id, file_id, "trans", "", now, now))

    # 批量插入 item_file
    for i in range(0, len(pending_item_file_rows), BATCH_SIZE):
        insert_batch(cursor, "sys_item_file", item_file_columns, pending_item_file_rows[i:i + BATCH_SIZE])

    item_count = len(pending_item_rows)
    print(f"  已插入 sys_dataset_item: {item_count} 条, sys_item_file: {len(pending_item_file_rows)} 条")

    return {
        "files": file_count,
        "items": item_count,
        "item_files": len(pending_item_file_rows),
        "skipped": False,
    }


def main():
    password = load_password()
    conn = get_connection(password)

    total_files = 0
    total_items = 0
    total_item_files = 0
    start_time = time.time()

    try:
        with conn.cursor() as cursor:
            truncate_tables(cursor)
            conn.commit()

            print(f"\n开始处理 {len(LEAF_DATASETS)} 个数据集...\n")

            for idx, (dataset_id, dataset_path, clear_sub, hazy_sub, trans_sub) in enumerate(LEAF_DATASETS, 1):
                print(f"[{idx}/{len(LEAF_DATASETS)}] dataset_id={dataset_id} path={dataset_path}")
                stats = process_dataset(
                    cursor, dataset_id, dataset_path,
                    clear_sub, hazy_sub, trans_sub,
                )
                conn.commit()

                if not stats.get("skipped"):
                    total_files += stats["files"]
                    total_items += stats["items"]
                    total_item_files += stats["item_files"]

                elapsed = time.time() - start_time
                print(f"  累计: files={total_files}, items={total_items}, item_files={total_item_files}, 耗时={elapsed:.1f}s\n")

        print("=" * 60)
        print(f"刷新完成！")
        print(f"  总文件数: {total_files}")
        print(f"  总数据项: {total_items}")
        print(f"  总关联记录: {total_item_files}")
        print(f"  总耗时: {time.time() - start_time:.1f}s")
        print(f"\nMinIO URL 前缀: {MINIO_URL_BASE}")
        print(f"  示例: {MINIO_URL_BASE}/Dense-Haze/clean/01_GT.png")

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] {e}", file=sys.stderr)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
