#!/usr/bin/env python3
"""数据集初始化脚本

为磁盘上已存在的数据集文件批量创建 DB 记录（sys_dataset_item / sys_file / sys_item_file），
源文件由 nginx-dataset 直服，不上传 MinIO。

依赖：pymysql、Pillow（dehaze-python/.venv 已安装）。
运行：E:\\DehazeSystem\\dehaze-python\\.venv\\Scripts\\python.exe scripts/init_dataset.py --help

配对规则（对齐需求规格 2.8.3）：按文件名前导数字分组，同组 clear/hazy/trans 归为一个数据项。
haze_level 自动解析规则：
  - xxx_hazy_{light|medium|heavy}.jpg  → light/medium/heavy
  - {id}_{idx}_{beta}.png              → beta={beta}   （如 1000_1_0.74905.png → beta=0.74905）
  - 其他                                → NULL
OTS / Haze4K 的 A+β 双参数格式需 --haze-format 显式指定。
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import pymysql
from PIL import Image

CLEAN_DIR_FLAGS = ("clean", "clear", "gt")
HAZY_DIR_FLAGS = ("haze", "hazy")
TRANS_DIR_FLAGS = ("trans", "transmission")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

MANUAL_HAZE_PATTERN = re.compile(r"(light|medium|heavy)", re.IGNORECASE)
# {id}_{idx}_{beta}：三段，首段数字，末段浮点
BETA_PATTERN = re.compile(r"^\d+_\d+_([0-9.]+)$")
# OTS：{id}_{A}_{beta}
OTS_PATTERN = re.compile(r"^\d+_([0-9.]+)_([0-9.]+)$")
# Haze4K：{id}_{beta}_{A}（与 OTS 同形，靠 --haze-format 区分）

LEADING_DIGIT_PATTERN = re.compile(r"^(\d+)")


def parse_haze_level(stem: str, haze_format: str) -> Optional[str]:
    manual = MANUAL_HAZE_PATTERN.search(stem)
    if manual:
        return manual.group(1).lower()

    if haze_format in ("ots", "haze4k"):
        m = OTS_PATTERN.match(stem)
        if m:
            a, beta = m.group(1), m.group(2)
            return f"A={a},beta={beta}" if haze_format == "ots" else f"beta={beta},A={a}"

    m = BETA_PATTERN.match(stem)
    if m:
        return f"beta={m.group(1)}"

    return None


def group_key(filename: str) -> str:
    """按文件名前导数字分组，无前导数字时按 stem 分组。"""
    stem = Path(filename).stem
    m = LEADING_DIGIT_PATTERN.match(stem)
    return m.group(1) if m else stem


def scan_dataset_dir(dataset_dir: Path) -> dict:
    """扫描数据集目录，返回 {group_key: {type: [file_path, ...]}}。"""
    groups: dict[str, dict[str, list[Path]]] = {}
    if not dataset_dir.is_dir():
        return groups

    for sub in dataset_dir.iterdir():
        if not sub.is_dir():
            continue
        name_lower = sub.name.lower()
        if name_lower in CLEAN_DIR_FLAGS:
            img_type = "clear"
        elif name_lower in HAZY_DIR_FLAGS:
            img_type = "hazy"
        elif name_lower in TRANS_DIR_FLAGS:
            img_type = "trans"
        else:
            continue

        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                key = group_key(f.name)
                groups.setdefault(key, {}).setdefault(img_type, []).append(f)

    return groups


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def get_leaf_dataset_ids(cur) -> list[tuple[int, str, str]]:
    """返回 [(id, name, path), ...]，path 为 sys_dataset.path 字段。"""
    cur.execute(
        "SELECT id, name, path FROM sys_dataset "
        "WHERE deleted = 0 AND id NOT IN (SELECT DISTINCT parent_id FROM sys_dataset WHERE deleted = 0 AND parent_id IS NOT NULL) "
        "ORDER BY id"
    )
    return cur.fetchall()


def count_items(cur, dataset_id: int) -> int:
    cur.execute("SELECT COUNT(*) FROM sys_dataset_item WHERE dataset_id = %s AND deleted = 0", (dataset_id,))
    return cur.fetchone()[0]


def delete_dataset_items(cur, conn, dataset_id: int):
    """级联逻辑删除数据集的所有数据项及关联文件记录（用于 --regenerate）。"""
    cur.execute("SELECT id FROM sys_dataset_item WHERE dataset_id = %s AND deleted = 0", (dataset_id,))
    item_ids = [r[0] for r in cur.fetchall()]
    if not item_ids:
        return

    placeholders = ",".join(["%s"] * len(item_ids))

    cur.execute(f"SELECT file_id, thumbnail_file_id FROM sys_item_file WHERE item_id IN ({placeholders}) AND deleted = 0", item_ids)
    file_ids = {fid for row in cur.fetchall() for fid in row if fid}

    cur.execute(f"UPDATE sys_item_file SET deleted = 1 WHERE item_id IN ({placeholders})", item_ids)
    cur.execute(f"UPDATE sys_dataset_item SET deleted = 1 WHERE id IN ({placeholders})", item_ids)

    if file_ids:
        placeholders = ",".join(["%s"] * len(file_ids))
        cur.execute(f"UPDATE sys_file SET deleted = 1 WHERE id IN ({placeholders})", list(file_ids))

    conn.commit()


def insert_file_record(cur, conn, *, name, object_name, size_bytes, ext, url, md5, path) -> int:
    """插入 sys_file（MD5 去重），返回 file_id。"""
    cur.execute("SELECT id FROM sys_file WHERE md5 = %s AND deleted = 0", (md5,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        "INSERT INTO sys_file (name, object_name, size, size_bytes, type, url, md5, path) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (name, object_name, human_size(size_bytes), size_bytes, ext, url, md5, path),
    )
    return cur.lastrowid


def insert_item_file(cur, *, item_id, file_id, img_type, width, height, haze_level):
    cur.execute(
        "INSERT INTO sys_item_file (item_id, file_id, type, width, height, haze_level, usage_count) "
        "VALUES (%s, %s, %s, %s, %s, %s, 0)",
        (item_id, file_id, img_type, width, height, haze_level),
    )


def init_single_dataset(cur, conn, *, dataset_id, dataset_name, dataset_path_field,
                        dataset_root: Path, dataset_base_url: str, haze_format: str,
                        regenerate: bool) -> tuple[int, int]:
    """初始化单个数据集，返回 (数据项数, 文件数)。"""
    if count_items(cur, dataset_id) > 0:
        if regenerate:
            print(f"  [--regenerate] 清理数据集[{dataset_id}]已有数据项...")
            delete_dataset_items(cur, conn, dataset_id)
        else:
            print(f"  数据集[{dataset_id}] {dataset_name} 已有数据项，跳过（--regenerate 强制重建）")
            return 0, 0

    dataset_dir = dataset_root / dataset_path_field
    if not dataset_dir.is_dir():
        print(f"  数据集[{dataset_id}] {dataset_name} 目录不存在: {dataset_dir}")
        return 0, 0

    groups = scan_dataset_dir(dataset_dir)
    if not groups:
        print(f"  数据集[{dataset_id}] {dataset_name} 未发现配对图片目录（clean/hazy/trans）")
        return 0, 0

    item_count = 0
    file_count = 0
    for key in sorted(groups.keys()):
        files_by_type = groups[key]
        if not any(files_by_type.values()):
            continue

        cur.execute(
            "INSERT INTO sys_dataset_item (dataset_id, name) VALUES (%s, %s)",
            (dataset_id, key),
        )
        item_id = cur.lastrowid
        item_count += 1

        for img_type in ("clear", "hazy", "trans"):
            for fp in sorted(files_by_type.get(img_type, []), key=lambda p: p.name):
                rel = fp.relative_to(dataset_root).as_posix()
                md5 = md5_file(fp)
                haze_level = parse_haze_level(fp.stem, haze_format) if img_type == "hazy" else None
                try:
                    width, height = read_size(fp)
                except Exception as e:
                    print(f"    读取尺寸失败 {fp}: {e}")
                    width, height = None, None

                file_id = insert_file_record(
                    cur, conn,
                    name=fp.name,
                    object_name=rel,
                    size_bytes=fp.stat().st_size,
                    ext=fp.suffix.lstrip(".").lower(),
                    url=f"{dataset_base_url}/{rel}",
                    md5=md5,
                    path=rel,
                )
                insert_item_file(
                    cur,
                    item_id=item_id,
                    file_id=file_id,
                    img_type=img_type,
                    width=width,
                    height=height,
                    haze_level=haze_level,
                )
                file_count += 1

        conn.commit()

    return item_count, file_count


def main():
    parser = argparse.ArgumentParser(description="数据集初始化：为磁盘文件批量创建 DB 记录")
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", required=True)
    parser.add_argument("--db-name", default="dehaze")
    parser.add_argument("--dataset-path", required=True, help="数据集根目录（对应 file.datasetPath）")
    parser.add_argument("--dataset-base-url", required=True, help="nginx-dataset 静态服务 URL（如 http://127.0.0.1:9000/datasets）")
    parser.add_argument("--dataset-id", type=int, help="仅初始化指定数据集 ID（不指定则初始化所有叶子数据集）")
    parser.add_argument("--haze-format", choices=["auto", "ots", "haze4k"], default="auto",
                        help="haze_level 解析格式：auto(默认) / ots(A=,beta=) / haze4k(beta=,A=)")
    parser.add_argument("--regenerate", action="store_true", help="强制重建（先删后建）")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    if not dataset_root.is_dir():
        print(f"数据集根目录不存在: {dataset_root}")
        sys.exit(1)

    conn = pymysql.connect(host=args.db_host, port=args.db_port, user=args.db_user,
                           password=args.db_password, database=args.db_name, autocommit=False)
    try:
        with conn.cursor() as cur:
            if args.dataset_id:
                cur.execute("SELECT id, name, path FROM sys_dataset WHERE id = %s AND deleted = 0", (args.dataset_id,))
                datasets = cur.fetchall()
            else:
                datasets = get_leaf_dataset_ids(cur)

            if not datasets:
                print("未找到可初始化的数据集")
                return

            print(f"发现 {len(datasets)} 个数据集，开始初始化（dataset_path={dataset_root}, base_url={args.dataset_base_url}）")
            total_items = 0
            total_files = 0
            for ds_id, ds_name, ds_path in datasets:
                print(f"\n[{ds_id}] {ds_name} (path={ds_path})")
                items, files = init_single_dataset(
                    cur, conn,
                    dataset_id=ds_id,
                    dataset_name=ds_name,
                    dataset_path_field=ds_path,
                    dataset_root=dataset_root,
                    dataset_base_url=args.dataset_base_url.rstrip("/"),
                    haze_format=args.haze_format,
                    regenerate=args.regenerate,
                )
                total_items += items
                total_files += files
                print(f"  新增数据项 {items} 条，文件记录 {files} 条")

            print(f"\n初始化完成：共 {total_items} 条数据项，{total_files} 条文件记录")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
