#!/usr/bin/env python3
"""WPX 文件初始化脚本

建立 WPX 预处理图与原始数据集图的一一对应关系，写入 sys_wpx_file 表。

配对规则（同一前导数字 + 同一数据集 + 同一类型目录）：
  WPX/O-HAZE/clean/01_GT.png      ↔  O-HAZE/clean/01_outdoor_GT.jpg
  WPX/O-HAZE/hazy/01_hazy.png     ↔  O-HAZE/hazy/01_outdoor_hazy.jpg

字段说明：
  - origin_*：原始数据集图（应由 init_dataset.py 先入库 sys_file）
  - new_*：WPX 预处理图（本脚本创建 sys_file 记录，nginx 直服，登记为 nginx-static 后端，不上传 MinIO）

前置条件：先执行 init_dataset.py，确保原始数据集图已写入 sys_file。

依赖：pymysql（dehaze-python/.venv 已安装）。
运行：E:\\DehazeSystem\\dehaze-python\\.venv\\Scripts\\python.exe scripts/init_wpx_file.py --help

存储约定：sys_file 写 object_name + storage='nginx-static'，URL 永不落库。
WPX 图在 datasets/WPX/... 下，object_name 含 datasets/ 资源前缀。
sys_wpx_file.origin_path/new_path 为路径快照、非访问 URL，仅用于记录原始/预处理图的相对位置。
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import pymysql

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
LEADING_DIGIT_PATTERN = re.compile(r"^(\d+)")
WPX_ROOT_DIR = "WPX"
CLEAN_DIR_FLAGS = ("clean", "clear", "gt")
HAZY_DIR_FLAGS = ("haze", "hazy")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def leading_digits(filename: str) -> Optional[str]:
    m = LEADING_DIGIT_PATTERN.match(Path(filename).stem)
    return m.group(1) if m else None


def index_dir_by_leading_digits(directory: Path) -> dict[str, list[Path]]:
    """将目录下文件按前导数字索引，返回 {leading_digits: [file_path, ...]}。

    无前导数字的文件被忽略。同前导数字有多张文件时保留全部，由调用方处理冲突。
    """
    index: dict[str, list[Path]] = {}
    if not directory.is_dir():
        return index
    for f in directory.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = leading_digits(f.name)
        if key is None:
            continue
        index.setdefault(key, []).append(f)
    return index


def scan_wpx_pairs(dataset_root: Path) -> list[dict]:
    """扫描 WPX 目录，返回配对列表。

    每项：{
        'dataset': 'O-HAZE',
        'img_type': 'clean',
        'leading': '01',
        'wpx_file': Path(...),
        'origin_file': Path(...),
    }
    """
    wpx_root = dataset_root / WPX_ROOT_DIR
    if not wpx_root.is_dir():
        print(f"WPX 目录不存在: {wpx_root}")
        return []

    pairs: list[dict] = []
    for dataset_dir in sorted(p for p in wpx_root.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        origin_dataset_dir = dataset_root / dataset_name
        if not origin_dataset_dir.is_dir():
            print(f"  跳过 [{dataset_name}]：原始数据集目录不存在 {origin_dataset_dir}")
            continue

        for type_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            type_name_lower = type_dir.name.lower()
            if type_name_lower in CLEAN_DIR_FLAGS:
                img_type = "clean"
            elif type_name_lower in HAZY_DIR_FLAGS:
                img_type = "hazy"
            else:
                continue

            origin_type_dir = origin_dataset_dir / type_dir.name
            if not origin_type_dir.is_dir():
                # 大小写回退：在原始数据集目录下找同名（忽略大小写）子目录
                match = next(
                    (p for p in origin_dataset_dir.iterdir()
                     if p.is_dir() and p.name.lower() == type_name_lower),
                    None,
                )
                if match is None:
                    print(f"  跳过 [{dataset_name}/{img_type}]：原始目录不存在")
                    continue
                origin_type_dir = match

            wpx_index = index_dir_by_leading_digits(type_dir)
            origin_index = index_dir_by_leading_digits(origin_type_dir)

            for key in sorted(wpx_index.keys()):
                wpx_files = wpx_index[key]
                origin_files = origin_index.get(key, [])

                if len(wpx_files) > 1:
                    print(f"  警告 [{dataset_name}/{img_type}/{key}] WPX 侧有多张文件，跳过: {wpx_files}")
                    continue
                if len(origin_files) > 1:
                    print(f"  警告 [{dataset_name}/{img_type}/{key}] 原始侧有多张文件，跳过: {origin_files}")
                    continue
                if not origin_files:
                    print(f"  警告 [{dataset_name}/{img_type}/{key}] 未找到原始配对文件，跳过")
                    continue

                pairs.append({
                    "dataset": dataset_name,
                    "img_type": img_type,
                    "leading": key,
                    "wpx_file": wpx_files[0],
                    "origin_file": origin_files[0],
                })

    return pairs


def find_file_id_by_md5(cur, md5: str) -> Optional[int]:
    cur.execute("SELECT id FROM sys_file WHERE md5 = %s AND deleted = 0", (md5,))
    row = cur.fetchone()
    return row[0] if row else None


def insert_file_record(cur, *, name, object_name, size_bytes, ext, md5) -> int:
    cur.execute(
        "INSERT INTO sys_file (name, object_name, storage, size, size_bytes, type, md5) "
        "VALUES (%s, %s, 'nginx-static', %s, %s, %s, %s)",
        (name, object_name, human_size(size_bytes), size_bytes, ext, md5),
    )
    return cur.lastrowid


def insert_wpx_file(cur, *, origin_file_id, origin_md5, origin_path,
                    new_file_id, new_md5, new_path):
    cur.execute(
        "INSERT INTO sys_wpx_file "
        "(origin_file_id, origin_md5, origin_path, new_file_id, new_md5, new_path) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (origin_file_id, origin_md5, origin_path, new_file_id, new_md5, new_path),
    )


def delete_existing_wpx_records(cur, conn, *, origin_md5_set: set, new_md5_set: set):
    """删除已存在的 sys_wpx_file 记录（按 origin_md5 或 new_md5 匹配），用于 --regenerate。"""
    if not origin_md5_set and not new_md5_set:
        return
    placeholders = ",".join(["%s"] * len(origin_md5_set))
    cur.execute(f"DELETE FROM sys_wpx_file WHERE origin_md5 IN ({placeholders})", list(origin_md5_set))
    placeholders = ",".join(["%s"] * len(new_md5_set))
    cur.execute(f"DELETE FROM sys_wpx_file WHERE new_md5 IN ({placeholders})", list(new_md5_set))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="WPX 文件初始化：建立 WPX 图与原始图的对应关系")
    parser.add_argument("--db-host", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--db-user", default="root")
    parser.add_argument("--db-password", required=True)
    parser.add_argument("--db-name", default="dehaze")
    parser.add_argument("--dataset-path", required=True, help="数据集根目录（对应 file.datasetPath）")
    parser.add_argument("--nginx-base-url", required=True,
                        help="nginx 静态服务根地址（如 http://127.0.0.1:9000），不带 /datasets 等资源子路径；WPX 图由此直服")
    parser.add_argument("--regenerate", action="store_true", help="强制重建（先删后建）")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    if not dataset_root.is_dir():
        print(f"数据集根目录不存在: {dataset_root}")
        sys.exit(1)

    base_url = args.nginx_base_url.rstrip("/")

    pairs = scan_wpx_pairs(dataset_root)
    if not pairs:
        print("未发现 WPX 配对文件")
        return

    print(f"发现 {len(pairs)} 对 WPX-原始图配对，开始处理（nginx_base_url={base_url}）...")

    conn = pymysql.connect(host=args.db_host, port=args.db_port, user=args.db_user,
                           password=args.db_password, database=args.db_name, autocommit=False)
    try:
        with conn.cursor() as cur:
            if args.regenerate:
                origin_md5_set = {md5_file(p["origin_file"]) for p in pairs}
                new_md5_set = {md5_file(p["wpx_file"]) for p in pairs}
                print(f"  [--regenerate] 清理已存在的 sys_wpx_file 记录...")
                delete_existing_wpx_records(cur, conn,
                                            origin_md5_set=origin_md5_set,
                                            new_md5_set=new_md5_set)

            created = 0
            skipped = 0
            for p in pairs:
                tag = f"[{p['dataset']}/{p['img_type']}/{p['leading']}]"
                origin_file = p["origin_file"]
                wpx_file = p["wpx_file"]

                origin_md5 = md5_file(origin_file)
                origin_file_id = find_file_id_by_md5(cur, origin_md5)
                if origin_file_id is None:
                    print(f"  {tag} 原始图未入库 sys_file (md5={origin_md5})，跳过。请先执行 init_dataset.py")
                    skipped += 1
                    continue

                new_md5 = md5_file(wpx_file)
                new_rel = wpx_file.relative_to(dataset_root).as_posix()

                new_file_id = find_file_id_by_md5(cur, new_md5)
                if new_file_id is None:
                    new_file_id = insert_file_record(
                        cur,
                        name=wpx_file.name,
                        object_name=f"datasets/{new_rel}",
                        size_bytes=wpx_file.stat().st_size,
                        ext=wpx_file.suffix.lstrip(".").lower(),
                        md5=new_md5,
                    )

                try:
                    insert_wpx_file(
                        cur,
                        origin_file_id=origin_file_id,
                        origin_md5=origin_md5,
                        origin_path=origin_file.relative_to(dataset_root).as_posix(),
                        new_file_id=new_file_id,
                        new_md5=new_md5,
                        new_path=new_rel,
                    )
                    created += 1
                    conn.commit()
                except pymysql.err.IntegrityError as e:
                    conn.rollback()
                    print(f"  {tag} 写入 sys_wpx_file 失败（可能已存在）：{e}")
                    skipped += 1

            print(f"\n完成：新建 {created} 条映射，跳过 {skipped} 条")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
