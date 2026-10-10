"""环境重置：把 .env 指向的命名空间清空并重建为种子态。

用途：pre（或 dev）被 UI 冒烟 / SDK 契约集测污染后，回到"种子数据 + 干净缓存"的可演示状态。

治理（与 dehaze-test 的危险操作白名单化一致）：
- 破坏性执行必须同时给 `--target <库名>` 与 `--yes`，且 `--target` 必须等于根 .env 的
  `MYSQL_DATABASE`——防呆，避免在 A 环境误重置 B 环境的库；
- 目标命名空间（MySQL 库 / Redis db / MinIO bucket / MongoDB 库 / ES 索引前缀）全部从根
  `.env` 派生，不写死；
- 重置 MySQL 前必须停三端（连接池在库被 DROP 后失效），故默认编排"停服 → 重建 → 起服"。

用法（在目标环境的机器上执行，用 dehaze-python 的 venv）：
    python scripts/reset_env.py --target dehaze_pre --yes                 # 全量重置
    python scripts/reset_env.py --target dehaze_pre --yes --dry-run       # 只打印将执行的动作
    python scripts/reset_env.py --target dehaze_pre --yes --skip-services # 服务未运行时（仅数据）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config  # noqa: E402

SERVICES = "java,go,python,mcp"


def _run_services(action: str) -> None:
    """经 scripts/run.py 停/起后端；调用前清掉 PYTHONPATH（IDE 注入的 sitecustomize 钩子会让
    run.py 的进程管理崩溃，两处环境都有这个前提）。"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cmd = [sys.executable, str(config.PROJECT_ROOT / "scripts" / "run.py"), action, SERVICES]
    subprocess.run(cmd, check=True, env=env)


def _reset_redis(dry_run: bool) -> None:
    print(f"\n=== Redis db{config.REDIS_DATABASE} 清空 ===")
    if dry_run:
        print(f"  [dry-run] 将 FLUSHDB db{config.REDIS_DATABASE} @ {config.REDIS_HOST}")
        return
    import redis

    client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        db=config.REDIS_DATABASE,
    )
    before = client.dbsize()
    client.flushdb()
    print(f"  ✓ 清空 {before} 个 key，剩余 {client.dbsize()}")


def _reset_object_storage(dry_run: bool) -> None:
    print(f"\n=== MinIO bucket {config.MINIO_BUCKET} 清空 ===")
    if dry_run:
        print(f"  [dry-run] 将清空 {config.MINIO_HOST}:{config.MINIO_PORT}/{config.MINIO_BUCKET}")
        return
    from minio import Minio

    client = Minio(
        f"{config.MINIO_HOST}:{config.MINIO_PORT}",
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=False,
    )
    removed = 0
    for obj in client.list_objects(config.MINIO_BUCKET, recursive=True):
        client.remove_object(config.MINIO_BUCKET, obj.object_name)
        removed += 1
    print(f"  ✓ 删除 {removed} 个对象（bucket 保留）")


def _reset_mongodb(dry_run: bool) -> None:
    print(f"\n=== MongoDB {config.MONGODB_DATABASE} 删除 ===")
    if dry_run:
        print(f"  [dry-run] 将 dropDatabase {config.MONGODB_DATABASE}")
        return
    from pymongo import MongoClient

    client = MongoClient(
        host=config.MONGODB_HOST,
        port=config.MONGODB_PORT,
        username=config.MONGODB_USERNAME,
        password=config.MONGODB_PASSWORD,
        authSource="admin",
    )
    client.drop_database(config.MONGODB_DATABASE)
    print("  ✓ 已删除")


def _reset_elasticsearch(dry_run: bool) -> None:
    pattern = f"{config.ES_INDEX_PREFIX}*"
    print(f"\n=== Elasticsearch 索引 {pattern} 删除 ===")
    if dry_run:
        print(f"  [dry-run] 将删除 {pattern} @ {config.ES_URL}")
        return
    from elasticsearch import Elasticsearch

    client = Elasticsearch(config.ES_URL, basic_auth=(config.ES_USERNAME, config.ES_PASSWORD))
    client.indices.delete(index=pattern, ignore_unavailable=True)
    print("  ✓ 已删除")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 .env 命名空间重置环境（MySQL + Redis + MinIO + MongoDB + ES）")
    parser.add_argument("--target", required=True, help="目标库名，必须等于根 .env 的 MYSQL_DATABASE")
    parser.add_argument("--yes", action="store_true", help="确认执行破坏性操作（缺省仅打印计划）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的动作，不做任何修改")
    parser.add_argument("--skip-services", action="store_true", help="跳过停/起三端（服务未运行时用）")
    parser.add_argument("--skip-mysql", action="store_true", help="跳过 MySQL 重建")
    parser.add_argument("--skip-redis", action="store_true", help="跳过 Redis 清空")
    parser.add_argument("--skip-object", action="store_true", help="跳过 MinIO 清空")
    parser.add_argument("--skip-mongo", action="store_true", help="跳过 MongoDB 删除")
    parser.add_argument("--skip-es", action="store_true", help="跳过 Elasticsearch 索引删除")
    args = parser.parse_args()

    if args.target != config.MYSQL_DATABASE:
        raise SystemExit(
            f"--target 与 .env 不一致：--target={args.target}，MYSQL_DATABASE={config.MYSQL_DATABASE}\n"
            "（防呆：确认 .env 指向的就是要重置的环境，避免误清另一个环境的库）"
        )
    dry_run = args.dry_run or not args.yes
    if not args.yes:
        print("未给 --yes，按 dry-run 仅打印计划\n")

    print("=== 目标命名空间（派生自根 .env）===")
    print(f"  MySQL      : {config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}")
    print(f"  Redis      : {config.REDIS_HOST}:{config.REDIS_PORT} db{config.REDIS_DATABASE}")
    print(f"  MinIO      : {config.MINIO_HOST}:{config.MINIO_PORT}/{config.MINIO_BUCKET}")
    print(f"  MongoDB    : {config.MONGODB_HOST}:{config.MONGODB_PORT}/{config.MONGODB_DATABASE}")
    print(f"  Elasticsearch: {config.ES_URL} 前缀 {config.ES_INDEX_PREFIX}")

    if args.skip_services:
        print("\n[跳过停/起三端]")
    else:
        print(f"\n=== 停三端（{SERVICES}）===")
        if dry_run:
            print("  [dry-run] 将执行 run.py stop " + SERVICES)
        else:
            _run_services("stop")

    if not args.skip_mysql:
        from rebuild_mysql import rebuild_mysql

        if dry_run:
            print(f"\n=== MySQL {config.MYSQL_DATABASE} 重建 ===\n  [dry-run] 将 DROP/CREATE 并导入 schema + data")
        else:
            rebuild_mysql([config.MYSQL_DATABASE])
    if not args.skip_redis:
        _reset_redis(dry_run)
    if not args.skip_object:
        _reset_object_storage(dry_run)
    if not args.skip_mongo:
        _reset_mongodb(dry_run)
    if not args.skip_es:
        _reset_elasticsearch(dry_run)

    if args.skip_services:
        print("\n[跳过停/起三端]")
    else:
        print(f"\n=== 起三端（{SERVICES}）===")
        if dry_run:
            print("  [dry-run] 将执行 run.py run " + SERVICES)
        else:
            _run_services("run")

    print("\n=== 重置完成 ===" if not dry_run else "\n=== dry-run 结束（未做任何修改）===")


if __name__ == "__main__":
    main()
