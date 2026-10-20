"""clarity-python 运维命令入口（显式执行，不在进程启动时运行）

内置模型/语音/图像供应商的配置权威源是数据库（初始化数据见 `config/sql/data/`）。
本命令只补齐**缺失**的行（供应商、服务密钥、内置模型、非 token 价格），已存在的行
原样不动——用户/管理端对 DB 的修改不会被覆盖。内置服务密钥（INTERNAL_SERVICE_KEY）
需 AES 加密入库、无法进 SQL 种子，故由本命令显式写入（幂等；密钥轮换时自动停用旧行）。

用法（在 clarity-python 目录下执行，env 由 app.config 按部署环境加载）：
    .venv/bin/python -m app.cli init-builtin

行为受部署开关约束：VOICE_LOCAL_ENABLED=false 时不初始化内置语音供应商并禁用已有行
（该部署由第三方语音引擎提供 ASR/TTS）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)


async def _init_builtin() -> None:
    """幂等初始化内置 LLM/语音/图像供应商、服务密钥、模型与价格（异常直接抛出，由调用方呈现）"""
    from app.database import get_db_session, init_db
    from app.infrastructure.llm.local.model_seeder import (
        ensure_local_infra_prices,
        ensure_local_infra_providers,
        ensure_local_models,
    )

    await init_db()
    # 分事务执行：模型先行（价格依赖模型/供应商行已存在），单项失败不吞——显式命令需要真实结果
    async with get_db_session() as db:
        await ensure_local_models(db)
    async with get_db_session() as db:
        await ensure_local_infra_providers(db)
    async with get_db_session() as db:
        await ensure_local_infra_prices(db)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="clarity-python 运维命令（幂等，可重复执行）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "init-builtin",
        help="初始化内置供应商与服务密钥、内置模型行、非 token 价格（只补缺失，不覆盖已配置行）",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "init-builtin":
        asyncio.run(_init_builtin())
        print("init-builtin 完成：缺失项已补齐，已存在的行未改动")
        return 0
    parser.error(f"未知命令: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
