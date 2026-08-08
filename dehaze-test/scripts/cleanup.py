"""清理限流 / 验证码 / session / 业务缓存。

用法：
    python scripts/cleanup.py [--target rate_limit|captcha|session|business_cache|all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import cleanup


def main() -> None:
    parser = argparse.ArgumentParser(description="清理 Redis 缓存")
    parser.add_argument(
        "--target",
        "-t",
        default="all",
        choices=["rate_limit", "captcha", "session", "business_cache", "all"],
    )
    args = parser.parse_args()

    if args.target == "all":
        result = cleanup.clear_all()
        for k, v in result.items():
            print(f"  {k}: 删除 {v} 个 key")
    else:
        fn = {
            "rate_limit": cleanup.clear_login_rate_limit,
            "captcha": cleanup.clear_captcha,
            "session": cleanup.clear_sessions,
            "business_cache": cleanup.clear_business_cache,
        }[args.target]
        n = fn()
        print(f"  {args.target}: 删除 {n} 个 key")


if __name__ == "__main__":
    main()
