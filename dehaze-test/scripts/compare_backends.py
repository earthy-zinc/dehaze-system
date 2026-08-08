"""三端 API 响应对比测试。

用法：
    python scripts/compare_backends.py /api/v1/auth/captcha
    python scripts/compare_backends.py /api/v1/messages/unread-count --method GET --auth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import config, api, auth


def main() -> None:
    parser = argparse.ArgumentParser(description="三端 API 响应对比")
    parser.add_argument("path", help="API 路径，如 /api/v1/auth/captcha")
    parser.add_argument("--method", "-m", default="GET", choices=["GET", "POST", "PUT", "PATCH", "DELETE"])
    parser.add_argument("--auth", action="store_true", help="是否需要登录（默认不需要）")
    parser.add_argument("--user", "-u", default="admin")
    parser.add_argument("--body", help="POST/PUT 请求体 JSON")
    args = parser.parse_args()

    body = json.loads(args.body) if args.body else None
    backends = list(config.BACKENDS.keys())

    print(f"{'backend':<10} {'status':<8} {'code':<6} {'traceId':<36} {'data_preview'}")
    print("-" * 120)

    for b in backends:
        if args.auth:
            auth.login(username=args.user, backend=b)
        try:
            resp = api.request(args.method, args.path, backend=b, json=body if body else None)
            data_preview = json.dumps(resp.get("data"), ensure_ascii=False)[:50]
            print(f"{b:<10} {'OK':<8} {resp.get('code'):<6} {resp.get('traceId', '-'):<36} {data_preview}")
        except Exception as e:
            print(f"{b:<10} {'ERR':<8} {'-':<6} {'-':<36} {str(e)[:50]}")


if __name__ == "__main__":
    main()
