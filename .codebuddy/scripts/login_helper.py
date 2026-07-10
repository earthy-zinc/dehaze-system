#!/usr/bin/env python3
"""
DehazeSystem 三端登录辅助脚本

用法:
    python login_helper.py [backend]

backend 可选: go(默认) / python / java / all

输出: 各端 accessToken（可直接用于后续 curl 测试）

环境依赖:
    - Docker Redis 容器名为 "redis"，密码 12345678
    - 三端服务: Java 8989 / Go 8999 / Python 8014
    - 统一账号: admin / 123456

注意:
    - Go/Python captcha 存 Redis db3，key 格式分别为
        Go:     captcha_code:{key}
        Python: captcha:{key}
    - Java captcha 存 Redis db0，key 格式 captcha_code:{key}，
      值经 Jackson 序列化带外层双引号，需去除
"""
import json
import subprocess
import sys
import urllib.request

# 后端配置
BACKENDS = {
    "go": {
        "base": "http://127.0.0.1:8999",
        "captcha_db": 3,
        "captcha_key_prefix": "captcha_code:",
        "strip_quotes": False,
    },
    "python": {
        "base": "http://127.0.0.1:8014",
        "captcha_db": 3,
        "captcha_key_prefix": "captcha:",
        "strip_quotes": False,
    },
    "java": {
        "base": "http://127.0.0.1:8989",
        "captcha_db": 0,
        "captcha_key_prefix": "captcha_code:",
        "strip_quotes": True,
    },
}

USERNAME = "admin"
PASSWORD = "123456"
REDIS_CONTAINER = "redis"
REDIS_PASSWORD = "12345678"


def http_post_json(url, payload):
    """发送 JSON POST 请求"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url):
    """发送 GET 请求"""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_captcha_code(captcha_key, db, prefix, strip_quotes):
    """从 Redis 读取验证码"""
    redis_key = f"{prefix}{captcha_key}"
    cmd = [
        "docker", "exec", "-i", REDIS_CONTAINER,
        "redis-cli", "-a", REDIS_PASSWORD, "-n", str(db),
        "get", redis_key,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    code = result.stdout.strip()
    if strip_quotes and code:
        # Jackson 序列化带外层双引号，去掉
        if code.startswith('"') and code.endswith('"'):
            code = code[1:-1]
    return code


def login(backend_name):
    """登录指定后端，返回 accessToken"""
    cfg = BACKENDS[backend_name]
    base = cfg["base"]

    # 1. 获取验证码
    captcha_resp = http_get_json(f"{base}/api/v1/auth/captcha")
    if captcha_resp.get("code") != "00000":
        print(f"[{backend_name}] 获取验证码失败: {captcha_resp}", file=sys.stderr)
        return None

    captcha_key = captcha_resp["data"]["captchaKey"]

    # 2. 从 Redis 读取验证码
    captcha_code = get_captcha_code(
        captcha_key,
        cfg["captcha_db"],
        cfg["captcha_key_prefix"],
        cfg["strip_quotes"],
    )
    if not captcha_code:
        print(f"[{backend_name}] 验证码已过期或不存在: {captcha_key}", file=sys.stderr)
        return None

    # 3. 登录
    login_resp = http_post_json(
        f"{base}/api/v1/auth/login",
        {
            "username": USERNAME,
            "password": PASSWORD,
            "captchaKey": captcha_key,
            "captchaCode": captcha_code,
        },
    )
    if login_resp.get("code") != "00000":
        print(f"[{backend_name}] 登录失败: {login_resp}", file=sys.stderr)
        return None

    return login_resp["data"]["accessToken"]


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "go"

    if target == "all":
        tokens = {}
        for name in BACKENDS:
            token = login(name)
            if token:
                tokens[name] = token
                print(f"{name}: {token}")
            else:
                print(f"{name}: FAILED", file=sys.stderr)
        # 也输出 JSON 便于脚本消费
        print("---JSON---")
        print(json.dumps(tokens))
    else:
        if target not in BACKENDS:
            print(f"未知后端: {target}，可选: {list(BACKENDS.keys())}", file=sys.stderr)
            sys.exit(1)
        token = login(target)
        if token:
            print(token)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
