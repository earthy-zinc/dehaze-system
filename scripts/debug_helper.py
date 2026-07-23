#!/usr/bin/env python3
import json
import subprocess
import sys
import urllib.error
import urllib.request

USAGE = """
DehazeSystem 调试辅助脚本

用法: python debug_helper.py <command> [args...]

命令:
    compare <path> [method] [body]   三端对比同一 API
    curl <backend> <path> [method] [body]  单端请求

示例:
    python debug_helper.py compare /api/v1/users/page
    python debug_helper.py compare /api/v1/users/2/form
    python debug_helper.py curl go /api/v1/users/page
"""

# 后端配置
BACKENDS = {
    "java": {
        "base": "http://127.0.0.1:8989",
        "captcha_db": 0,
        "captcha_key_prefix": "captcha_code:",
        "strip_quotes": False,
    },
    "go": {
        "base": "http://127.0.0.1:8990",
        "captcha_db": 0,
        "captcha_key_prefix": "captcha_code:",
        "strip_quotes": False,
    },
    "python": {
        "base": "http://127.0.0.1:8991",
        "captcha_db": 0,
        "captcha_key_prefix": "captcha:",
        "strip_quotes": False,
    },
}

USERNAME = "admin"
PASSWORD = "123456"
REDIS_CONTAINER = "redis"
REDIS_PASSWORD = "12345678"

BACKEND_NAMES = list(BACKENDS.keys())


# ── 通用工具 ──────────────────────────────────────

def http_request(url, method="GET", token=None, body=None):
    """发送 HTTP 请求，返回解析后的 JSON"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body:
        if isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"error": f"HTTP {e.code}", "msg": str(e)}
    except (urllib.error.URLError, OSError) as e:
        return {"error": str(e)}


def get_captcha_code(captcha_key, db, prefix, strip_quotes):
    """从 Redis 读取验证码"""
    redis_key = f"{prefix}{captcha_key}"
    cmd = [
        "docker", "exec", "-i", REDIS_CONTAINER,
        "redis-cli", "-a", REDIS_PASSWORD, "-n", str(db),
        "get", redis_key,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    code = result.stdout.strip()
    if strip_quotes and code and code.startswith('"') and code.endswith('"'):
        code = code[1:-1]
    return code


def login(backend):
    """登录指定后端，返回 accessToken"""
    cfg = BACKENDS[backend]
    base = cfg["base"]

    # 1. 获取验证码
    captcha_resp = http_request(f"{base}/api/v1/auth/captcha")
    if captcha_resp.get("code") != "00000":
        print(f"[{backend}] 获取验证码失败: {captcha_resp}", file=sys.stderr)
        return None

    captcha_data = captcha_resp["data"]
    if not isinstance(captcha_data, dict):
        print(f"[{backend}] 获取验证码失败: 响应格式异常", file=sys.stderr)
        return None
    captcha_key = captcha_data["captchaKey"]

    # 2. 从 Redis 读取验证码
    captcha_code = get_captcha_code(
        captcha_key,
        cfg["captcha_db"],
        cfg["captcha_key_prefix"],
        cfg["strip_quotes"],
    )
    if not captcha_code:
        print(f"[{backend}] 验证码已过期或不存在: {captcha_key}", file=sys.stderr)
        return None

    # 3. 登录
    login_resp = http_request(
        f"{base}/api/v1/auth/login",
        method="POST",
        body={
            "username": USERNAME,
            "password": PASSWORD,
            "captchaKey": captcha_key,
            "captchaCode": captcha_code,
        },
    )
    if login_resp.get("code") != "00000":
        print(f"[{backend}] 登录失败: {login_resp}", file=sys.stderr)
        return None

    login_data = login_resp["data"]
    if not isinstance(login_data, dict):
        print(f"[{backend}] 登录失败: 响应格式异常", file=sys.stderr)
        return None
    return login_data["accessToken"]


def get_token(backend):
    """获取后端 token"""
    token = login(backend)
    if not token:
        print(f"[{backend}] 登录失败，跳过", file=sys.stderr)
    return token


def normalize_path(path):
    """规范化 API 路径，处理 Git Bash MSYS 路径转换并确保以 /api/v1 开头"""
    git_prefix = "C:/Program Files/Git"
    if path.startswith(git_prefix):
        path = path[len(git_prefix):]
    if "/api/v1" not in path and "api/v1" in path:
        idx = path.find("api/v1")
        path = "/" + path[idx:]
    if not path.startswith("/api/"):
        path = "/api/v1" + (path if path.startswith("/") else "/" + path)
    return path


# ── 命令实现 ──────────────────────────────────────

def cmd_compare(args):
    """三端对比同一 API"""
    if not args:
        print("用法: compare <path> [method] [body]")
        return

    path = normalize_path(args[0])
    method = args[1].upper() if len(args) > 1 else "GET"
    body = args[2] if len(args) > 2 else None

    results = {}
    for name in BACKEND_NAMES:
        token = get_token(name)
        if not token:
            results[name] = {"error": "login failed"}
            continue
        url = BACKENDS[name]["base"] + path
        result = http_request(url, method=method, token=token, body=body)
        results[name] = result

    # 对比输出
    for name in BACKEND_NAMES:
        r = results[name]
        print(f"\n=== {name} ===")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        code = r.get("code", "?")
        msg = r.get("msg", "?")
        data = r.get("data")
        print(f"  code: {code} | msg: {msg}")
        if isinstance(data, dict):
            for k in sorted(data.keys()):
                v = data[k]
                if isinstance(v, str) and len(v) > 60:
                    v = v[:60] + "..."
                elif isinstance(v, list):
                    v = f"[{len(v)} items]"
                print(f"  {k}: {v}")
        elif isinstance(data, list):
            print(f"  list: [{len(data)} items]")
            if data:
                print(f"  list[0] keys: {sorted(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        else:
            print(f"  data: {data}")

    # 一致性判断
    print("\n--- 一致性 ---")
    active = [n for n in BACKEND_NAMES if "error" not in results[n]]
    codes = [results[n].get("code") for n in active]
    msgs = [results[n].get("msg") for n in active]

    code_consistent = len(set(codes)) <= 1
    msg_consistent = len(set(msgs)) <= 1

    print(f"  code 一致: {'✅' if code_consistent else '❌'} {codes}")
    print(f"  msg  一致: {'✅' if msg_consistent else '❌'} {msgs}")

    # 对比 data keys
    data_keys = []
    for name in active:
        if isinstance(results[name].get("data"), dict):
            data_keys.append((name, sorted(results[name]["data"].keys())))
    if data_keys:
        all_same = all(k == data_keys[0][1] for _, k in data_keys)
        print(f"  data keys 一致: {'✅' if all_same else '❌'}")
        if not all_same:
            for name, keys in data_keys:
                print(f"    {name}: {keys}")


def cmd_curl(args):
    """单端请求"""
    if len(args) < 2:
        print("用法: curl <backend> <path> [method] [body]")
        return
    backend = args[0]
    path = normalize_path(args[1])
    method = args[2].upper() if len(args) > 2 else "GET"
    body = args[3] if len(args) > 3 else None

    if backend not in BACKENDS:
        print(f"未知后端: {backend}，可选: {BACKEND_NAMES}")
        return

    token = get_token(backend)
    if not token:
        return
    url = BACKENDS[backend]["base"] + path
    result = http_request(url, method=method, token=token, body=body)
    print(json.dumps(result, indent=2, ensure_ascii=False))


COMMANDS = {
    "compare": cmd_compare,
    "curl": cmd_curl,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(USAGE)
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd not in COMMANDS:
        print(f"未知命令: {cmd}")
        print(USAGE)
        sys.exit(1)

    COMMANDS[cmd](args)


if __name__ == "__main__":
    main()