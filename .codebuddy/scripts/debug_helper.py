#!/usr/bin/env python3
"""
DehazeSystem 三端 API 调试辅助脚本

封装调试中常用的重复操作，避免每次手写大段命令。

用法:
    python debug_helper.py <command> [args...]

命令列表:
    restart <go|python|all>          重启后端服务（自动编译 Go）
    compare <path> [method] [body]   三端对比同一 API（自动登录获取 token）
    curl <backend> <path> [method] [body]  单端请求
    build <go>                       编译指定后端
    kill <port>                      杀掉占用指定端口的进程
    status                           查看三端服务运行状态
    logs <python|go>                 查看服务日志（最近 30 行）
    db <sql>                         执行 MySQL 查询
    redis <get|keys> <key> [db]      Redis 操作

示例:
    python debug_helper.py restart go
    python debug_helper.py compare /api/v1/users/page
    python debug_helper.py compare /api/v1/users/2/form
    python debug_helper.py compare /api/v1/users/2 PATCH '{"status":0}'
    python debug_helper.py curl go /api/v1/users/page
    python debug_helper.py db "SELECT id,username FROM sys_user WHERE deleted=0"
    python debug_helper.py status

环境:
    - Go: 8990 (dehaze-go/)
    - Python: 8991 (dehaze-python/)
    - Java: 8989 (dehaze-java/, devtools 热重载)
    - MySQL: 容器 "mysql", 密码 12345678, 库 dehaze
    - Redis: 容器 "redis", 密码 12345678
    - 账号: admin / 123456
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGIN_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "login_helper.py")
PYTHON_BIN = r"C:\Python313\python.exe"
PY_VENV = os.path.join(WORKSPACE, "dehaze-python", ".venv", "Scripts", "python.exe")

BACKENDS = {
    "java":   {"base": "http://127.0.0.1:8989", "port": 8989},
    "go":     {"base": "http://127.0.0.1:8990", "port": 8990},
    "python": {"base": "http://127.0.0.1:8991", "port": 8991},
}

MYSQL_CONTAINER = "mysql"
MYSQL_PASSWORD = "12345678"
MYSQL_DB = "dehaze"
REDIS_CONTAINER = "redis"
REDIS_PASSWORD = "12345678"

# 日志文件路径
LOG_FILES = {
    "go": os.path.join(WORKSPACE, "dehaze-go", "go_server.log"),
    "python": os.path.join(WORKSPACE, "dehaze-python", "py_server.log"),
}


# ── 通用工具 ──────────────────────────────────────

def run(cmd, cwd=None, shell=False, check=False):
    """执行命令，返回 (returncode, stdout, stderr)"""
    result = subprocess.run(
        cmd, cwd=cwd, shell=shell,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if check and result.returncode != 0:
        print(f"命令失败: {' '.join(cmd) if isinstance(cmd, list) else cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.returncode, result.stdout, result.stderr


def http_request(url, method="GET", token=None, body=None):
    """发送 HTTP 请求，返回解析后的 JSON"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body:
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"error": f"HTTP {e.code}", "msg": str(e)}
    except Exception as e:
        return {"error": str(e)}


def get_token(backend):
    """通过 login_helper 获取 token"""
    code, stdout, stderr = run([PYTHON_BIN, LOGIN_HELPER, backend])
    token = stdout.strip()
    if not token or token.startswith("["):
        return None
    return token


def kill_port(port):
    """杀掉占用指定端口的进程"""
    # 查找 PID
    code, stdout, _ = run(["netstat", "-ano"], shell=True)
    pids = set()
    for line in stdout.splitlines():
        line = line.strip()
        if f":{port} " in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                pids.add(parts[-1])
    if not pids:
        return False
    for pid in pids:
        run(["powershell", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"])
    time.sleep(2)
    return True


def is_port_listening(port):
    """检查端口是否在监听"""
    code, stdout, _ = run(["netstat", "-ano"], shell=True)
    for line in stdout.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            return True
    return False


# ── 命令实现 ──────────────────────────────────────

def cmd_restart(args):
    """重启后端服务"""
    target = args[0] if args else "all"

    if target in ("go", "all"):
        print("[Go] 编译中...", end=" ", flush=True)
        go_dir = os.path.join(WORKSPACE, "dehaze-go")
        code, stdout, stderr = run(
            "go build -o bin/dehaze-go.exe ./cmd/main.go",
            cwd=go_dir, shell=True, check=False
        )
        if code != 0:
            print("编译失败!")
            print(stderr[:500])
            return
        print("编译成功")

        print("[Go] 重启中...", end=" ")
        kill_port(8990)
        log_file = LOG_FILES["go"]
        with open(log_file, "w") as f:
            pass  # 清空日志
        subprocess.Popen(
            [os.path.join(go_dir, "bin", "dehaze-go.exe")],
            cwd=go_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        time.sleep(3)
        print("已启动" if is_port_listening(8990) else "启动失败!")

    if target in ("python", "all"):
        print("[Python] 重启中...", end=" ")
        kill_port(8991)
        py_dir = os.path.join(WORKSPACE, "dehaze-python")
        log_file = LOG_FILES["python"]
        with open(log_file, "w") as f:
            pass
        subprocess.Popen(
            [PY_VENV, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8991"],
            cwd=py_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        time.sleep(5)
        print("已启动" if is_port_listening(8991) else "启动失败!")

    if target == "all":
        print("[Java] devtools 热重载，无需手动重启（XML 改动需手动复制到 target/classes/）")


def cmd_compare(args):
    """三端对比同一 API"""
    if not args:
        print("用法: compare <path> [method] [body]")
        return

    path = args[0]
    # 修复 Git Bash MSYS 路径转换（/api/v1/xxx → C:/Program Files/Git/api/v1/xxx）
    git_prefix = "C:/Program Files/Git"
    if path.startswith(git_prefix):
        path = path[len(git_prefix):]
    # 也处理反斜杠变体
    if "/api/v1" not in path and "api/v1" in path:
        # 提取 api/v1 之后的部分
        idx = path.find("api/v1")
        path = "/" + path[idx:]

    method = args[1].upper() if len(args) > 1 else "GET"
    body = args[2] if len(args) > 2 else None

    # 确保 path 以 /api/v1 开头
    if not path.startswith("/api/"):
        path = "/api/v1" + (path if path.startswith("/") else "/" + path)

    results = {}
    for name in ["java", "go", "python"]:
        token = get_token(name)
        if not token:
            print(f"[{name}] 登录失败，跳过")
            results[name] = {"error": "login failed"}
            continue
        url = BACKENDS[name]["base"] + path
        result = http_request(url, method=method, token=token, body=body)
        results[name] = result

    # 对比输出
    for name in ["java", "go", "python"]:
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
            # 打印 keys 和值（截断长值）
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
    codes = [results[n].get("code") for n in ["java", "go", "python"] if "error" not in results[n]]
    msgs = [results[n].get("msg") for n in ["java", "go", "python"] if "error" not in results[n]]

    code_consistent = len(set(codes)) <= 1
    msg_consistent = len(set(msgs)) <= 1

    print(f"  code 一致: {'✅' if code_consistent else '❌'} {codes}")
    print(f"  msg  一致: {'✅' if msg_consistent else '❌'} {msgs}")

    # 对比 data keys
    data_keys = []
    for name in ["java", "go", "python"]:
        if "error" not in results[name] and isinstance(results[name].get("data"), dict):
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
    path = args[1]
    method = args[2].upper() if len(args) > 2 else "GET"
    body = args[3] if len(args) > 3 else None

    if backend not in BACKENDS:
        print(f"未知后端: {backend}，可选: {list(BACKENDS.keys())}")
        return
    if not path.startswith("/api/"):
        path = "/api/v1" + (path if path.startswith("/") else "/" + path)

    token = get_token(backend)
    if not token:
        print(f"[{backend}] 登录失败")
        return
    url = BACKENDS[backend]["base"] + path
    result = http_request(url, method=method, token=token, body=body)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_build(args):
    """编译后端"""
    target = args[0] if args else "go"
    if target == "go":
        go_dir = os.path.join(WORKSPACE, "dehaze-go")
        print("[Go] 编译中...")
        code, _, stderr = run(
            "go build -o bin/dehaze-go.exe ./cmd/main.go",
            cwd=go_dir, shell=True, check=False
        )
        if code == 0:
            print("[Go] 编译成功 ✅")
        else:
            print("[Go] 编译失败 ❌")
            print(stderr[:1000])
    else:
        print(f"暂不支持编译 {target}")


def cmd_kill(args):
    """杀掉端口进程"""
    if not args:
        print("用法: kill <port>")
        return
    port = int(args[0])
    if kill_port(port):
        print(f"端口 {port} 的进程已终止")
    else:
        print(f"端口 {port} 没有监听的进程")


def cmd_status(args):
    """查看三端服务状态"""
    print("=== 服务状态 ===")
    for name, cfg in BACKENDS.items():
        port = cfg["port"]
        listening = is_port_listening(port)
        status = "运行中 ✅" if listening else "未运行 ❌"
        print(f"  {name:8s} :{port}  {status}")


def cmd_logs(args):
    """查看服务日志"""
    target = args[0] if args else "python"
    log_file = LOG_FILES.get(target)
    if not log_file or not os.path.exists(log_file):
        print(f"无 {target} 日志文件")
        return
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # 输出最后 30 行
    for line in lines[-30:]:
        print(line, end="")


def cmd_db(args):
    """执行 MySQL 查询"""
    if not args:
        print("用法: db <sql>")
        return
    sql = " ".join(args)
    code, stdout, stderr = run([
        "docker", "exec", "-i", MYSQL_CONTAINER,
        "mysql", "-uroot", f"-p{MYSQL_PASSWORD}", MYSQL_DB,
        "-e", sql,
    ], check=False)
    # 过滤密码警告
    output = stdout
    if "Warning" in output and "password" in output.lower():
        lines = output.splitlines()
        output = "\n".join(l for l in lines if "Warning" not in l or "password" not in l.lower())
    print(output)
    if stderr and code != 0:
        print(f"错误: {stderr}", file=sys.stderr)


def cmd_redis(args):
    """Redis 操作"""
    if not args:
        print("用法: redis <get|keys> <key> [db]")
        return
    op = args[0]
    key = args[1] if len(args) > 1 else "*"
    db = args[2] if len(args) > 2 else "3"

    if op == "get":
        code, stdout, _ = run([
            "docker", "exec", "-i", REDIS_CONTAINER,
            "redis-cli", "-a", REDIS_PASSWORD, "-n", db, "get", key,
        ], check=False)
        print(stdout.strip())
    elif op == "keys":
        code, stdout, _ = run([
            "docker", "exec", "-i", REDIS_CONTAINER,
            "redis-cli", "-a", REDIS_PASSWORD, "-n", db, "keys", key,
        ], check=False)
        print(stdout.strip())
    else:
        print(f"未知操作: {op}，支持: get, keys")


# ── 入口 ──────────────────────────────────────────

COMMANDS = {
    "restart": cmd_restart,
    "compare": cmd_compare,
    "curl": cmd_curl,
    "build": cmd_build,
    "kill": cmd_kill,
    "status": cmd_status,
    "logs": cmd_logs,
    "db": cmd_db,
    "redis": cmd_redis,
}

USAGE = """
DehazeSystem 调试辅助脚本

用法: python debug_helper.py <command> [args...]

命令:
    restart <go|python|all>          重启后端服务（自动编译 Go）
    compare <path> [method] [body]   三端对比同一 API
    curl <backend> <path> [method] [body]  单端请求
    build <go>                       编译指定后端
    kill <port>                      杀掉占用端口的进程
    status                           查看三端服务运行状态
    logs <python|go>                 查看服务日志（最后 30 行）
    db <sql>                         执行 MySQL 查询
    redis <get|keys> <key> [db]      Redis 操作

示例:
    python debug_helper.py status
    python debug_helper.py restart go
    python debug_helper.py compare /api/v1/users/page
    python debug_helper.py compare /api/v1/users/2/form
    python debug_helper.py curl go /api/v1/users/page
    python debug_helper.py db "SELECT id,username FROM sys_user WHERE deleted=0"
    python debug_helper.py logs python
"""


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
