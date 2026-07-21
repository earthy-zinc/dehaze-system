#!/usr/bin/env python3
"""
DehazeSystem 后端服务生命周期统一管理脚本

docker-like 命令风格，跨平台（Windows / macOS / Linux）。

用法:
    python scripts/run.py <command> [args...]

命令:
    run <service>[,service...]      启动服务
    stop <service>[,service...]     停止服务
    restart <service>[,service...]  重启服务
    ps                              查看所有服务状态
    logs <service> [lines]          查看服务日志（默认 50 行）
    kill <port>                     杀掉占用端口的进程

服务名（支持别名 / 逗号分隔 / all）:
    dehaze-go | go         端口 8990  (Go 二进制，启动前自动 go build)
    dehaze-python | python 端口 8991  (uvicorn + .venv)
    dehaze-java | java     端口 8989  (mvn spring-boot:run)

示例:
    python scripts/run.py run dehaze-go
    python scripts/run.py run dehaze-go,dehaze-java
    python scripts/run.py stop all
    python scripts/run.py restart dehaze-go
    python scripts/run.py ps
    python scripts/run.py logs dehaze-python 100
    python scripts/run.py kill 8990

薄壳调用（无需写 python 前缀）:
    Windows cmd / PowerShell:  run.cmd run dehaze-go
    Bash / Git Bash:           ./run run dehaze-go
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ============================== 环境 ==============================

WORKSPACE = Path(__file__).resolve().parent.parent
IS_WINDOWS = sys.platform == "win32"

# Python 虚拟环境解释器
_PY_BIN = "python.exe" if IS_WINDOWS else "python"
PY_VENV = str(WORKSPACE / "dehaze-python" / ".venv" /
              ("Scripts" if IS_WINDOWS else "bin") / _PY_BIN)

# Go 二进制输出路径
_GO_BIN_NAME = "dehaze-go.exe" if IS_WINDOWS else "dehaze-go"
GO_BIN = str(WORKSPACE / "dehaze-go" / "bin" / _GO_BIN_NAME)


# ============================== 服务定义 ==============================

SERVICES = {
    "dehaze-go": {
        "aliases": ["go"],
        "port": 8990,
        "cwd": str(WORKSPACE / "dehaze-go"),
        "build_cmd": ["go", "build", "-o", GO_BIN, "./cmd/main.go"],
        "start_cmd": [GO_BIN],
        "log_file": str(WORKSPACE / "dehaze-go" / "go_server.log"),
        "pid_file": str(WORKSPACE / "dehaze-go" / "go_server.pid"),
        "start_timeout": 15,
    },
    "dehaze-python": {
        "aliases": ["python"],
        "port": 8991,
        "cwd": str(WORKSPACE / "dehaze-python"),
        "build_cmd": None,
        "start_cmd": [PY_VENV, "-m", "uvicorn", "app.main:app",
                      "--host", "127.0.0.1", "--port", "8991"],
        "log_file": str(WORKSPACE / "dehaze-python" / "py_server.log"),
        "pid_file": str(WORKSPACE / "dehaze-python" / "py_server.pid"),
        "start_timeout": 20,
    },
    "dehaze-java": {
        "aliases": ["java"],
        "port": 8989,
        "cwd": str(WORKSPACE / "dehaze-java"),
        "build_cmd": None,
        "start_cmd": ["mvn", "spring-boot:run", "-DskipTests", "-Dmaven.test.skip=true"],
        "log_file": str(WORKSPACE / "dehaze-java" / "java_server.log"),
        "pid_file": str(WORKSPACE / "dehaze-java" / "java_server.pid"),
        "start_timeout": 90,  # Spring Boot 启动较慢
    },
}

# 别名 → 标准服务名
ALIAS_TO_KEY = {alias: key
                for key, cfg in SERVICES.items()
                for alias in [key] + cfg["aliases"]}


# ============================== 通用工具 ==============================

def resolve_services(arg: str) -> list[str]:
    """解析服务参数，支持逗号分隔、别名、all"""
    if arg == "all":
        return list(SERVICES.keys())
    resolved: list[str] = []
    for name in arg.split(","):
        name = name.strip()
        if not name:
            continue
        key = ALIAS_TO_KEY.get(name)
        if key is None:
            print(f"[ERROR] 未知服务名: {name}，可选: {sorted(ALIAS_TO_KEY.keys())}",
                  file=sys.stderr)
            sys.exit(1)
        if key not in resolved:
            resolved.append(key)
    if not resolved:
        print("[ERROR] 未指定任何服务", file=sys.stderr)
        sys.exit(1)
    return resolved


def is_port_listening(port: int) -> bool:
    """探测端口是否在监听"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但无权限发信号
        return True
    except OSError:
        return False


def read_pid(pid_file: str) -> int | None:
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_pid(pid_file: str, pid: int):
    Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_pid(pid_file: str):
    try:
        os.remove(pid_file)
    except FileNotFoundError:
        pass


def find_pids_on_port(port: int) -> list[int]:
    """查找占用指定端口的进程 PID（跨平台）"""
    if IS_WINDOWS:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, shell=True,
        )
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            # 形如 TCP  127.0.0.1:8990  0.0.0.0:0  LISTENING  12345
            if f":{port} " in line and "LISTENING" in line:
                parts = line.split()
                try:
                    pids.add(int(parts[-1]))
                except (ValueError, IndexError):
                    pass
        return sorted(pids)
    # POSIX: 优先 lsof，其次 ss
    for cmd in (["lsof", "-t", f"-i:{port}"],
                ["ss", "-lptn", f"sport = :{port}"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                continue
            if cmd[0] == "lsof":
                return [int(p) for p in result.stdout.split() if p.isdigit()]
            return [int(m) for m in re.findall(r"pid=(\d+)", result.stdout)]
        except FileNotFoundError:
            continue
    return []


def kill_pid(pid: int, force: bool = False):
    """终止进程"""
    if IS_WINDOWS:
        cmd = ["taskkill", "/F" if force else "", "/PID", str(pid)]
        subprocess.run([c for c in cmd if c], capture_output=True)
    else:
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def wait_pid_exit(pid: int, timeout: int = 10) -> bool:
    for _ in range(timeout):
        if not is_pid_running(pid):
            return True
        time.sleep(1)
    return False


# ============================== 命令实现 ==============================

def start_service(key: str) -> bool:
    cfg = SERVICES[key]
    port = cfg["port"]

    if is_port_listening(port):
        print(f"[{key}] 端口 {port} 已在监听，跳过启动（如需重启请先 stop）")
        return False

    # 编译（仅 Go）
    if cfg["build_cmd"]:
        print(f"[{key}] 编译中...", end=" ", flush=True)
        result = subprocess.run(
            cfg["build_cmd"], cwd=cfg["cwd"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("失败")
            print(result.stderr[:800])
            return False
        print("完成")

    # 启动
    print(f"[{key}] 启动中 (port={port})...", end=" ", flush=True)
    Path(cfg["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(cfg["log_file"], "w", encoding="utf-8")

    creation_flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
    start_new_session = not IS_WINDOWS

    proc = subprocess.Popen(
        cfg["start_cmd"],
        cwd=cfg["cwd"],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags,
        start_new_session=start_new_session,
    )
    write_pid(cfg["pid_file"], proc.pid)

    # 等待端口监听
    max_wait = cfg.get("start_timeout", 30)
    for _ in range(max_wait):
        if is_port_listening(port):
            print(f"已启动 (pid={proc.pid}, log={cfg['log_file']})")
            return True
        if proc.poll() is not None:
            print(f"进程已退出 (code={proc.returncode})，查看日志: {cfg['log_file']}")
            remove_pid(cfg["pid_file"])
            return False
        time.sleep(1)
    print(f"超时未监听端口，查看日志: {cfg['log_file']}")
    return False


def stop_service(key: str) -> bool:
    cfg = SERVICES[key]
    port = cfg["port"]

    # 1. 优先用 PID 文件
    pid = read_pid(cfg["pid_file"])
    if pid and is_pid_running(pid):
        print(f"[{key}] 停止 pid={pid}...", end=" ", flush=True)
        kill_pid(pid)
        if wait_pid_exit(pid, timeout=10):
            print("已停止")
            remove_pid(cfg["pid_file"])
            return True
        kill_pid(pid, force=True)
        if wait_pid_exit(pid, timeout=3):
            print("已强制停止")
            remove_pid(cfg["pid_file"])
            return True

    # 2. fallback: 杀端口占用进程
    pids = find_pids_on_port(port)
    if pids:
        print(f"[{key}] 通过端口 {port} 定位 pid={pids}，强制停止...", end=" ", flush=True)
        for p in pids:
            kill_pid(p, force=True)
        time.sleep(2)
        if not is_port_listening(port):
            print("已停止")
            remove_pid(cfg["pid_file"])
            return True
        print("失败")
        return False

    print(f"[{key}] 未在运行")
    remove_pid(cfg["pid_file"])
    return True


def restart_service(key: str) -> bool:
    stop_service(key)
    time.sleep(1)
    return start_service(key)


def show_status():
    print(f"{'SERVICE':<16} {'PORT':<8} {'STATUS':<14} {'PID':<10} LOG")
    print("-" * 88)
    for key, cfg in SERVICES.items():
        port = cfg["port"]
        pid = read_pid(cfg["pid_file"])
        if is_port_listening(port):
            status = "running ✅"
            pid_str = str(pid) if (pid and is_pid_running(pid)) else "?"
        else:
            status = "stopped ❌"
            pid_str = "-"
        print(f"{key:<16} {port:<8} {status:<14} {pid_str:<10} {cfg['log_file']}")


def show_logs(key: str, lines: int = 50):
    cfg = SERVICES[key]
    if not os.path.exists(cfg["log_file"]):
        print(f"日志文件不存在: {cfg['log_file']}")
        return
    with open(cfg["log_file"], "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    for line in all_lines[-lines:]:
        print(line, end="")


def kill_port(port: int):
    pids = find_pids_on_port(port)
    if not pids:
        print(f"端口 {port} 无监听进程")
        return
    print(f"终止端口 {port} 的进程: {pids}")
    for p in pids:
        kill_pid(p, force=True)
    time.sleep(1)
    print("完成" if not is_port_listening(port) else "仍有进程监听")


# ============================== 入口 ==============================

USAGE = """DehazeSystem 后端服务生命周期管理（docker-like）

用法:
    python scripts/run.py <command> [args...]

命令:
    run <service>[,service...]      启动服务
    stop <service>[,service...]     停止服务
    restart <service>[,service...]  重启服务
    ps                              查看所有服务状态
    logs <service> [lines]          查看服务日志（默认 50 行）
    kill <port>                     杀掉占用端口的进程

服务名 (支持别名 / 逗号分隔 / all):
    dehaze-go | go         端口 8990
    dehaze-python | python 端口 8991
    dehaze-java | java     端口 8989

示例:
    python scripts/run.py run dehaze-go
    python scripts/run.py run dehaze-go,dehaze-java
    python scripts/run.py stop all
    python scripts/run.py restart dehaze-go
    python scripts/run.py ps
    python scripts/run.py logs dehaze-python 100
"""


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return

    cmd = args[0]
    rest = args[1:]

    if cmd in ("run", "stop", "restart"):
        if not rest:
            print(f"用法: {cmd} <service>[,service...]")
            sys.exit(1)
        keys = resolve_services(rest[0])
        handler = {"run": start_service, "stop": stop_service,
                   "restart": restart_service}[cmd]
        for k in keys:
            handler(k)
    elif cmd == "ps":
        show_status()
    elif cmd == "logs":
        if not rest:
            print("用法: logs <service> [lines]")
            sys.exit(1)
        key = ALIAS_TO_KEY.get(rest[0])
        if not key:
            print(f"未知服务: {rest[0]}，可选: {sorted(ALIAS_TO_KEY.keys())}")
            sys.exit(1)
        lines = int(rest[1]) if len(rest) > 1 else 50
        show_logs(key, lines)
    elif cmd == "kill":
        if not rest:
            print("用法: kill <port>")
            sys.exit(1)
        kill_port(int(rest[0]))
    else:
        print(f"未知命令: {cmd}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
