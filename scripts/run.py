#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform == "win32"
VENV_PY = str(ROOT / "dehaze-python" / ".venv" /
              ("Scripts" if IS_WIN else "bin") /
              ("python.exe" if IS_WIN else "python"))

SERVICES: dict[str, tuple[Path, list[str], int]] = {
    "go":     (ROOT / "dehaze-go",     ["go", "run", "./cmd/main.go"], 8990),
    "python": (ROOT / "dehaze-python", [VENV_PY, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8991"], 8991),
    "java":   (ROOT / "dehaze-java",   ["mvn.cmd" if IS_WIN else "mvn", "spring-boot:run", "-DskipTests"], 8989),
}

USAGE = """DehazeSystem 后端服务管理

用法:
    run.py run|stop|restart <svc[,svc]|all>
    run.py ps
    run.py logs <svc> [lines]

服务: go(8990)  python(8991)  java(8989)

日志统一存放在各服务 logs/{yyyy-MM-dd}/ 下：
    console.log  启动/控制台输出（本脚本重定向）
    info.log     应用 INFO 及以上日志（JSON 结构化）
    error.log    应用 ERROR 日志（JSON 结构化）
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _console_log_path(cwd: Path) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return cwd / "logs" / today / "console.log"


def start(svc: str):
    cwd, cmd, port = SERVICES[svc]
    pid_file = cwd / f".{svc}.pid"
    try:
        pid = int(pid_file.read_text())
        if _alive(pid):
            print(f"[{svc}] already running (pid={pid})")
            return
    except (FileNotFoundError, ValueError):
        pass

    log_path = _console_log_path(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if IS_WIN else 0,
        start_new_session=not IS_WIN,
    )
    pid_file.write_text(str(proc.pid))
    print(f"[{svc}] started (pid={proc.pid}, :{port})")
    print(f"[{svc}] console log: {log_path}")


def _kill_port_occupants(port: int):
    # 清理占用端口的残留进程（go run/uvicorn --reload 的孤儿子进程，PPID=1，不在 PID 文件记录的进程组）
    if IS_WIN:
        return
    out = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    for p in set(re.findall(r"pid=(\d+)", out)):
        try:
            os.killpg(os.getpgid(int(p)), 9)
        except (ProcessLookupError, PermissionError):
            pass


def stop(svc: str):
    cwd = SERVICES[svc][0]
    pid_file = cwd / f".{svc}.pid"
    try:
        pid = int(pid_file.read_text())
    except (FileNotFoundError, ValueError):
        print(f"[{svc}] not running")
        return

    if IS_WIN:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        # start 用 start_new_session=True，子进程（go run 二进制、uvicorn worker）在同一进程组，
        # 组 ID = 父 PID，需 killpg 才能连带杀掉子进程
        try:
            os.killpg(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        _kill_port_occupants(SERVICES[svc][2])
    pid_file.unlink(missing_ok=True)
    print(f"[{svc}] stopped")


def status():
    for svc, (cwd, _, port) in SERVICES.items():
        try:
            pid = int((cwd / f".{svc}.pid").read_text())
            st = f"running (pid={pid})" if _alive(pid) else "stopped"
        except (FileNotFoundError, ValueError):
            st = "stopped"
        print(f"{svc:<10} :{port:<5} {st}")


def show_logs(svc: str, n: int = 50):
    log_dir = SERVICES[svc][0] / "logs"
    if not log_dir.is_dir():
        print(f"[{svc}] 无日志目录 {log_dir}")
        return
    date_dirs = sorted(
        [d for d in log_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
        reverse=True,
    )
    if not date_dirs:
        print(f"[{svc}] 无日志")
        return
    p = date_dirs[0] / "console.log"
    if p.exists():
        for line in p.read_text(errors="replace").splitlines()[-n:]:
            print(line)
    else:
        print(f"[{svc}] 无控制台日志，可查看 {date_dirs[0]}/info.log")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return

    cmd, *rest = args

    if cmd in ("run", "stop", "restart"):
        if not rest:
            print(f"用法: {cmd} <svc[,svc]|all>")
            sys.exit(1)
        svcs = list(SERVICES) if rest[0] == "all" else [s for s in rest[0].split(",") if s in SERVICES]
        for s in svcs:
            if cmd == "run":
                start(s)
            elif cmd == "stop":
                stop(s)
            elif cmd == "restart":
                stop(s)
                start(s)
    elif cmd == "ps":
        status()
    elif cmd == "logs":
        if not rest or rest[0] not in SERVICES:
            print("用法: logs <svc> [lines]")
            sys.exit(1)
        show_logs(rest[0], int(rest[1]) if len(rest) > 1 else 50)
    else:
        print(f"未知命令: {cmd}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
