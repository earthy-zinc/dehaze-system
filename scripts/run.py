#!/usr/bin/env python3
# 注解延迟求值：宿主 python3 可能是 3.8（本脚本大量使用 `dict[str, ...]` 这类内置泛型注解）
from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
IS_WIN = sys.platform == "win32"
# CREATE_NO_WINDOW 仅 Windows 有（typeshed 按平台条件声明），跨平台直接取属性会触发静态检查告警；非 Windows 下为 0 即默认行为
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SERVICE_ROOT = ROOT / "dehaze-service"
MVN = "mvn.cmd" if IS_WIN else "mvn"


def _venv_py(*parts: str) -> str:
    """项目私有虚拟环境的解释器路径（各项目自行 uv sync 生成，不共享依赖）"""
    return str(ROOT.joinpath(*parts, ".venv", "Scripts" if IS_WIN else "bin",
                             "python.exe" if IS_WIN else "python"))


SERVICES: dict[str, tuple[Path, list[str], int]] = {
    "go":     (ROOT / "dehaze-go",     ["go", "run", "./cmd/main.go"], 8990),
    "python": (ROOT / "dehaze-python", [_venv_py("dehaze-python"), "-m", "app.main"], 8991),
    "java":   (ROOT / "dehaze-java",   [MVN, "spring-boot:run", "-DskipTests"], 8989),
    # uv 走系统证书库：办公网/企业代理会替换 TLS 证书链，uv 内置根证书校验不过（UnknownIssuer）
    "mcp":    (ROOT / "dehaze-mcp-gateway", ["uv", "run", "--native-tls", "python", "server.py"], 8082),
    # 外置推理服务（dehaze-service/*，内网专用，各自独立 venv）
    "llm":    (_SERVICE_ROOT / "dehaze-llm",   [_venv_py("dehaze-service", "dehaze-llm"), "-m", "app.main"], 8992),
    "voice":  (_SERVICE_ROOT / "dehaze-voice", [_venv_py("dehaze-service", "dehaze-voice"), "-m", "app.main"], 8993),
    "modal":  (_SERVICE_ROOT / "dehaze-modal", [_venv_py("dehaze-service", "dehaze-modal"), "-m", "app.main"], 8994),
}

# 需要存活守护的服务：uvicorn 的 reload（dehaze-python 在 dev 下 `reload=settings.DEBUG`）会经
# pdeathsig 连带杀掉本地 LLM 子进程（8992），表现是推理服务**静默消失**——上游只看到超时/失败。
# 由守护进程持有 llm 进程后，其父进程不再是 reloader；万一仍被杀，也由守护按 console 日志记录
# 原因后重拉。守护只服务该集合内的服务，不改变其它服务的生命周期语义。
GUARDED_SERVICES = ("llm",)
# 守护重拉节流（秒）：进程反复起不来时避免忙循环刷日志，同时把恢复时延控制在秒级
GUARD_RESTART_DELAY = 3.0
# 内部子命令：由 start() 拉起服务守护，不在 USAGE 中暴露、不建议手工调用
GUARD_COMMAND = "_guard"

USAGE = """DehazeSystem 后端服务管理

用法:
    run.py run|start|stop|restart <svc[,svc]|all>
    run.py ps
    run.py logs <svc> [lines]

服务: go(8990)  python(8991)  java(8989)  mcp(8082)  llm(8992)  voice(8993)  modal(8994)

日志统一存放在各服务 logs/{yyyy-MM-dd}/ 下：
    console.log  启动/控制台输出（本脚本重定向）
    info.log     应用 INFO 及以上日志（JSON 结构化）
    error.log    应用 ERROR 日志（JSON 结构化）
"""


def _alive(pid: int) -> bool:
    if IS_WIN:
        # Windows 下 os.kill(pid, 0) 会发送 CTRL_C_EVENT 而非存活检查，改用 tasklist
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _console_log_path(cwd: Path) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return cwd / "logs" / today / "console.log"


def _env_file() -> dict[str, str]:
    """读取仓库根 .env 的键值对。

    三端各自也会读 .env，但 SPRING_PROFILES_ACTIVE / GIN_MODE / APP_ENV 必须在进程启动那一刻
    就可见（Go 靠 GIN_MODE 决定读 config.yaml 还是 config.prod.yaml，Python 靠 APP_ENV 选配置类），
    各自读 .env 的时刻都晚于这一步，故由脚本注入为真实环境变量。"""
    try:
        content = (ROOT / ".env").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _start_env(svc: str) -> dict[str, str]:
    """按部署拓扑补齐启动环境：xxl-job-admin 跑在 Docker 容器内、三端后端跑在宿主，
    执行器的注册地址必须是容器可回调的地址。host.docker.internal 由 Docker Desktop
    提供，仅装 Docker 的 Windows/macOS 可解析（未装 Docker 时 XXL-Job 初始化会因域名
    解析失败抛 NPE 而启动失败）；解析不到则留空交由执行器探测本机 IP。
    该值属拓扑信息而非业务配置，故由脚本注入而不写进各端配置；外部已设置时以外部为准。"""
    env = os.environ.copy()
    # 云 IDE 会给 shell 注入 SERVER__PORT（指向 IDE 自身监听端口），Spring Boot 的
    # 宽松绑定会把它当 server.port，导致 java 起在 IDE 的端口上并 bind 失败。
    # .env 已用 JAVA_SERVER_PORT 显式表达意图（与 Python 端 SERVER_PORT 区分），此处剔除噪音。
    env.pop("SERVER__PORT", None)
    # 云 IDE 还会注入 PYTHONPATH（sitecustomize 删除钩子会把 unlink 变成 XDG trash 的
    # shutil.move 并递归调用自身 ⇒ RecursionError，曾导致 go 被停掉却没起来）；
    # 各服务一律用自带 venv，不依赖外部 PYTHONPATH。
    env.pop("PYTHONPATH", None)
    for key, value in _env_file().items():
        env.setdefault(key, value)
    if svc in ("java", "go", "python"):
        try:
            socket.getaddrinfo("host.docker.internal", None)
        except OSError:
            pass
        else:
            env.setdefault("XXLJOB_EXECUTOR_IP", "host.docker.internal")
    return env


def _prod_jar(cwd: Path) -> Optional[Path]:
    """prod profile 下的 java 生产化启动：先构建，再从 target/ 取唯一的 repackage fat jar。

    构建失败或产物不唯一时打印原因并返回 None，交由调用方中止启动，绝不猜路径继续。"""
    build = subprocess.run(
        [MVN, "-q", "-DskipTests", "package"],
        cwd=cwd, capture_output=True, text=True,
    )
    if build.returncode != 0:
        print(f"[java] 构建失败（mvn package 退出码 {build.returncode}），已中止启动：")
        print((build.stderr or build.stdout).strip()[-2000:])
        return None
    jars = [
        p for p in (cwd / "target").glob("*.jar")
        if not p.name.endswith(".original")
        and "-sources" not in p.name
        and "-javadoc" not in p.name
    ]
    if len(jars) != 1:
        print(f"[java] target/ 下未找到唯一 fat jar（命中 {[p.name for p in jars]}），已中止启动")
        return None
    return jars[0]


def _spawn(
    svc: str, cmd: list[str], cwd: Path, log: IO[str], *, new_session: bool
) -> subprocess.Popen:
    """按统一口径启动子进程：输出追加到 console 日志、Windows 下不弹控制台窗口。

    new_session=True（`start` 直接启动）：进程成为新会话/进程组首进程，`stop` 可按 PGID 整组清理；
    守护进程内的被守护服务用 new_session=False ⇒ 与服务守护同组，`stop` 一并带走。"""
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=_start_env(svc),
        creationflags=CREATE_NO_WINDOW,
        start_new_session=new_session and not IS_WIN,
    )


def _skip_unprovisioned(svc: str, cmd: list[str]) -> None:
    """未 provision（dehaze-service/* 首次需在该目录 uv sync）视为可缺省：打印提示后跳过，不报错。

    --native-tls 同 SERVICES 中 uv 命令的理由：企业代理替换证书链时需走系统证书库。"""
    print(f"[{svc}] skip：未 provision（未找到 {cmd[0]}），如需本地/GPU 部署请先在该目录 uv sync --native-tls")


def _launch_argv(svc: str, cmd: list[str]) -> list[str]:
    """守护服务的实际启动命令：改拉本脚本的守护子命令（复用自身解释器与代码，不引新依赖/新文件）"""
    if svc in GUARDED_SERVICES:
        return [sys.executable, str(SELF), GUARD_COMMAND, svc]
    return cmd


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _exit_reason(code: int) -> str:
    """退出码可读化：负值为信号终止（-9/SIGKILL 常见于被连带 kill，如 pdeathsig 或 kill -9）"""
    if code < 0:
        try:
            return f"被信号 {signal.Signals(-code).name} 终止"
        except ValueError:
            return f"被信号 {-code} 终止"
    return "正常退出" if code == 0 else f"异常退出（退出码 {code}）"


def guard(svc: str) -> None:
    """服务守护主体（内部命令，由 `start` 拉起）：持有服务进程，进程消失即记录原因并重拉。

    - 守护进程由 `start` 以独立会话拉起，被守护服务不另开会话 ⇒ `run.py stop <svc>` 的进程组
      清理同时带走两者，不遗留常驻进程；
    - 只重拉自己持有的那个子进程，不触碰其它服务的生命周期；
    - 每次进程消失都在 console 日志写明时间、pid 与退出原因（pdeathsig 连带杀、OOM、崩溃均可归因）；
      进程起不来时按 GUARD_RESTART_DELAY 节流重试，日志逐次可见，不静默吞掉失败。"""
    cwd, cmd, port = SERVICES[svc]
    pid_file = cwd / f".{svc}.pid"
    log_path = _console_log_path(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        while True:
            try:
                child = _spawn(svc, cmd, cwd, log, new_session=False)
            except FileNotFoundError:
                # 未 provision（dehaze-service/* 首次需在该目录 uv sync）：同 start 的 skip 口径
                print(
                    f"[{svc}] {_now()} 守护退出：未找到 {cmd[0]}（服务未 provision）",
                    file=log,
                    flush=True,
                )
                pid_file.unlink(missing_ok=True)
                return
            # PID 文件记守护 pid：`ps` 显示 running，`stop` 按该进程组整组清理
            pid_file.write_text(str(os.getpid()))
            print(f"[{svc}] {_now()} 守护拉起服务进程 (pid={child.pid}, :{port})", file=log, flush=True)
            code = child.wait()
            print(
                f"[{svc}] {_now()} 服务进程消失 (pid={child.pid}, {_exit_reason(code)})，"
                f"{GUARD_RESTART_DELAY:.0f}s 后重拉",
                file=log,
                flush=True,
            )
            time.sleep(GUARD_RESTART_DELAY)


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

    # prod（含复用 prod 的 pre）走 mvn package + java -jar：devtools 只在 fat jar 打包时被排除，
    # spring-boot:run 会直接加载它并在共享机上被 maven 构建触发热重启（详见《06-部署架构》§6.2）
    if svc == "java" and _env_file().get("SPRING_PROFILES_ACTIVE", "dev") == "prod":
        jar = _prod_jar(cwd)
        if jar is None:
            return
        cmd = ["java", "-jar", str(jar)]

    # 未 provision（dehaze-service/* 首次需在该目录 uv sync 生成 venv）视为可缺省：跳过而非报错，
    # 不中断其余服务。llm/voice 已退出默认链路、按需手动启动（验证 local / 极简协议调用时用）；modal 仍是去雾核心。
    if not cwd.is_dir():
        print(f"[{svc}] skip：服务未 provision（目录缺失 {cwd}），如需本地/GPU 部署请先在该目录 uv sync")
        return

    # 守护服务由守护进程持有子进程，解释器缺失要到守护内才暴露；为保留"未 provision 直接跳过"
    # 的既有提示（而不是先报 started 再由守护退出），拉起前按同一判据先看一眼解释器绝对路径
    interpreter = Path(cmd[0])
    if svc in GUARDED_SERVICES and interpreter.is_absolute() and not interpreter.exists():
        _skip_unprovisioned(svc, cmd)
        return

    log_path = _console_log_path(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    try:
        proc = _spawn(svc, _launch_argv(svc, cmd), cwd, log, new_session=True)
    except FileNotFoundError:
        # 解释器缺失（dehaze-service 各服务首次需在该目录同步依赖），同属未 provision：跳过。
        log.close()
        _skip_unprovisioned(svc, cmd)
        return
    pid_file.write_text(str(proc.pid))
    # 守护服务记的是守护进程 pid：服务进程消失不会让 PID 文件失效，`stop` 仍能按进程组收口
    print(f"[{svc}] started (pid={proc.pid}{' 守护进程' if svc in GUARDED_SERVICES else ''}, :{port})")
    if svc in GUARDED_SERVICES:
        print(f"[{svc}] 存活守护已启用：服务进程消失即记录原因并重拉（见 console log）")
    print(f"[{svc}] console log: {log_path}")


def stop(svc: str):
    cwd = SERVICES[svc][0]
    pid_file = cwd / f".{svc}.pid"
    port = SERVICES[svc][2]

    # 收集目标 PID：PID 文件 + 端口占用（ss 可能返回多个，如 uvicorn --reload 的 reloader+worker）
    file_pid = None
    try:
        file_pid = int(pid_file.read_text())
    except (FileNotFoundError, ValueError):
        pass

    target_pids = {file_pid, *_port_pids(port)} - {None}

    killed = False
    # 第一轮：按进程组清理（start_new_session=True 启动的进程 PGID=PID）
    for pid in target_pids:
        if not _alive(pid):
            continue
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(pid), 9)
            killed = True
        except (ProcessLookupError, PermissionError):
            pass

    # 第二轮：复查端口兜底——go run 编译产物、uvicorn --reload 的 worker 可能被 init
    # 接管（PPID=1）脱离原进程组，killpg 杀不到，需按端口逐个 kill 清理
    for _ in range(3):
        leftover = _port_pids(port)
        if not leftover:
            break
        for pid in leftover:
            try:
                if IS_WIN:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                else:
                    os.kill(pid, 9)
                killed = True
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.3)

    pid_file.unlink(missing_ok=True)
    print(f"[{svc}] stopped" if killed else f"[{svc}] not running")


def _port_pids(port: int) -> list[int]:
    """返回所有监听指定端口的进程 PID，无监听则返回空列表"""
    if IS_WIN:
        # Windows 无 ss，用 netstat -ano，最后一列为 PID，匹配 LISTENING 且本地地址以 :port 结尾
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True,
        ).stdout
        pids = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-2] == "LISTENING" and parts[1].endswith(f":{port}"):
                try:
                    pids.append(int(parts[-1]))
                except ValueError:
                    pass
        return pids
    if sys.platform == "darwin":
        # macOS 无 ss，lsof -t 直接输出监听进程 PID
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP", f":{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True,
        ).stdout
        return [int(line) for line in out.split() if line.isdigit()]
    out = subprocess.run(
        ["ss", "-tlnp", f"sport = :{port}"],
        capture_output=True, text=True,
    ).stdout
    return [int(m) for m in re.findall(r"pid=(\d+)", out)]


def status():
    for svc, (cwd, _, port) in SERVICES.items():
        pid = None
        # 优先读取 PID 文件
        try:
            pid = int((cwd / f".{svc}.pid").read_text())
        except (FileNotFoundError, ValueError):
            pass

        if pid and _alive(pid):
            st = f"running (pid={pid})"
        else:
            # 兜底：通过端口检测实际运行的进程
            pids = _port_pids(port)
            st = f"running (pid={pids[0]})" if pids else "stopped"
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

    if cmd == GUARD_COMMAND:
        # 内部命令：由 start() 拉起服务守护（见 GUARDED_SERVICES），不在 USAGE 中暴露
        if len(rest) != 1 or rest[0] not in SERVICES:
            print(f"用法: {GUARD_COMMAND} <svc>（内部命令，由 start 调用）")
            sys.exit(1)
        guard(rest[0])
        return

    if cmd in ("run", "start", "stop", "restart"):
        if not rest:
            print(f"用法: {cmd} <svc[,svc]|all>")
            sys.exit(1)
        svcs = list(SERVICES) if rest[0] == "all" else [s for s in rest[0].split(",") if s in SERVICES]
        for s in svcs:
            if cmd in ("run", "start"):
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
