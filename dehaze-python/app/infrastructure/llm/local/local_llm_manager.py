"""本地轻量 LLM 子进程管理（懒启动 + 健康探测 + 进程守护）

local provider 的推理服务运行在独立子进程（OpenAI 兼容，默认 127.0.0.1:8992）。
首次需要本地模型时：自动下载模型文件（见 local_llm_model）-> 拉起子进程。
主服务退出时回收子进程。不做常驻心跳重建（生产可用进程管理器托管同一入口），
仅保证开发/测试/默认部署"零手工步骤"即可用。

健康判定与误杀防护：
- 推理满载（尤其纯 CPU 构建）时 /health 响应会明显变慢，属正常现象而非假死，
  探测超时需留足余量，且对自管理子进程只等待恢复、绝不误杀；
- 仅当端口被外部进程占用且不健康（主进程被强杀遗留的孤儿）时才 SIGKILL 接管；
- ensure_running 全程持锁串行化，防止并发请求交叉误杀对方刚拉起的进程。
"""

import atexit
import io
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

import app as app_pkg
from app.config import settings
from app.infrastructure.llm.local.local_llm_model import ensure_embedding_model, ensure_model

logger = logging.getLogger(__name__)

_PROC: subprocess.Popen | None = None
_SHUTDOWN_REGISTERED = False

# 子进程 stdout/stderr 落盘的日志文件句柄：由父进程持有，shutdown 时关闭。
# 子进程经 Popen 继承该 fd 的副本，父进程 close 前句柄保持有效，避免 fd 泄漏。
_LOG_FILE: io.TextIOWrapper | None = None

# ensure_running 串行化：并发请求（推理重试/多会话并发）同时探测、拉起或清理时，
# 保证只有一个执行者，避免交叉误杀对方刚拉起的进程或重复拉起
_ensure_lock = threading.Lock()

# 自管理子进程"忙而不死"的等待上限：推理满载拖慢 /health 属正常，等待恢复；
# 超时仍无响应才判定真死锁，杀掉重启自愈
_RECOVERY_WAIT_SECONDS = 60.0


def _base_url() -> str:
    return f"http://{settings.LOCAL_LLM_HOST}:{settings.LOCAL_LLM_PORT}"


def _port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((settings.LOCAL_LLM_HOST, settings.LOCAL_LLM_PORT)) == 0


def _healthy() -> bool:
    """健康探测：要求服务就绪且模型已加载（loaded=true）。

    /health 仅在 loaded=true 时返回就绪语义；仅进程存活（loaded=false）不算就绪，
    否则 ensure_running 会在模型加载完成前返回，首次推理请求将因模型未就绪而失败。
    超时 10s：推理满载（纯 CPU 构建占满所有核心）时 /health 响应变慢属正常，
    短超时会把健康进程误判为假死，误杀将中断所有进行中的推理流。
    """
    try:
        resp = httpx.get(f"{_base_url()}/health", timeout=10.0)
        if resp.status_code != 200:
            return False
        return resp.json().get("loaded", False) is True
    except (httpx.HTTPError, ValueError) as e:
        # 探测契约是返回 bool、绝不抛出：连接被拒/超时/响应非合法 JSON 均判为不健康。
        # 该探测在启动等待与重连循环中高频调用，明细用 debug 留痕（避免刷屏），
        # 用于区分"未就绪/满载忙"与"端口被非本服务占用"等真实故障
        logger.debug("本地 LLM 健康探测失败: %s", e)
        return False


def _port_holder_pid() -> int | None:
    """占用 LOCAL_LLM_PORT 监听端口的进程 pid（排除主进程自身）"""
    try:
        output = subprocess.run(
            ["ss", "-tlnpH", f"sport = :{settings.LOCAL_LLM_PORT}"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        pids = [int(m) for m in re.findall(r"pid=(\d+)", output) if int(m) != os.getpid()]
        return pids[0] if pids else None
    except Exception as exc:
        logger.warning("解析端口占用进程失败: %s", exc)
        return None


def _self_managed_pid() -> int | None:
    """当前主进程自管理的子进程 pid（已启动且未退出）"""
    return _PROC.pid if _PROC is not None and _PROC.poll() is None else None


def _wait_recovery() -> bool:
    """等待自管理子进程恢复健康，返回是否恢复。

    进程存活但 /health 无响应通常是推理满载的正常表现（CPU 争用拖慢探测响应），
    等待即可；进程退出或等待超时（真死锁）返回 False，由调用方重启。
    """
    deadline = time.monotonic() + _RECOVERY_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _healthy():
            return True
        if _self_managed_pid() is None:
            return False  # 进程已退出，直接走重启
        time.sleep(0.5)
    return False


def ensure_running() -> str:
    """确保本地 LLM 服务可用，返回 base_url。

    模型不存在时自动下载（首次约 378MB，含进度日志与断点续传）；
    下载或启动失败抛 RuntimeError，由调用方决定错误语义。
    注意：本方法可能长时间阻塞（下载/等待满载进程恢复），异步上下文请用
    asyncio.to_thread 包装。
    """
    with _ensure_lock:
        if _healthy():
            return _base_url()

        if _port_in_use():
            holder = _port_holder_pid()
            if holder is not None and holder == _self_managed_pid():
                # 自管理子进程存活但探测无响应：推理满载的正常表现，等待恢复
                # （覆盖子进程启动后模型加载中的场景）；等待超时才判定真死锁重启
                logger.warning(
                    "本地 LLM 子进程 pid=%s 健康探测无响应（推理满载属正常），等待恢复", holder
                )
                if _wait_recovery():
                    return _base_url()
                logger.warning("本地 LLM 子进程 pid=%s 持续无响应，判定假死，重启自愈", holder)
            else:
                # 端口被外部进程占用且不健康：主进程被强杀（kill -9 / 部署重启）遗留的
                # 假死孤儿（atexit 不会触发），或无关进程占端口。不健康的服务无保留
                # 价值，直接杀掉接管重启。
                _kill_port_holder(holder)
                if _port_in_use():
                    raise RuntimeError(
                        f"端口 {settings.LOCAL_LLM_PORT} 被非本地 LLM 服务占用，"
                        "请检查 LOCAL_LLM_PORT 配置"
                    )

        ensure_model()  # 不存在则自动下载
        ensure_embedding_model()  # 向量模型由同一子进程提供（知识库检索依赖）
        _start_and_wait()
        return _base_url()


def _open_subprocess_log() -> io.TextIOWrapper | None:
    """打开本次子进程输出日志（logs/{yyyy-MM-dd}/local_llm.log，行缓冲追加）。

    子进程 stdout/stderr 必须落盘：DEVNULL 会让服务端日志、崩溃栈与 llama.cpp 加载
    错误全部不可见（本项目排查困难的头号来源）。按启动日期命名即可（进程生命周期短、
    重启频繁，无需滚动）；父进程启动日志时该目录通常已存在，仍兜底创建以防缺失。
    创建失败**不静默退回 DEVNULL**（那等于问题继续隐身），留痕后由调用方降级。
    """
    path = Path(settings.LOG_DIR, datetime.now().strftime("%Y-%m-%d"), "local_llm.log")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("本地 LLM 子进程日志: %s", path)
        return path.open("a", encoding="utf-8", buffering=1)
    except OSError as e:
        logger.warning("本地 LLM 子进程日志文件创建失败，输出将丢弃: %s", e, exc_info=True)
        return None


def _start_and_wait() -> None:
    """拉起子进程并等待就绪（含模型加载）"""
    global _PROC, _SHUTDOWN_REGISTERED, _LOG_FILE
    logger.info("拉起本地 LLM 子进程（%s）", _base_url())
    # cwd 指向 dehaze-python 根（app 包所在），保证 `python -m app.xxx` 能解析模块；
    # 以 app 包位置为锚点，避免按 __file__ 相对层数推算在重构后漂移
    root = str(Path(app_pkg.__file__).resolve().parent.parent)
    # 子进程 stdout/stderr 同指向日志文件，日志与崩溃栈可查；文件创建失败才退回 DEVNULL
    _LOG_FILE = _open_subprocess_log()
    stdout_target: int | io.TextIOWrapper = (
        _LOG_FILE if _LOG_FILE is not None else subprocess.DEVNULL
    )
    # preexec_fn 仅 POSIX 支持（Windows 传入直接抛 ValueError）；PDEATHSIG 见 _set_pdeathsig
    popen_kwargs: dict = {"preexec_fn": _set_pdeathsig} if os.name == "posix" else {}
    _PROC = subprocess.Popen(
        [sys.executable, "-m", "app.infrastructure.llm.local.local_llm_server"],
        cwd=root,
        stdout=stdout_target,
        stderr=stdout_target,
        **popen_kwargs,
    )
    if not _SHUTDOWN_REGISTERED:
        atexit.register(shutdown)
        _SHUTDOWN_REGISTERED = True

    for _ in range(120):  # 最多等 60s（含模型加载约 5-15s）
        if _healthy():
            logger.info("本地 LLM 服务就绪: %s", _base_url())
            return
        if _PROC.poll() is not None:
            msg = (
                f"本地 LLM 子进程启动失败（退出码 {_PROC.returncode}），"
                "请检查 llama-cpp-python 安装与模型文件"
            )
            raise RuntimeError(msg)
        time.sleep(0.5)
    raise RuntimeError("本地 LLM 服务启动超时")


def _set_pdeathsig() -> None:  # pragma: no cover - 仅在 Linux 子进程内执行
    """PR_SET_PDEATHSIG：主进程退出（含被 kill）时子进程收到 SIGTERM 自动退出。

    在 preexec_fn（fork 后的子进程）内执行，属尽力而为：非 glibc 平台（musl/Alpine）
    载入 libc.so.6 抛 OSError、符号缺失抛 AttributeError，此时跳过（父进程 atexit 仍
    负责正常回收），但必须留痕以定位该平台 Pdeathsig 未生效。
    """
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1
    except (OSError, AttributeError) as e:
        logger.warning(
            "设置 PR_SET_PDEATHSIG 失败（子进程可能不随父进程退出）: %s", e, exc_info=True
        )


def _kill_port_holder(pid: int | None) -> None:
    """杀掉占用端口的假死孤儿进程（pid 已由调用方解析）"""
    if pid is None:
        return
    logger.warning("终止假死的本地 LLM 孤儿进程 pid=%s", pid)
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)  # 等端口释放
    except ProcessLookupError:
        # 进程在探测后、kill 前已自行退出：目标已达成，留痕以便排查端口占用时序
        logger.debug("端口占用进程 pid=%s 已退出，无需终止", pid)
    except Exception as exc:
        logger.warning("清理端口占用进程失败: %s", exc)


def shutdown() -> None:
    """回收本地 LLM 子进程（lifespan 优雅关闭调用，atexit 兜底复用，幂等）。"""
    global _PROC, _LOG_FILE
    proc = _PROC
    if proc is not None:
        if proc.poll() is None:
            logger.info("回收本地 LLM 子进程 pid=%s", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # terminate 5s 未退出才升级 SIGKILL：必须留痕，否则"优雅停止失败"
                # 这一信号被静默吞掉，进程异常挂起将无从排查
                logger.warning("本地 LLM 子进程 5s 内未退出，升级 SIGKILL: pid=%s", proc.pid)
                proc.kill()
        _PROC = None
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None
