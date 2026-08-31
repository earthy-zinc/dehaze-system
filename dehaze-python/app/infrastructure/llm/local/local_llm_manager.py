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
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx

import app as app_pkg
from app.config import settings
from app.infrastructure.llm.local.local_llm_model import ensure_embedding_model, ensure_model

logger = logging.getLogger(__name__)

_PROC: subprocess.Popen | None = None
_SHUTDOWN_REGISTERED = False

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
    except Exception:
        return False


def _port_holder_pid() -> int | None:
    """占用 LOCAL_LLM_PORT 监听端口的进程 pid（排除主进程自身）"""
    try:
        output = subprocess.run(
            ["ss", "-tlnpH", f"sport = :{settings.LOCAL_LLM_PORT}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        pids = [int(m) for m in re.findall(r"pid=(\d+)", output) if int(m) != os.getpid()]
        return pids[0] if pids else None
    except Exception as exc:  # noqa: BLE001 解析失败由上层端口复查兜底
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


def _start_and_wait() -> None:
    """拉起子进程并等待就绪（含模型加载）"""
    global _PROC, _SHUTDOWN_REGISTERED
    logger.info("拉起本地 LLM 子进程（%s）", _base_url())
    # cwd 指向 dehaze-python 根（app 包所在），保证 `python -m app.xxx` 能解析模块；
    # 以 app 包位置为锚点，避免按 __file__ 相对层数推算在重构后漂移
    root = os.path.dirname(os.path.dirname(os.path.abspath(app_pkg.__file__)))
    # preexec_fn 仅 POSIX 支持（Windows 传入直接抛 ValueError）；PDEATHSIG 见 _set_pdeathsig
    popen_kwargs: dict = {"preexec_fn": _set_pdeathsig} if os.name == "posix" else {}
    _PROC = subprocess.Popen(
        [sys.executable, "-m", "app.infrastructure.llm.local.local_llm_server"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
            raise RuntimeError(
                "本地 LLM 子进程启动失败（退出码 %s），请检查 llama-cpp-python 安装与模型文件"
                % _PROC.returncode
            )
        time.sleep(0.5)
    raise RuntimeError("本地 LLM 服务启动超时")


def _set_pdeathsig() -> None:  # pragma: no cover - 仅在 Linux 子进程内执行
    """PR_SET_PDEATHSIG：主进程退出（含被 kill）时子进程收到 SIGTERM 自动退出"""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG=1
    except Exception:  # noqa: BLE001 非 Linux 环境无此机制，跳过（依赖 atexit 兜底）
        pass


def _kill_port_holder(pid: int | None) -> None:
    """杀掉占用端口的假死孤儿进程（pid 已由调用方解析）"""
    if pid is None:
        return
    logger.warning("终止假死的本地 LLM 孤儿进程 pid=%s", pid)
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)  # 等端口释放
    except ProcessLookupError:
        pass
    except Exception as exc:  # noqa: BLE001 清理失败由上层端口复查兜底
        logger.warning("清理端口占用进程失败: %s", exc)


def shutdown() -> None:
    """回收本地 LLM 子进程（lifespan 优雅关闭调用，atexit 兜底复用，幂等）。"""
    global _PROC
    if _PROC is None:
        return
    if _PROC.poll() is None:
        logger.info("回收本地 LLM 子进程 pid=%s", _PROC.pid)
        _PROC.terminate()
        try:
            _PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _PROC.kill()
    _PROC = None
