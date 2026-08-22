"""本地轻量 LLM 子进程管理（懒启动 + 健康探测 + 进程守护）

local provider 的推理服务运行在独立子进程（OpenAI 兼容，默认 127.0.0.1:8992）。
首次需要本地模型时：自动下载模型文件（见 local_llm_model）→ 拉起子进程。
主服务退出时回收子进程。不做常驻心跳重建（生产可用进程管理器托管同一入口），
仅保证开发/测试/默认部署"零手工步骤"即可用。
"""

import atexit
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time

import httpx

from app.config import settings
from app.service.ai.local_llm_model import ensure_embedding_model, ensure_model

logger = logging.getLogger(__name__)

_PROC: subprocess.Popen | None = None
_SHUTDOWN_REGISTERED = False


def _base_url() -> str:
    return f"http://{settings.LOCAL_LLM_HOST}:{settings.LOCAL_LLM_PORT}"


def _port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((settings.LOCAL_LLM_HOST, settings.LOCAL_LLM_PORT)) == 0


def _healthy() -> bool:
    """健康探测：要求服务就绪且模型已加载（loaded=true）。

    /health 仅在 loaded=true 时返回就绪语义；仅进程存活（loaded=false）不算就绪，
    否则 ensure_running 会在模型加载完成前返回，首次推理请求将因模型未就绪而失败。
    """
    try:
        resp = httpx.get(f"{_base_url()}/health", timeout=2.0)
        if resp.status_code != 200:
            return False
        return resp.json().get("loaded", False) is True
    except Exception:
        return False


def ensure_running() -> str:
    """确保本地 LLM 服务可用，返回 base_url。

    模型不存在时自动下载（首次约 378MB，含进度日志与断点续传）；
    下载或启动失败抛 RuntimeError，由调用方决定错误语义。
    注意：本方法可能长时间阻塞（下载），异步上下文请用 asyncio.to_thread 包装。
    """
    if _healthy():
        return _base_url()
    if _port_in_use():
        # 端口被占用但服务不健康：通常为主进程被强杀（kill -9 / 部署重启）遗留的
        # 假死孤儿（atexit 不会触发）。不健康的服务无保留价值，直接接管重启。
        _kill_port_holder()
        if _port_in_use():
            raise RuntimeError(
                f"端口 {settings.LOCAL_LLM_PORT} 被非本地 LLM 服务占用，请检查 LOCAL_LLM_PORT 配置"
            )

    ensure_model()  # 不存在则自动下载
    ensure_embedding_model()  # 向量模型由同一子进程提供（知识库检索依赖）

    global _PROC, _SHUTDOWN_REGISTERED
    logger.info("拉起本地 LLM 子进程（%s）", _base_url())
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _PROC = subprocess.Popen(
        [sys.executable, "-m", "app.service.ai.local_llm_server"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_set_pdeathsig,  # 父进程被强杀时子进程一并退出，避免孤儿
    )
    if not _SHUTDOWN_REGISTERED:
        atexit.register(_shutdown)
        _SHUTDOWN_REGISTERED = True

    for _ in range(120):  # 最多等 60s（含模型加载约 5-15s）
        if _healthy():
            logger.info("本地 LLM 服务就绪: %s", _base_url())
            return _base_url()
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


def _kill_port_holder() -> None:
    """杀掉占用 LOCAL_LLM_PORT 但不健康的进程（孤儿自愈）"""
    try:
        output = subprocess.run(
            ["ss", "-tlnpH", f"sport = :{settings.LOCAL_LLM_PORT}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        pids = {int(m) for m in re.findall(r"pid=(\d+)", output) if int(m) != os.getpid()}
        for pid in pids:
            logger.warning("终止假死的本地 LLM 孤儿进程 pid=%s", pid)
            os.kill(pid, signal.SIGKILL)
        if pids:
            time.sleep(1)  # 等端口释放
    except Exception as exc:  # noqa: BLE001 清理失败由上层端口复查兜底
        logger.warning("清理端口占用进程失败: %s", exc)


def _shutdown() -> None:
    if _PROC and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _PROC.kill()
