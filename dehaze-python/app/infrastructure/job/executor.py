from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pyxxl import ExecutorConfig, PyxxlRunner

from app.config import settings

logger = logging.getLogger(__name__)

_runner: PyxxlRunner | None = None
_watch_task: asyncio.Task | None = None

# daemon 启动宽限期：超过此时间仍存活才认为启动成功
_DAEMON_STARTUP_GRACE_SECONDS = 2.0
# daemon 存活巡检间隔
_DAEMON_WATCH_INTERVAL_SECONDS = 30.0

# pyxxl 内部 logger 名称（定义于 pyxxl.log）
_PYXXL_LOGGERS = ("pyxxl", "pyxxl.setting", "pyxxl.executor", "pyxxl.xxl_client")


def _attach_pyxxl_to_root() -> logging.Logger:
    """将 pyxxl 执行器日志接入项目统一日志，不再产生独立的 pyxxl.log。

    pyxxl 的 setup_logging() 在目标 logger 已有 handler 时会跳过自建
    RotatingFileHandler（即不写 pyxxl.log）。因此预先给 pyxxl.* 各 logger
    挂 NullHandler 占位并设 propagate=True，使日志传播到 root logger，
    由 app.infrastructure.logging 的 DailyDirectoryFileHandler 落盘到
    logs/{yyyy-MM-dd}/info.log|error.log，与 Go 端一致。

    依赖 multiprocessing fork 模型（Linux 默认）：daemon 子进程继承父进程
    的 logger 配置，子进程的 _setup_logging 同样跳过自建 handler。

    Returns:
        pyxxl.executor logger，传给 ExecutorConfig(executor_logger=...) 以
        跳过 pyxxl 对该 logger 的二次 setup_logging。
    """
    for name in _PYXXL_LOGGERS:
        lg = logging.getLogger(name)
        if not lg.handlers:
            lg.addHandler(logging.NullHandler())
        lg.propagate = True
        lg.setLevel(logging.INFO)
    return logging.getLogger("pyxxl.executor")


def _poll_daemon_exit(runner: PyxxlRunner) -> int | None:
    """轮询 daemon 子进程，返回 None 表示仍在运行。

    multiprocessing 子进程只有在 join()/is_alive()/exitcode 被读取时才会被父进程回收，
    因此必须显式轮询；否则子进程退出后会一直停留在僵尸态。
    """
    daemon = runner.daemon
    if daemon is None or daemon.is_alive():
        return None
    return daemon.exitcode


async def _watch_daemon() -> None:
    """巡检 daemon 存活状态：回收已退出的子进程并暴露故障。

    不做自动重启：daemon 退出后 XXL-Job 定时任务不再被调度，需要人工介入。
    """
    global _runner

    while True:
        await asyncio.sleep(_DAEMON_WATCH_INTERVAL_SECONDS)
        runner = _runner
        if runner is None:
            return
        exitcode = _poll_daemon_exit(runner)
        if exitcode is not None:
            logger.error(
                "XXL-Job 执行器 daemon 非预期退出: pid=%s exitcode=%s，定时任务将不再被调度",
                runner.daemon.pid if runner.daemon else None,
                exitcode,
            )
            _runner = None
            return


def _start_daemon_watch() -> None:
    """启动 daemon 存活巡检任务（重复调用只保留一个）。"""
    global _watch_task
    if _watch_task is not None and not _watch_task.done():
        return
    _watch_task = asyncio.create_task(_watch_daemon())


async def init_xxljob() -> PyxxlRunner | None:
    global _runner

    if not settings.XXLJOB_ENABLED:
        logger.info("XXL-Job 未启用，跳过初始化")
        return None

    # 延迟导入 handler 注册（触发装饰器注册）
    from app.infrastructure.job.handlers import xxl_handler

    try:
        executor_logger = _attach_pyxxl_to_root()
        executor_ip = settings.XXLJOB_EXECUTOR_IP
        config = ExecutorConfig(
            xxl_admin_baseurl=settings.XXLJOB_ADMIN_URL,
            executor_app_name=settings.XXLJOB_EXECUTOR_APP_NAME,
            executor_listen_port=settings.XXLJOB_EXECUTOR_PORT,
            access_token=settings.XXLJOB_ACCESS_TOKEN,
            log_local_dir=settings.XXLJOB_TASK_LOG_DIR,
            executor_logger=executor_logger,
            # 显式声明注册地址（admin 在容器时是 host.docker.internal 这类域名）必须同时绑所有网卡，
            # 否则 pyxxl 会用注册地址推导监听地址，绑定失败；未声明注册地址时交给 pyxxl 以首网卡 IP 绑定并注册
            executor_listen_host="0.0.0.0" if executor_ip else "",
            executor_url=f"http://{executor_ip}:{settings.XXLJOB_EXECUTOR_PORT}" if executor_ip else "",
        )

        runner = PyxxlRunner(config, handler=xxl_handler)
        runner.run_with_daemon()

        daemon = runner.daemon
        if daemon is None:
            logger.error("XXL-Job 执行器 daemon 未创建")
            _runner = None
            return None

        # daemon 在启动阶段退出（典型原因：端口被上一个残留 daemon 占用）时必须在此回收，
        # 否则该子进程会永久停留在僵尸态，且 XXL-Job 静默不可用。
        await asyncio.sleep(_DAEMON_STARTUP_GRACE_SECONDS)
        exitcode = _poll_daemon_exit(runner)
        if exitcode is not None:
            logger.error(
                "XXL-Job 执行器 daemon 启动失败: pid=%s exitcode=%s, 端口 %s 可能已被占用，"
                "请检查 %s 记录的残留进程",
                daemon.pid,
                exitcode,
                settings.XXLJOB_EXECUTOR_PORT,
                settings.XXLJOB_PID_FILE,
            )
            _runner = None
            return None

        # 记录子进程 PID，方便 start.sh 管理
        pid_file = settings.XXLJOB_PID_FILE
        Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(pid_file).open("w") as f:
            f.write(str(daemon.pid))
        logger.info("XXL-Job 子进程 PID=%s 已写入 %s", daemon.pid, pid_file)

        _runner = runner
        _start_daemon_watch()
        logger.info(
            "XXL-Job 执行器已启动: appName=%s, port=%s, admin=%s",
            settings.XXLJOB_EXECUTOR_APP_NAME,
            settings.XXLJOB_EXECUTOR_PORT,
            settings.XXLJOB_ADMIN_URL,
        )
        return _runner

    except Exception as e:
        logger.error("XXL-Job 执行器初始化失败（服务继续启动）: %s", e)
        _runner = None
        return None


async def close_xxljob() -> None:
    global _runner, _watch_task

    if _watch_task is not None:
        _watch_task.cancel()
        _watch_task = None

    if _runner is not None:
        try:
            # PyxxlRunner 使用 daemon (multiprocessing.Process) 运行
            # 通过终止 daemon 进程来关闭执行器
            if _runner.daemon is not None:
                _runner.daemon.terminate()
                _runner.daemon.join(timeout=5)
        except Exception as e:
            logger.warning("XXL-Job 执行器关闭异常: %s", e)
        finally:
            # 清理 PID 文件
            pid_file = settings.XXLJOB_PID_FILE
            if Path(pid_file).exists():
                Path(pid_file).unlink()
            _runner = None
        logger.info("XXL-Job 执行器已关闭")
