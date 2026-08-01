from __future__ import annotations

import logging
import os
from typing import Optional

from app.config import settings
from pyxxl import ExecutorConfig, PyxxlRunner

logger = logging.getLogger(__name__)

_runner: Optional[PyxxlRunner] = None

# pyxxl 内部 logger 名称（定义于 pyxxl.log）
_PYXXL_LOGGERS = ("pyxxl", "pyxxl.setting", "pyxxl.executor", "pyxxl.xxl_client")


def get_xxljob_runner() -> Optional[PyxxlRunner]:
    return _runner


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


async def init_xxljob() -> Optional[PyxxlRunner]:
    global _runner

    if not settings.XXLJOB_ENABLED:
        logger.info("XXL-Job 未启用，跳过初始化")
        return None

    # 延迟导入 handler 注册（触发装饰器注册）
    from app.infrastructure.job.handlers import xxl_handler

    try:
        executor_logger = _attach_pyxxl_to_root()
        config = ExecutorConfig(
            xxl_admin_baseurl=settings.XXLJOB_ADMIN_URL,
            executor_app_name=settings.XXLJOB_EXECUTOR_APP_NAME,
            executor_listen_host=settings.XXLJOB_EXECUTOR_HOST,
            executor_listen_port=settings.XXLJOB_EXECUTOR_PORT,
            access_token=settings.XXLJOB_ACCESS_TOKEN,
            log_local_dir=settings.XXLJOB_TASK_LOG_DIR,
            executor_logger=executor_logger,
        )

        runner = PyxxlRunner(config, handler=xxl_handler)
        runner.run_with_daemon()

        # 记录子进程 PID，方便 start.sh 管理
        if runner.daemon and runner.daemon.pid:
            pid_file = settings.XXLJOB_PID_FILE
            os.makedirs(os.path.dirname(pid_file), exist_ok=True)
            with open(pid_file, "w") as f:
                f.write(str(runner.daemon.pid))
            logger.info("XXL-Job 子进程 PID=%s 已写入 %s", runner.daemon.pid, pid_file)

        _runner = runner
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
    global _runner

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
            if os.path.exists(pid_file):
                os.remove(pid_file)
            _runner = None
        logger.info("XXL-Job 执行器已关闭")
