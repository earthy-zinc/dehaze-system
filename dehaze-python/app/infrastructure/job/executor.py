from __future__ import annotations

import logging
import os
from typing import Optional

from app.config import settings
from pyxxl import ExecutorConfig, PyxxlRunner

logger = logging.getLogger(__name__)

_runner: Optional[PyxxlRunner] = None


def get_xxljob_runner() -> Optional[PyxxlRunner]:
    return _runner


async def init_xxljob() -> Optional[PyxxlRunner]:
    global _runner

    if not settings.XXLJOB_ENABLED:
        logger.info("XXL-Job 未启用，跳过初始化")
        return None

    # 延迟导入 handler 注册（触发装饰器注册）
    from app.infrastructure.job.handlers import xxl_handler

    try:
        config = ExecutorConfig(
            xxl_admin_baseurl=settings.XXLJOB_ADMIN_URL,
            executor_app_name=settings.XXLJOB_EXECUTOR_APP_NAME,
            executor_listen_host=settings.XXLJOB_EXECUTOR_HOST,
            executor_listen_port=settings.XXLJOB_EXECUTOR_PORT,
            access_token=settings.XXLJOB_ACCESS_TOKEN,
            executor_log_path=settings.XXLJOB_EXECUTOR_LOG_PATH,
            log_local_dir=settings.XXLJOB_TASK_LOG_DIR,
        )

        runner = PyxxlRunner(config, handler=xxl_handler)
        runner.run_with_daemon()

        # 记录子进程 PID，方便 start.sh 管理
        if runner.daemon and runner.daemon.pid:
            pid_file = settings.XXLJOB_PID_FILE
            os.makedirs(os.path.dirname(pid_file), exist_ok=True)
            with open(pid_file, "w") as f:
                f.write(str(runner.daemon.pid))
            logger.info(f"XXL-Job 子进程 PID={runner.daemon.pid} 已写入 {pid_file}")

        _runner = runner
        logger.info(
            f"XXL-Job 执行器已启动: "
            f"appName={settings.XXLJOB_EXECUTOR_APP_NAME}, "
            f"port={settings.XXLJOB_EXECUTOR_PORT}, "
            f"admin={settings.XXLJOB_ADMIN_URL}"
        )
        return _runner

    except Exception as e:
        logger.error(f"XXL-Job 执行器初始化失败（服务继续启动）: {e}")
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
            logger.warning(f"XXL-Job 执行器关闭异常: {e}")
        finally:
            # 清理 PID 文件
            pid_file = settings.XXLJOB_PID_FILE
            if os.path.exists(pid_file):
                os.remove(pid_file)
            _runner = None
        logger.info("XXL-Job 执行器已关闭")
