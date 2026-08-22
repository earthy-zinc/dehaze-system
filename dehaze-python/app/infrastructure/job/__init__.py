"""
XXL-Job 定时任务模块

提供 XXL-Job 执行器集成、任务 handler 注册，与 Java/Go 端共享调度平台。
基于 pyxxl（Python XXL-Job 执行器），支持 asyncio 原生异步任务。

显式导入（不做包级 re-export）：
    from app.infrastructure.job.executor import init_xxljob, close_xxljob
"""
