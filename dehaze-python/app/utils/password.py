"""
密码工具函数

提供异步密码哈希和验证功能，使用线程池避免阻塞事件循环。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import bcrypt

# 密码操作线程池（CPU 密集型操作不应阻塞事件循环）
_password_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pwd-hash")


async def hash_password_async(password: str) -> str:
    """
    异步哈希密码（在线程池中执行）

    bcrypt.gensalt/hashpw 是 CPU 密集型操作，会阻塞事件循环。
    将其移至线程池执行，避免影响其他并发请求。

    Args:
        password: 明文密码

    Returns:
        哈希后的密码
    """
    loop = asyncio.get_running_loop()
    hashed = await loop.run_in_executor(
        _password_executor,
        lambda: bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
    )
    return hashed


async def check_password_async(password: str, hashed: str) -> bool:
    """
    异步验证密码（在线程池中执行）

    bcrypt.checkpw 是 CPU 密集型操作，会阻塞事件循环。
    将其移至线程池执行，避免影响其他并发请求。

    Args:
        password: 明文密码
        hashed: 哈希后的密码

    Returns:
        是否匹配
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _password_executor,
        bcrypt.checkpw,
        password.encode("utf-8"),
        hashed.encode("utf-8"),
    )
