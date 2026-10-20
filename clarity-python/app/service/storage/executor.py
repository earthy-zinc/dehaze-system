"""
存储操作同步执行线程池

各存储后端（MinIO SDK / 本地文件系统 / nginx HTTP 请求）均为同步操作，
业务侧统一经本线程池执行以避免阻塞事件循环。
"""

from concurrent.futures import ThreadPoolExecutor

# 存储操作线程池（消费方 run_in_executor 使用）
storage_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="storage-ops")
