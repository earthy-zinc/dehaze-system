"""内置本地模型（local provider）在编排侧的注册表播种。

推理本身由独立服务 clarity-service/clarity-llm（8992）承担，编排侧只负责：
把 local provider、占位 Key 与三个内置模型（chat / embedding / rerank）幂等写入
sys_ai_* 注册表，使本地模型走与第三方供应商一致的 Key 轮换、健康度与降级链路。
"""
