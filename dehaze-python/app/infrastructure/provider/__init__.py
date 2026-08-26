"""跨模态供应商能力：模型路由注册表、Key 轮换/冷却/日额度、健康/熔断。

LLM / Embedding / TTS 共用的供应商选择能力，上提自 infrastructure/llm/，
消除 embedding 反向引用 llm 子包的跨模态错位。
"""
