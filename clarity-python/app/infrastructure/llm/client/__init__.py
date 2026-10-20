"""LLM 协议适配：统一接口与工厂、OpenAI 兼容/Anthropic 供应商实现、langchain 适配。

model_client 工厂在函数内延迟 import 各供应商实现（model_client ↔
anthropic_client/openai_compat_client 存在模块级双向依赖），故不做聚合导入。
"""
