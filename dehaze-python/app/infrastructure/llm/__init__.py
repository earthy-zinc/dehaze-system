"""LLM 基础设施，按职责分三组：

- client/：协议适配（统一接口工厂 + openai_compat/anthropic 供应商实现 + langchain 适配）
- call/：韧性调用编排（候选路由序列 + 逐 Key 重试，统一入口 llm_client）
- local/：本地模型（子进程生命周期、下载、推理服务、播种）

跨模态供应商能力（模型路由/Key 轮换/健康熔断）在 infrastructure/provider/，
A2A 协议层在 infrastructure/a2a/。
"""
