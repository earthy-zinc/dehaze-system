"""AI 对话服务。

内部按职责分四个子包（见改造计划 §6.4）：
- service/：业务编排层（推理用例 + 领域服务）
- builders/：图/工具/上下文构建
- middleware/：推理链横切（hooks/护栏/恢复）
- strategies/：策略/规则/模板
- paradigms/、skills/：推理范式与内置工作流

协议转换/子进程/外部客户端等基础设施实现已下沉 app/infrastructure/
（a2a/、clients/、sandbox/、llm/ 等）。
"""
