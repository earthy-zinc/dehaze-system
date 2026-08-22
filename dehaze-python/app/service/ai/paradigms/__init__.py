"""推理范式实现：Plan-and-Execute / Reflexion

设计文档 §4（多步推理范式）：
- plan_execute：Planner 分解任务 → 依赖拓扑分批并行执行 → Replanner 修订 → 计划干预
- reflexion：evaluator 自评 → self_reflection 根因分析 → 反思记忆 → 迭代改进
- 混合架构：Planner 为子任务标注 paradigm，executor 按标注调度
"""
