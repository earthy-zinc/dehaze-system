"""任务域：任务生命周期 + 状态维护 + 后台执行 + MQ 消费 + 策略分发。

外部引用统一走模块路径，如：
`from app.service.task.task_service import create_task`。
（声明式白名单：不做聚合导入，避免包级循环 import。）
"""
