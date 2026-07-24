"""
路由注册模块

统一注册所有 API 路由
"""

from fastapi import FastAPI


def init_routes(app: FastAPI, prometheus_enabled: bool = False):
    """
    注册所有路由

    Args:
        app: FastAPI 应用实例
        prometheus_enabled: 是否启用 Prometheus 指标端点
    """
    # 健康检查路由
    from app.router.health import ready_router, router as health_router
    app.include_router(health_router)
    app.include_router(ready_router)

    # Prometheus 指标端点（如果启用）
    if prometheus_enabled:
        from app.router.metrics import router as metrics_router
        app.include_router(metrics_router)

    # 认证路由
    from app.router.auth import router as auth_router
    app.include_router(auth_router)

    # API密钥路由
    from app.router.api_key import router as api_key_router
    app.include_router(api_key_router)

    # 算法路由
    from app.router.algorithm import router as algorithm_router
    app.include_router(algorithm_router)

    # 字典路由
    from app.router.dict import router as dict_router
    app.include_router(dict_router)

    # 部门路由
    from app.router.dept import router as dept_router
    app.include_router(dept_router)

    # 文件路由
    from app.router.file import router as file_router
    app.include_router(file_router)

    # 用户路由
    from app.router.user import router as user_router
    app.include_router(user_router)

    # 数据集路由
    from app.router.dataset import router as dataset_router
    app.include_router(dataset_router)

    # 数据项路由
    from app.router.dataset_item import router as dataset_item_router
    app.include_router(dataset_item_router)

    # 图片文件路由
    from app.router.item_file import router as item_file_router
    app.include_router(item_file_router)

    # 菜单路由
    from app.router.menu import router as menu_router
    app.include_router(menu_router)

    # 角色路由
    from app.router.role import router as role_router
    app.include_router(role_router)

    # 导出任务路由
    from app.router.task import router as task_router
    app.include_router(task_router)

    # 预测路由（去雾处理核心入口）
    from app.router.prediction import router as prediction_router
    app.include_router(prediction_router)

    # 评估路由（效果评估）
    from app.router.evaluation import router as evaluation_router
    app.include_router(evaluation_router)

    # 图像输入历史记录路由
    from app.router.image_input import router as image_input_router
    app.include_router(image_input_router)

    # 算法选择路由（智能推荐/收藏/对比）
    from app.router.algorithm_select import router as algorithm_select_router
    app.include_router(algorithm_select_router)
