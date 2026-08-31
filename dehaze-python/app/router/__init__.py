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
    from app.router.health import ready_router
    from app.router.health import router as health_router

    app.include_router(health_router)
    app.include_router(ready_router)

    # Prometheus 指标端点（如果启用）
    if prometheus_enabled:
        from app.router.metrics import router as metrics_router

        app.include_router(metrics_router)

    from app.router.auth import router as auth_router

    app.include_router(auth_router)

    from app.router.api_key import router as api_key_router

    app.include_router(api_key_router)

    # 通用导入导出路由（必须先于各模块 CRUD 注册，否则 /{module}/template、/_export、/_import
    # 会被 CRUD 的 /{module}/{id} 路径参数吞掉，如 /algorithms/template 被当作 algorithm_id 解析）
    from app.router.import_export import router as import_export_router

    app.include_router(import_export_router)

    from app.router.algorithm import router as algorithm_router

    app.include_router(algorithm_router)

    from app.router.dict import router as dict_router

    app.include_router(dict_router)

    from app.router.dept import router as dept_router

    app.include_router(dept_router)

    from app.router.file import router as file_router

    app.include_router(file_router)

    from app.router.user import router as user_router

    app.include_router(user_router)

    from app.router.dataset import router as dataset_router

    app.include_router(dataset_router)

    from app.router.dataset_item import router as dataset_item_router

    app.include_router(dataset_item_router)

    from app.router.item_file import router as item_file_router

    app.include_router(item_file_router)

    from app.router.menu import router as menu_router

    app.include_router(menu_router)

    from app.router.role import router as role_router

    app.include_router(role_router)

    from app.router.task import router as task_router

    app.include_router(task_router)

    # 预测路由（去雾处理核心入口）
    from app.router.prediction import router as prediction_router

    app.include_router(prediction_router)

    from app.router.evaluation import router as evaluation_router

    app.include_router(evaluation_router)

    from app.router.image_input import router as image_input_router

    app.include_router(image_input_router)

    # 算法选择路由（智能推荐/收藏/对比）
    from app.router.algorithm_select import router as algorithm_select_router

    app.include_router(algorithm_select_router)

    from app.router.announcement import router as announcement_router
    from app.router.message import router as message_router
    from app.router.message_template import router as message_template_router
    from app.router.notification_setting import router as notification_setting_router

    app.include_router(message_router)
    app.include_router(announcement_router)
    app.include_router(message_template_router)
    app.include_router(notification_setting_router)

    from app.router.member import router as member_router

    app.include_router(member_router)

    # 套餐管理模块路由（含优惠券）
    from app.router.package import router as package_router

    app.include_router(package_router)

    # 促销活动路由（套餐管理-营销侧）
    from app.router.promotion import router as promotion_router

    app.include_router(promotion_router)

    from app.router.order import router as order_router

    app.include_router(order_router)

    # 支付回调路由（无需认证，由支付平台调用）
    from app.router.payment import router as payment_router

    app.include_router(payment_router)

    from app.router.feedback import router as feedback_router

    app.include_router(feedback_router)

    from app.router.recommendation import router as recommendation_router

    app.include_router(recommendation_router)

    from app.router.favorite import router as favorite_router

    app.include_router(favorite_router)

    from app.router.compare import router as compare_router

    app.include_router(compare_router)

    # 参数预设路由（去雾处理-参数预设）
    from app.router.preset import router as preset_router

    app.include_router(preset_router)

    from app.router.client_log import router as client_log_router

    app.include_router(client_log_router)

    from app.router.ai_agent import router as ai_agent_router
    from app.router.ai_artifact import router as ai_artifact_router
    from app.router.ai_billing import router as ai_billing_router
    from app.router.ai_conversation import router as ai_conversation_router
    from app.router.ai_feedback import router as ai_feedback_router
    from app.router.ai_memory import router as ai_memory_router
    from app.router.ai_model import router as ai_model_router
    from app.router.ai_provider import router as ai_provider_router
    from app.router.ai_usage_stats import router as ai_usage_stats_router

    app.include_router(ai_model_router)
    app.include_router(ai_conversation_router)
    app.include_router(ai_artifact_router)
    app.include_router(ai_memory_router)
    app.include_router(ai_feedback_router)
    app.include_router(ai_provider_router)
    app.include_router(ai_usage_stats_router)
    app.include_router(ai_billing_router)
    app.include_router(ai_agent_router)

    # AI 定时调度路由（F-M08-009）
    from app.router.ai_schedule import router as ai_schedule_router

    app.include_router(ai_schedule_router)

    # Skills 管理路由（F-M08-006）
    from app.router.ai_skill import router as ai_skill_router

    app.include_router(ai_skill_router)

    # 外部 MCP Server 管理路由（F-M08-006 §2.6.13）
    from app.router.ai_mcp import router as ai_mcp_router

    app.include_router(ai_mcp_router)

    from app.router.ai_agent_eval import center_router as ai_eval_center_router
    from app.router.ai_agent_eval import router as ai_agent_eval_router

    app.include_router(ai_agent_eval_router)
    app.include_router(ai_eval_center_router)

    # A2A 协议路由（服务端对外暴露：Agent 挂载路径 + 全局标准入口）
    from app.router.a2a import global_router as a2a_global_router
    from app.router.a2a import router as a2a_router

    app.include_router(a2a_router)
    app.include_router(a2a_global_router)

    # A2A 端点管理路由（外部端点注册）
    from app.router.ai_agent_endpoint import router as ai_agent_endpoint_router

    app.include_router(ai_agent_endpoint_router)

    # 第三方兼容 API 路由（OpenAI/Claude 协议适配）
    from app.router.compatible_claude import router as compatible_claude_router
    from app.router.compatible_openai import router as compatible_openai_router

    app.include_router(compatible_openai_router)
    app.include_router(compatible_claude_router)

    # AI 兼容调用审计查询（内部 API，F-M08-010 接入治理）
    from app.router.ai_compat_call import router as ai_compat_call_router

    app.include_router(ai_compat_call_router)

    # AI 可观测性查询路由（F-M08-013）
    from app.router.ai_observability import router as ai_observability_router

    app.include_router(ai_observability_router)

    from app.router.kb import router as kb_router

    app.include_router(kb_router)

    # 语音交互模块路由（ASR/TTS/热词/服务状态）
    from app.router.voice import router as voice_router

    app.include_router(voice_router)

    # 语音引擎注册表管理路由（Provider/Key/Model，voice:engine:manage）
    from app.router.voice_admin import router as voice_admin_router

    app.include_router(voice_admin_router)

    # 管理端缓存统一失效入口（ROOT/ADMIN）
    from app.router.cache import router as cache_router

    app.include_router(cache_router)
