class LoginLogDocument:
    """登录审计日志（MongoDB document）

    collection: login_log
    字段：
        user_id: int | None
        username: str
        ip: str
        location: str
        browser: str
        os: str
        status: int (1:成功;0:失败)
        message: str
        create_time: datetime
    """

    COLLECTION = "login_log"


class AuditLogDocument:
    """业务操作审计日志（MongoDB document，白名单驱动）

    collection: audit_log
    字段：
        operator_id: int
        target_type: str (member/order/dataset/role/user)
        target_id: int | None
        action: str (create/update/delete/level_change/refund)
        module: str (member/order/dataset/role/user)
        before_value: dict | None
        after_value: dict | None
        ip: str
        user_agent: str
        create_time: datetime
    """

    COLLECTION = "audit_log"


class AiApiCallLogDocument:
    """AI 兼容 API 调用审计日志（MongoDB document）

    collection: ai_api_call_log
    字段：
        user_id: int
        key_id: int | None
        key_prefix: str (脱敏前缀 dhak_xxx...，不存完整 Key)
        conversation_id: int | None
        model: str | None
        endpoint: str (chat/completions、messages、models)
        protocol: str (openai/claude)
        is_stream: bool
        input_tokens: int
        output_tokens: int
        credits: float | None
        status_code: int (200/401/403/429/402/5xx)
        duration_ms: int
        client_ip: str
        request_id: str
        error_msg: str | None
        create_time: datetime
    """

    COLLECTION = "ai_api_call_log"
