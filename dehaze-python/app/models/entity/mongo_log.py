from datetime import datetime


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

    COLLECTION = 'login_log'


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

    COLLECTION = 'audit_log'
