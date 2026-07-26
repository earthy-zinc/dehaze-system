from enum import Enum


class ResultCode(Enum):
    SUCCESS = ("00000", "一切ok")

    USER_ERROR = ("A0001", "用户端错误")
    REPEAT_SUBMIT_ERROR = ("A0002", "您的请求已提交，请不要重复提交或等待片刻再尝试。")

    USER_LOGIN_ERROR = ("A0200", "用户登录异常")

    USER_NOT_EXIST = ("A0201", "用户不存在")
    USER_ACCOUNT_LOCKED = ("A0202", "用户账户被冻结")
    USER_ACCOUNT_INVALID = ("A0203", "用户账户已作废")

    USERNAME_OR_PASSWORD_ERROR = ("A0210", "用户名或密码错误")
    PASSWORD_ENTER_EXCEED_LIMIT = ("A0211", "用户输入密码次数超限")
    CLIENT_AUTHENTICATION_FAILED = ("A0212", "客户端认证失败")

    VERIFY_CODE_TIMEOUT = ("A0213", "验证码已过期")
    VERIFY_CODE_ERROR = ("A0214", "验证码错误")

    # 按文档要求：A0230 token无效，A0231 token已被禁止
    TOKEN_INVALID = ("A0230", "token无效或已过期")
    TOKEN_ACCESS_FORBIDDEN = ("A0231", "token已被禁止访问")

    AUTHORIZED_ERROR = ("A0300", "访问权限异常")
    ACCESS_UNAUTHORIZED = ("A0301", "访问未授权")
    FORBIDDEN_OPERATION = ("A0302", "演示环境禁止新增、修改和删除数据，请本地部署后测试")
    IP_BLOCKED = ("A0304", "IP 已被临时封禁，请稍后重试")

    PARAM_ERROR = ("A0400", "用户请求参数错误")
    RESOURCE_NOT_FOUND = ("A0401", "请求资源不存在")
    PARAM_IS_NULL = ("A0410", "请求必填参数为空")

    BUSINESS_ERROR = ("A0500", "业务异常")
    DATA_EXISTS = ("A0501", "数据已存在")
    DATA_STATE_NOT_ALLOW = ("A0502", "数据状态不允许")
    OPERATION_NOT_ALLOW = ("A0503", "操作不允许")
    DATA_BIND_EXISTS = ("A0504", "存在关联数据，无法删除")

    USER_UPLOAD_FILE_ERROR = ("A0700", "用户上传文件异常")
    USER_UPLOAD_FILE_TYPE_NOT_MATCH = ("A0701", "文件格式不支持")
    USER_UPLOAD_FILE_SIZE_EXCEEDS = ("A0702", "文件大小超限")
    IMPORT_FILE_EMPTY = ("A0703", "文件内容为空")
    IMPORT_FILE_PARSE_ERROR = ("A0704", "文件解析失败")
    IMPORT_TEMPLATE_MISMATCH = ("A0705", "模板字段不匹配")
    IMPORT_REQUIRED_FIELD_EMPTY = ("A0706", "必填字段为空")
    IMPORT_DATA_VALIDATE_ERROR = ("A0707", "数据校验失败")
    IMPORT_ROWS_EXCEED_LIMIT = ("A0708", "导入数据超出限制")
    EXPORT_ROWS_EXCEED_LIMIT = ("A0709", "导出行数超出限制")
    MODULE_IMPORT_NOT_SUPPORTED = ("A0710", "不支持该模块导入")

    # 系统级错误码 B0xxx
    SYSTEM_EXECUTION_ERROR = ("B0001", "系统执行出错")

    # 算法模块错误码 B02xx
    ALGORITHM_NOT_FOUND = ("B0200", "算法不存在")
    ALGORITHM_NAME_EXISTS = ("B0201", "算法名称已存在")
    ALGORITHM_STATUS_NOT_ALLOWED = ("B0202", "当前状态不允许此操作")
    ALGORITHM_IN_USE = ("B0203", "算法正在使用中，无法删除")
    ALGORITHM_MODEL_CORRUPTED = ("B0204", "模型文件已损坏")
    ALGORITHM_VERSION_EXISTS = ("B0205", "版本号已存在")
    ALGORITHM_ROLLBACK_NOT_ALLOWED = ("B0206", "不允许回滚到该版本")
    ALGORITHM_AUDIT_PERMISSION_DENIED = ("B0207", "无审核权限")
    ALGORITHM_AUDIT_REMARK_REQUIRED = ("B0208", "驳回时必须填写原因")
    ALGORITHM_IMPORT_FORMAT_ERROR = ("B0209", "导入包格式错误")
    PREDICTION_TASK_NOT_FOUND = ("B0210", "预测任务不存在")
    PREDICTION_TASK_EXPIRED = ("B0211", "预测任务结果已过期")
    PREDICTION_IMAGE_FORMAT_UNSUPPORTED = ("B0212", "图片格式不支持")
    PREDICTION_GT_MISSING = ("B0213", "缺少清晰图（Ground Truth）")
    EVALUATION_TASK_NOT_FOUND = ("B0220", "评估任务不存在")
    EVALUATION_TASK_EXPIRED = ("B0221", "评估任务结果已过期")

    # 任务模块错误码 B03xx
    TASK_NOT_FOUND = ("B0301", "任务不存在")
    TASK_UNAUTHORIZED = ("B0302", "无权操作该任务")
    TASK_STATUS_INVALID = ("B0303", "任务状态不允许此操作")
    TASK_TYPE_UNSUPPORTED = ("B0304", "不支持的任务类型")
    TASK_PARAM_ERROR = ("B0305", "任务参数错误")
    TASK_CANCELLED = ("B0306", "任务已被取消")
    TASK_CONCURRENT_LIMIT = ("B0307", "同类型任务并发数已达上限")
    TASK_CONCURRENT_EXCEED_LIMIT = ("B0308", "导入导出任务并发超限")

    # 文件模块错误码 B04xx
    FILE_NOT_FOUND = ("B0401", "文件不存在")
    FILE_TOO_LARGE = ("B0402", "文件大小超过限制")
    FILE_TYPE_NOT_SUPPORTED = ("B0403", "不支持的文件类型")
    FILE_MD5_INVALID = ("B0404", "MD5格式无效")
    FILE_STORAGE_ERROR = ("B0405", "文件存储失败")
    FILE_CORRUPTED = ("B0406", "文件已损坏")

    CALL_THIRD_PARTY_SERVICE_ERROR = ("C0001", "调用第三方服务出错")
    MIDDLEWARE_SERVICE_ERROR = ("C0100", "中间件服务出错")
    INTERFACE_NOT_EXIST = ("C0113", "接口不存在")

    MESSAGE_SERVICE_ERROR = ("C0120", "消息服务出错")
    MESSAGE_DELIVERY_ERROR = ("C0121", "消息投递出错")
    MESSAGE_CONSUMPTION_ERROR = ("C0122", "消息消费出错")
    MESSAGE_SUBSCRIPTION_ERROR = ("C0123", "消息订阅出错")
    MESSAGE_GROUP_NOT_FOUND = ("C0124", "消息分组未查到")

    DATABASE_ERROR = ("C0300", "数据库服务出错")
    DATABASE_TABLE_NOT_EXIST = ("C0311", "表不存在")
    DATABASE_COLUMN_NOT_EXIST = ("C0312", "列不存在")
    DATABASE_DUPLICATE_COLUMN_NAME = ("C0321", "多表关联中存在多个相同名称的列")
    DATABASE_DEADLOCK = ("C0331", "数据库死锁")
    DATABASE_PRIMARY_KEY_CONFLICT = ("C0341", "主键冲突")

    def __init__(self, code: str, msg: str) -> None:
        self.code = code
        self.msg = msg

    def __str__(self):
        return f'{{"code":"{self.code}", "msg":"{self.msg}"}}'
