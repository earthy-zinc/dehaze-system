package common

import (
	"fmt"
	"strings"
)

// ResultCode 响应码定义
// 仿照 Java ResultCode 的实现方式
type ResultCode struct {
	Code string
	Msg  string
}

// String 实现 Stringer 接口，返回 JSON 格式字符串
func (rc *ResultCode) String() string {
	return fmt.Sprintf(`{"code":"%s", "msg":"%s"}`, rc.Code, rc.Msg)
}

// GetCode 获取错误码
func (rc *ResultCode) GetCode() string {
	return rc.Code
}

// GetMsg 获取错误消息
func (rc *ResultCode) GetMsg() string {
	return rc.Msg
}

// ========== 所有错误码定义 ==========

var (
	// ========== 成功状态码 ==========
	SUCCESS = &ResultCode{"00000", "一切ok"}

	// ========== A 类：用户端错误 ==========

	// A00xx: 通用错误
	USER_ERROR          = &ResultCode{"A0001", "用户端错误"}
	REPEAT_SUBMIT_ERROR = &ResultCode{"A0002", "您的请求已提交，请不要重复提交或等待片刻再尝试。"}

	// A02xx: 认证相关
	USER_LOGIN_ERROR            = &ResultCode{"A0200", "用户登录异常"}
	USER_NOT_EXIST              = &ResultCode{"A0201", "用户不存在"}
	USER_ACCOUNT_LOCKED         = &ResultCode{"A0202", "用户账户被冻结"}
	USER_ACCOUNT_INVALID        = &ResultCode{"A0203", "用户账户已作废"}
	USERNAME_OR_PASSWORD_ERROR  = &ResultCode{"A0210", "用户名或密码错误"}
	PASSWORD_ENTER_EXCEED_LIMIT = &ResultCode{"A0211", "用户输入密码次数超限"}
	CLIENT_AUTH_FAILED          = &ResultCode{"A0212", "客户端认证失败"}
	VERIFY_CODE_TIMEOUT         = &ResultCode{"A0213", "验证码已过期"}
	VERIFY_CODE_ERROR           = &ResultCode{"A0214", "验证码错误"}
	TOKEN_INVALID               = &ResultCode{"A0230", "token无效或已过期"}
	TOKEN_ACCESS_FORBIDDEN      = &ResultCode{"A0231", "token已被禁止访问"}

	// A03xx: 权限相关
	AUTHORIZED_ERROR    = &ResultCode{"A0300", "访问权限异常"}
	ACCESS_UNAUTHORIZED = &ResultCode{"A0301", "访问未授权"}
	FORBIDDEN_OPERATION = &ResultCode{"A0302", "演示环境禁止新增、修改和删除数据，请本地部署后测试"}

	// A04xx: 参数相关
	PARAM_ERROR        = &ResultCode{"A0400", "用户请求参数错误"}
	RESOURCE_NOT_FOUND = &ResultCode{"A0401", "请求资源不存在"}
	PARAM_IS_NULL      = &ResultCode{"A0410", "请求必填参数为空"}

	// A05xx: 业务规则
	BUSINESS_ERROR       = &ResultCode{"A0500", "业务异常"}
	DATA_EXISTS          = &ResultCode{"A0501", "数据已存在"}
	DATA_STATE_NOT_ALLOW = &ResultCode{"A0502", "数据状态不允许"}
	OPERATION_NOT_ALLOW  = &ResultCode{"A0503", "操作不允许"}
	DATA_BIND_EXISTS     = &ResultCode{"A0504", "存在关联数据，无法删除"}

	// 会员模块业务错误码 A051x
	MEMBER_NOT_FOUND      = &ResultCode{"A0510", "会员不存在"}
	MEMBER_FROZEN         = &ResultCode{"A0511", "会员已冻结"}
	SIGN_IN_ALREADY       = &ResultCode{"A0512", "今日已签到"}
	GROWTH_INSUFFICIENT   = &ResultCode{"A0513", "成长值不足"}
	BENEFIT_CONFIG_INVALID = &ResultCode{"A0514", "权益配置无效"}
	QUOTA_EXCEEDED        = &ResultCode{"A0515", "配额已用尽"}

	// 套餐模块业务错误码 A052x
	PACKAGE_NOT_FOUND    = &ResultCode{"A0520", "套餐不存在"}
	PACKAGE_OFF_SHELF    = &ResultCode{"A0521", "套餐已下架"}
	PACKAGE_HAS_ORDERS   = &ResultCode{"A0522", "套餐下已有关联订单，无法删除"}
	COUPON_NOT_FOUND     = &ResultCode{"A0523", "优惠券不存在"}
	COUPON_EXPIRED       = &ResultCode{"A0524", "优惠券已过期"}
	COUPON_ALREADY_USED  = &ResultCode{"A0525", "优惠券已使用"}
	COUPON_STOCK_EMPTY   = &ResultCode{"A0526", "优惠券已领完"}
	COUPON_NOT_APPLICABLE = &ResultCode{"A0527", "优惠券不适用于该套餐"}
	COUPON_LIMIT_EXCEEDED = &ResultCode{"A0528", "超过每人限领数量"}
	COUPON_STATUS_INVALID = &ResultCode{"A0529", "优惠券状态无效"}
	COUPON_LOCK_FAILED    = &ResultCode{"A052A", "优惠券锁定失败"}

	// 订单模块业务错误码 A053x
	ORDER_NOT_FOUND       = &ResultCode{"A0530", "订单不存在"}
	ORDER_STATUS_INVALID  = &ResultCode{"A0531", "订单状态不允许此操作"}
	ORDER_EXPIRED         = &ResultCode{"A0532", "订单已超时"}
	ORDER_ALREADY_PAID    = &ResultCode{"A0533", "订单已支付"}
	REFUND_TIME_EXCEEDED  = &ResultCode{"A0534", "超过退款时限"}
	REFUND_USAGE_EXCEEDED = &ResultCode{"A0535", "权益使用超限"}
	REFUND_NOT_SUPPORTED  = &ResultCode{"A0536", "该套餐不支持退款"}
	REFUND_NOT_FOUND      = &ResultCode{"A0537", "退款记录不存在"}
	PAYMENT_AMOUNT_MISMATCH = &ResultCode{"A0538", "支付金额与订单金额不一致"}
	DUPLICATE_ORDER       = &ResultCode{"A0539", "短时间内重复下单"}
	REFUND_ALREADY_EXISTS = &ResultCode{"A053A", "该订单已存在退款申请"}

	// 反馈评价模块业务错误码 A054x
	RATING_ALREADY_EXISTS    = &ResultCode{"A0540", "该处理记录已评价"}
	RATING_NOT_FOUND         = &ResultCode{"A0541", "评价不存在"}
	RATING_EXPIRED           = &ResultCode{"A0542", "已超过评价时限"}
	FEEDBACK_NOT_FOUND       = &ResultCode{"A0543", "反馈不存在"}
	FEEDBACK_CLOSED         = &ResultCode{"A0544", "反馈已关闭"}
	FEEDBACK_LIMIT_EXCEEDED  = &ResultCode{"A0545", "今日反馈次数已达上限"}
	PREDICTION_LOG_NOT_FOUND = &ResultCode{"A0546", "处理记录不存在"}

	// A06xx: 操作相关
	OPERATION_FAILED    = &ResultCode{"A0600", "操作失败"}
	OPERATION_COMPLETED = &ResultCode{"A0601", "操作已完成"}

	// A07xx: 文件上传与导入导出
	USER_UPLOAD_FILE_ERROR          = &ResultCode{"A0700", "用户上传文件异常"}
	USER_UPLOAD_FILE_TYPE_NOT_MATCH = &ResultCode{"A0701", "用户上传文件类型不匹配"}
	USER_UPLOAD_FILE_SIZE_EXCEEDS   = &ResultCode{"A0702", "用户上传文件太大"}
	USER_UPLOAD_IMAGE_SIZE_EXCEEDS  = &ResultCode{"A0703", "用户上传图片太大"}
	IMPORT_FILE_EMPTY               = &ResultCode{"A0704", "上传文件为空或无数据行"}
	IMPORT_FILE_PARSE_ERROR         = &ResultCode{"A0705", "文件解析失败"}
	IMPORT_TEMPLATE_MISMATCH        = &ResultCode{"A0706", "导入文件表头与模板不一致"}
	IMPORT_REQUIRED_FIELD_MISSING   = &ResultCode{"A0707", "必填字段为空"}
	IMPORT_DATA_VALIDATE_ERROR      = &ResultCode{"A0708", "数据校验失败"}
	IMPORT_ROWS_EXCEED_LIMIT        = &ResultCode{"A0709", "导入数据超出限制"}
	EXPORT_ROWS_EXCEED_LIMIT        = &ResultCode{"A0710", "导出行数超出限制"}
	MODULE_IMPORT_NOT_SUPPORTED     = &ResultCode{"A0711", "不支持该模块导入"}
	MODULE_EXPORT_NOT_SUPPORTED     = &ResultCode{"A0712", "不支持该模块导出"}

	// ========== B 类：系统端错误 ==========

	// B00xx: 通用系统错误
	SYSTEM_EXECUTION_ERROR = &ResultCode{"B0001", "系统执行出错"}

	// B01xx: 超时相关
	SYSTEM_EXECUTION_TIMEOUT        = &ResultCode{"B0100", "系统执行超时"}
	SYSTEM_ORDER_PROCESSING_TIMEOUT = &ResultCode{"B0101", "系统订单处理超时"}

	// B02xx: 容灾与限流
	SYSTEM_DISASTER_RECOVERY_TRIGGER = &ResultCode{"B0200", "系统容灾功能被触发"}
	FLOW_LIMIT                       = &ResultCode{"B0210", "系统并发限流"}
	RATE_LIMIT                       = &ResultCode{"B0211", "系统速率限流"}
	DEGRADATION                      = &ResultCode{"B0220", "系统功能降级"}

	// B03xx: 资源相关
	SYSTEM_RESOURCE_ERROR      = &ResultCode{"B0300", "系统资源异常"}
	SYSTEM_RESOURCE_EXHAUSTION = &ResultCode{"B0310", "系统资源耗尽"}
	SYSTEM_RESOURCE_ACCESS_ERR = &ResultCode{"B0320", "系统资源访问异常"}
	SYSTEM_READ_DISK_FILE_ERR  = &ResultCode{"B0321", "系统读取磁盘文件失败"}
	TASK_CONCURRENT_EXCEED     = &ResultCode{"B0308", "导入导出任务并发数超限"}

	// ========== C 类：第三方服务错误 ==========

	// C00xx: 通用第三方错误
	CALL_THIRD_PARTY_SERVICE_ERROR = &ResultCode{"C0001", "调用第三方服务出错"}

	// C01xx: 中间件服务
	MIDDLEWARE_SERVICE_ERROR = &ResultCode{"C0100", "中间件服务出错"}
	INTERFACE_NOT_EXIST      = &ResultCode{"C0113", "接口不存在"}

	// C02xx: 缓存服务
	CACHE_SERVICE_ERROR = &ResultCode{"C0200", "缓存服务出错"}
	CACHE_MISS          = &ResultCode{"C0201", "缓存未命中"}
	CACHE_WRITE_FAILED  = &ResultCode{"C0202", "缓存写入失败"}

	// C03xx: 数据库服务
	DATABASE_ERROR                = &ResultCode{"C0300", "数据库服务出错"}
	DATABASE_TABLE_NOT_EXIST      = &ResultCode{"C0311", "表不存在"}
	DATABASE_COLUMN_NOT_EXIST     = &ResultCode{"C0312", "列不存在"}
	DATABASE_DUPLICATE_COLUMN     = &ResultCode{"C0321", "多表关联中存在多个相同名称的列"}
	DATABASE_DEADLOCK             = &ResultCode{"C0331", "数据库死锁"}
	DATABASE_PRIMARY_KEY_CONFLICT = &ResultCode{"C0341", "主键冲突"}

	// C04xx: 对象存储
	OBJECT_STORAGE_ERROR = &ResultCode{"C0400", "对象存储服务出错"}
	FILE_UPLOAD_FAILED   = &ResultCode{"C0401", "文件上传失败"}
	FILE_DOWNLOAD_FAILED = &ResultCode{"C0402", "文件下载失败"}

	// C12xx: 消息服务
	MESSAGE_SERVICE_ERROR      = &ResultCode{"C0120", "消息服务出错"}
	MESSAGE_DELIVERY_ERROR     = &ResultCode{"C0121", "消息投递出错"}
	MESSAGE_CONSUMPTION_ERROR  = &ResultCode{"C0122", "消息消费出错"}
	MESSAGE_SUBSCRIPTION_ERROR = &ResultCode{"C0123", "消息订阅出错"}
	MESSAGE_GROUP_NOT_FOUND    = &ResultCode{"C0124", "消息分组未查到"}
)

// allResultCodes 所有错误码的映射表
var allResultCodes = map[string]*ResultCode{
	// A 类：用户端错误
	"A0001": USER_ERROR,
	"A0002": REPEAT_SUBMIT_ERROR,
	"A0200": USER_LOGIN_ERROR,
	"A0201": USER_NOT_EXIST,
	"A0202": USER_ACCOUNT_LOCKED,
	"A0203": USER_ACCOUNT_INVALID,
	"A0210": USERNAME_OR_PASSWORD_ERROR,
	"A0211": PASSWORD_ENTER_EXCEED_LIMIT,
	"A0212": CLIENT_AUTH_FAILED,
	"A0213": VERIFY_CODE_TIMEOUT,
	"A0214": VERIFY_CODE_ERROR,
	"A0230": TOKEN_INVALID,
	"A0231": TOKEN_ACCESS_FORBIDDEN,
	"A0300": AUTHORIZED_ERROR,
	"A0301": ACCESS_UNAUTHORIZED,
	"A0302": FORBIDDEN_OPERATION,
	"A0400": PARAM_ERROR,
	"A0401": RESOURCE_NOT_FOUND,
	"A0410": PARAM_IS_NULL,
	"A0500": BUSINESS_ERROR,
	"A0501": DATA_EXISTS,
	"A0502": DATA_STATE_NOT_ALLOW,
	"A0503": OPERATION_NOT_ALLOW,
	"A0504": DATA_BIND_EXISTS,
	// 会员模块 A051x
	"A0510": MEMBER_NOT_FOUND,
	"A0511": MEMBER_FROZEN,
	"A0512": SIGN_IN_ALREADY,
	"A0513": GROWTH_INSUFFICIENT,
	"A0514": BENEFIT_CONFIG_INVALID,
	"A0515": QUOTA_EXCEEDED,
	// 套餐模块 A052x
	"A0520": PACKAGE_NOT_FOUND,
	"A0521": PACKAGE_OFF_SHELF,
	"A0522": PACKAGE_HAS_ORDERS,
	"A0523": COUPON_NOT_FOUND,
	"A0524": COUPON_EXPIRED,
	"A0525": COUPON_ALREADY_USED,
	"A0526": COUPON_STOCK_EMPTY,
	"A0527": COUPON_NOT_APPLICABLE,
	"A0528": COUPON_LIMIT_EXCEEDED,
	"A0529": COUPON_STATUS_INVALID,
	"A052A": COUPON_LOCK_FAILED,
	// 订单模块 A053x
	"A0530": ORDER_NOT_FOUND,
	"A0531": ORDER_STATUS_INVALID,
	"A0532": ORDER_EXPIRED,
	"A0533": ORDER_ALREADY_PAID,
	"A0534": REFUND_TIME_EXCEEDED,
	"A0535": REFUND_USAGE_EXCEEDED,
	"A0536": REFUND_NOT_SUPPORTED,
	"A0537": REFUND_NOT_FOUND,
	"A0538": PAYMENT_AMOUNT_MISMATCH,
	"A0539": DUPLICATE_ORDER,
	"A053A": REFUND_ALREADY_EXISTS,
	// 反馈评价模块 A054x
	"A0540": RATING_ALREADY_EXISTS,
	"A0541": RATING_NOT_FOUND,
	"A0542": RATING_EXPIRED,
	"A0543": FEEDBACK_NOT_FOUND,
	"A0544": FEEDBACK_CLOSED,
	"A0545": FEEDBACK_LIMIT_EXCEEDED,
	"A0546": PREDICTION_LOG_NOT_FOUND,
	"A0600": OPERATION_FAILED,
	"A0601": OPERATION_COMPLETED,
	"A0700": USER_UPLOAD_FILE_ERROR,
	"A0701": USER_UPLOAD_FILE_TYPE_NOT_MATCH,
	"A0702": USER_UPLOAD_FILE_SIZE_EXCEEDS,
	"A0703": USER_UPLOAD_IMAGE_SIZE_EXCEEDS,
	"A0704": IMPORT_FILE_EMPTY,
	"A0705": IMPORT_FILE_PARSE_ERROR,
	"A0706": IMPORT_TEMPLATE_MISMATCH,
	"A0707": IMPORT_REQUIRED_FIELD_MISSING,
	"A0708": IMPORT_DATA_VALIDATE_ERROR,
	"A0709": IMPORT_ROWS_EXCEED_LIMIT,
	"A0710": EXPORT_ROWS_EXCEED_LIMIT,
	"A0711": MODULE_IMPORT_NOT_SUPPORTED,
	"A0712": MODULE_EXPORT_NOT_SUPPORTED,

	// B 类：系统端错误
	"B0001": SYSTEM_EXECUTION_ERROR,
	"B0100": SYSTEM_EXECUTION_TIMEOUT,
	"B0101": SYSTEM_ORDER_PROCESSING_TIMEOUT,
	"B0200": SYSTEM_DISASTER_RECOVERY_TRIGGER,
	"B0210": FLOW_LIMIT,
	"B0211": RATE_LIMIT,
	"B0220": DEGRADATION,
	"B0300": SYSTEM_RESOURCE_ERROR,
	"B0308": TASK_CONCURRENT_EXCEED,
	"B0310": SYSTEM_RESOURCE_EXHAUSTION,
	"B0320": SYSTEM_RESOURCE_ACCESS_ERR,
	"B0321": SYSTEM_READ_DISK_FILE_ERR,

	// C 类：第三方服务错误
	"C0001": CALL_THIRD_PARTY_SERVICE_ERROR,
	"C0100": MIDDLEWARE_SERVICE_ERROR,
	"C0113": INTERFACE_NOT_EXIST,
	"C0120": MESSAGE_SERVICE_ERROR,
	"C0121": MESSAGE_DELIVERY_ERROR,
	"C0122": MESSAGE_CONSUMPTION_ERROR,
	"C0123": MESSAGE_SUBSCRIPTION_ERROR,
	"C0124": MESSAGE_GROUP_NOT_FOUND,
	"C0200": CACHE_SERVICE_ERROR,
	"C0201": CACHE_MISS,
	"C0202": CACHE_WRITE_FAILED,
	"C0300": DATABASE_ERROR,
	"C0311": DATABASE_TABLE_NOT_EXIST,
	"C0312": DATABASE_COLUMN_NOT_EXIST,
	"C0321": DATABASE_DUPLICATE_COLUMN,
	"C0331": DATABASE_DEADLOCK,
	"C0341": DATABASE_PRIMARY_KEY_CONFLICT,
	"C0400": OBJECT_STORAGE_ERROR,
	"C0401": FILE_UPLOAD_FAILED,
	"C0402": FILE_DOWNLOAD_FAILED,

	// 成功
	"00000": SUCCESS,
}

// GetValue 根据 code 字符串获取对应的 ResultCode
// 如果未找到，返回 SYSTEM_EXECUTION_ERROR 作为默认值
// 仿照 Java 的 ResultCode.getValue() 方法
func GetValue(code string) *ResultCode {
	if strings.ToUpper(code) == "00000" {
		return SUCCESS
	}

	if rc, ok := allResultCodes[strings.ToUpper(code)]; ok {
		return rc
	}

	// 默认返回系统执行错误
	return SYSTEM_EXECUTION_ERROR
}

// GetMsg 根据 code 字符串获取对应的错误消息
// 如果未找到，返回"未知错误"
func GetMsg(code string) string {
	rc := GetValue(code)
	return rc.Msg
}

// IsSuccess 判断是否为成功状态码
func IsSuccess(code string) bool {
	return strings.ToUpper(code) == SUCCESS.Code
}
