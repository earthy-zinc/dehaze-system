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

	// A06xx: 操作相关
	OPERATION_FAILED    = &ResultCode{"A0600", "操作失败"}
	OPERATION_COMPLETED = &ResultCode{"A0601", "操作已完成"}

	// A07xx: 文件上传
	USER_UPLOAD_FILE_ERROR          = &ResultCode{"A0700", "用户上传文件异常"}
	USER_UPLOAD_FILE_TYPE_NOT_MATCH = &ResultCode{"A0701", "用户上传文件类型不匹配"}
	USER_UPLOAD_FILE_SIZE_EXCEEDS   = &ResultCode{"A0702", "用户上传文件太大"}
	USER_UPLOAD_IMAGE_SIZE_EXCEEDS  = &ResultCode{"A0703", "用户上传图片太大"}

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
	"A0600": OPERATION_FAILED,
	"A0601": OPERATION_COMPLETED,
	"A0700": USER_UPLOAD_FILE_ERROR,
	"A0701": USER_UPLOAD_FILE_TYPE_NOT_MATCH,
	"A0702": USER_UPLOAD_FILE_SIZE_EXCEEDS,
	"A0703": USER_UPLOAD_IMAGE_SIZE_EXCEEDS,

	// B 类：系统端错误
	"B0001": SYSTEM_EXECUTION_ERROR,
	"B0100": SYSTEM_EXECUTION_TIMEOUT,
	"B0101": SYSTEM_ORDER_PROCESSING_TIMEOUT,
	"B0200": SYSTEM_DISASTER_RECOVERY_TRIGGER,
	"B0210": FLOW_LIMIT,
	"B0211": RATE_LIMIT,
	"B0220": DEGRADATION,
	"B0300": SYSTEM_RESOURCE_ERROR,
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
