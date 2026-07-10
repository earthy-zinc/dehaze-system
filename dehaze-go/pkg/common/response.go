package common

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// Response 通用响应结构（与 Java 参考实现保持一致）
type Response struct {
	Code      string      `json:"code"`
	Data      interface{} `json:"data,omitempty"`
	Msg       string      `json:"msg"`
	TraceId   string      `json:"traceId,omitempty"`
	Timestamp int64       `json:"timestamp,omitempty"`
	Errors    []ErrorItem `json:"errors,omitempty"`
}

// ErrorItem 参数校验错误项
type ErrorItem struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Code    string `json:"code"`
}

// result 核心响应函数
func result(resultCode *ResultCode, data interface{}, c *gin.Context) {
	c.JSON(http.StatusOK, Response{
		Code: resultCode.Code,
		Data: data,
		Msg:  resultCode.Msg,
	})
}

// resultWithMsg 使用指定消息覆盖默认消息
func resultWithMsg(resultCode *ResultCode, data interface{}, message string, c *gin.Context) {
	c.JSON(http.StatusOK, Response{
		Code: resultCode.Code,
		Data: data,
		Msg:  message,
	})
}

// ========== 成功响应 ==========

// Ok 操作成功，返回空数据
// 仿照 Java: return Result.ok();
func Ok(c *gin.Context) {
	result(SUCCESS, nil, c)
}

// OkWithMessage 操作成功，返回自定义消息
// 仿照 Java: return Result.ok().message("自定义消息");
func OkWithMessage(message string, c *gin.Context) {
	resultWithMsg(SUCCESS, nil, message, c)
}

// OkWithData 操作成功，返回数据
// 仿照 Java: return Result.ok(data);
func OkWithData(data interface{}, c *gin.Context) {
	result(SUCCESS, data, c)
}

// OkWithDetailed 操作成功，返回数据和消息
// 仿照 Java: return Result.ok(data).message("消息");
func OkWithDetailed(data interface{}, message string, c *gin.Context) {
	resultWithMsg(SUCCESS, data, message, c)
}

// ========== 失败响应 ==========

// Fail 操作失败，使用通用错误码
// 仿照 Java: return Result.failed(ResultCode.SYSTEM_EXECUTION_ERROR);
func Fail(c *gin.Context) {
	result(SYSTEM_EXECUTION_ERROR, map[string]interface{}{}, c)
}

// FailWithMessage 操作失败，返回自定义消息
// 仿照 Java: return Result.failed(ResultCode.SYSTEM_EXECUTION_ERROR).message("自定义消息");
func FailWithMessage(message string, c *gin.Context) {
	resultWithMsg(SYSTEM_EXECUTION_ERROR, map[string]interface{}{}, message, c)
}

// FailWithCode 操作失败，使用指定错误码
// 仿照 Java: return Result.failed(ResultCode.PARAM_ERROR);
func FailWithCode(resultCode *ResultCode, c *gin.Context) {
	result(resultCode, map[string]interface{}{}, c)
}

// FailWithCodeAndMessage 操作失败，使用指定错误码和消息
// 仿照 Java: return Result.failed(ResultCode.PARAM_ERROR).message("自定义消息");
func FailWithCodeAndMessage(resultCode *ResultCode, message string, c *gin.Context) {
	resultWithMsg(resultCode, map[string]interface{}{}, message, c)
}

// FailWithDataAndCode 操作失败，返回数据和指定错误码
// 仿照 Java: return Result.failed(ResultCode.PARAM_ERROR).data(data);
func FailWithDataAndCode(resultCode *ResultCode, data interface{}, c *gin.Context) {
	result(resultCode, data, c)
}

// FailWithDetailed 操作失败，返回数据和消息
func FailWithDetailed(data interface{}, message string, c *gin.Context) {
	resultWithMsg(SYSTEM_EXECUTION_ERROR, data, message, c)
}
