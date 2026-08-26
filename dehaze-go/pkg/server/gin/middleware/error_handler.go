package middleware

import (
	"encoding/json"
	"errors"
	"io"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"go.uber.org/zap"
)

// ContextErrorHandler 统一处理通过 c.Error 传递的错误
func ContextErrorHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()

		if c.Writer.Written() {
			return
		}

		if len(c.Errors) == 0 {
			return
		}

		err := c.Errors.Last().Err

		// 优先处理 validator 参数校验错误，翻译为中文后返回
		if _, ok := err.(validator.ValidationErrors); ok {
			msg := dehazevalidator.TranslateValidationErrors(err)
			logger.WithContext(c.Request.Context()).Warn("参数校验失败: "+msg,
				zap.String("code", common.PARAM_ERROR.Code),
				zap.Int("status", 400),
			)
			common.FailWithCodeAndMessage(common.PARAM_ERROR, msg, c)
			c.Abort()
			return
		}

		// 绑定类错误：gin ShouldBind 对非 validator 的解析失败（非法 JSON、空 body、JSON 类型不匹配）。
		// 这些确定属于客户端请求格式问题，归 A0400 而非被兜底成系统错误；
		// 其它无法确认归属的错误保持 B0001（宁缺毋滥，避免掩盖真正的系统异常）。
		if isRequestBindingError(err) {
			logger.WithContext(c.Request.Context()).Warn("请求绑定失败: "+err.Error(),
				zap.String("code", common.PARAM_ERROR.Code),
				zap.Int("status", 400),
			)
			common.FailWithCodeAndMessage(common.PARAM_ERROR, err.Error(), c)
			c.Abort()
			return
		}

		common.HandleError(err, c)
		c.Abort()
	}
}

// isRequestBindingError 判断错误是否来自 gin ShouldBind 对请求体的解析失败。
// 仅圈定确定属于客户端请求格式问题的类型：
//   - *json.SyntaxError：非法 JSON
//   - io.EOF：空 body（JSON 解码读到流末尾）
//   - *json.UnmarshalTypeError：JSON 类型不匹配（如数字字段传字符串）
//
// 其它错误（如 Content-Type 不符导致 validator.required 失败、真正的系统异常）不在此列，
// 保持原处理链（宁缺毋滥，避免把系统错误误报为客户端错误）。
func isRequestBindingError(err error) bool {
	if err == nil {
		return false
	}
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return true
	}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return true
	}
	// 空 body 时 gin 的 codec/json 把 io.EOF 经 Unwrap 链透传，errors.Is 可识别
	return errors.Is(err, io.EOF)
}
