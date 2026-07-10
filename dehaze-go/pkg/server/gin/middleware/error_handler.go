package middleware

import (
	"github.com/earthyzinc/dehaze-go/pkg/common"
	dehazevalidator "github.com/earthyzinc/dehaze-go/pkg/validator"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
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
			common.FailWithCodeAndMessage(common.PARAM_ERROR, msg, c)
			c.Abort()
			return
		}

		common.HandleError(err, c)
		c.Abort()
	}
}
