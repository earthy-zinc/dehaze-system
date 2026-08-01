package middleware

import (
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

		common.HandleError(err, c)
		c.Abort()
	}
}
