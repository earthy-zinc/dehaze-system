package common

import (
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// HandleError 统一错误处理，根据错误类型返回对应响应
func HandleError(err error, c *gin.Context) {
	if err == nil {
		return
	}

	// 尝试转换为业务错误
	if bizErr, ok := AsBizError(err); ok {
		logger.WithContext(c.Request.Context()).Warn("业务异常: "+bizErr.Message(),
			zap.String("code", bizErr.Code().Code),
			zap.Int("status", 400),
		)
		FailWithCodeAndMessage(bizErr.Code(), bizErr.Message(), c)
		return
	}

	// 记录未处理错误，避免对外暴露内部错误
	logger.WithContext(c.Request.Context()).Error("未处理异常: "+err.Error(),
		zap.String("code", SYSTEM_EXECUTION_ERROR.Code),
		zap.Int("status", 500),
		zap.String("exc_info", fmt.Sprintf("%+v", err)),
	)
	FailWithCode(SYSTEM_EXECUTION_ERROR, c)
}
