package common

import (
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
		logger.Warn("业务错误",
			zap.String("code", bizErr.Code().Code),
			zap.String("msg", bizErr.Message()),
			zap.Error(err),
		)
		FailWithCodeAndMessage(bizErr.Code(), bizErr.Message(), c)
		return
	}

	// 记录未处理错误，避免对外暴露内部错误
	logger.Error("未处理错误", zap.Error(err))
	FailWithCode(SYSTEM_EXECUTION_ERROR, c)
}
