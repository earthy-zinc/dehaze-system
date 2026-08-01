package middleware

import (
	"fmt"
	"net"
	"os"
	"runtime/debug"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// Recovery recover掉项目可能出现的panic，并使用zap记录相关日志
func Recovery(stack bool) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if err := recover(); err != nil {
				// Check for a broken connection, as it is not really a
				// condition that warrants a panic stack trace.
				var brokenPipe bool
				if ne, ok := err.(*net.OpError); ok {
					if se, ok := ne.Err.(*os.SyscallError); ok {
						if strings.Contains(strings.ToLower(se.Error()), "broken pipe") || strings.Contains(strings.ToLower(se.Error()), "connection reset by peer") {
							brokenPipe = true
						}
					}
				}

				log := logger.WithContext(c.Request.Context())
				if brokenPipe {
					log.Error("未处理异常: broken pipe",
						zap.String("code", common.SYSTEM_EXECUTION_ERROR.Code),
						zap.Int("status", 500),
						zap.String("exc_info", fmt.Sprintf("%v", err)),
					)
					c.Abort()
					return
				}

				fields := []zap.Field{
					zap.String("code", common.SYSTEM_EXECUTION_ERROR.Code),
					zap.Int("status", 500),
					zap.String("exc_info", string(debug.Stack())),
				}
				log.Error("未处理异常: panic recovered", fields...)

				if !c.Writer.Written() {
					_ = c.Error(common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "系统执行出错"))
				}
				c.Abort()
			}
		}()
		c.Next()
	}
}
