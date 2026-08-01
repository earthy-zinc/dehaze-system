package middleware

import (
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"github.com/gin-contrib/timeout"
	"github.com/gin-gonic/gin"
)

// TimeoutMiddleware 创建超时中间件
// 入参 duration 设置超时时间（例如：time.Second * 5）
// 使用示例: xxx.Get("path", middleware.TimeoutMiddleware(30*time.Second), HandleFunc)
func TimeoutMiddleware(duration time.Duration) gin.HandlerFunc {
	return timeout.New(
		timeout.WithTimeout(duration),
		timeout.WithResponse(timeoutResponse),
	)
}

// timeoutResponse 超时响应处理
// 使用与 common.Response 一致的结构，便于前端拦截器统一解析
func timeoutResponse(c *gin.Context) {
	c.Header("Connection", "close")
	c.JSON(http.StatusOK, common.Response{
		Code:    common.SYSTEM_EXECUTION_TIMEOUT.Code,
		Msg:     common.SYSTEM_EXECUTION_TIMEOUT.Msg,
		TraceId: trace.FromContext(c.Request.Context()),
	})
}
