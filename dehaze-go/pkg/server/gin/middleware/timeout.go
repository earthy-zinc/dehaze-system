package middleware

import (
	"net/http"
	"time"

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
func timeoutResponse(c *gin.Context) {
	c.Header("Connection", "close")
	c.AbortWithStatusJSON(http.StatusGatewayTimeout, gin.H{
		"code": 504,
		"msg":  "请求超时",
	})
}
