package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterClientLogRoutes 注册前端日志接收路由。
//
// 路径 /api/v1/logs/client，匿名允许上报（OptionalSessionAuth 仅在携带合法 session 时解析
// user_id，未登录放行）；路径级限流 PathRateLimiter(60, 1000)（60 秒 1000 次，IP 维度）。
func RegisterClientLogRoutes(rg *gin.RouterGroup, clientLogApi *api.ClientLogApi) {
	clientLogGroup := rg.Group("/logs")
	clientLogGroup.Use(middleware.OptionalSessionAuth())
	clientLogGroup.POST("/client", middleware.PathRateLimiter(60, 1000), clientLogApi.Collect)
}
