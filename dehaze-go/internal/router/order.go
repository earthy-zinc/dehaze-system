package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterOrderRoutes(rg *gin.RouterGroup, orderApi *api.OrderApi) {
	orderRouter := rg.Group("/orders")
	{
		orderRouter.POST("", orderApi.Create)
		orderRouter.GET("/my", orderApi.ListMy)
		orderRouter.GET("/page", middleware.Permission("order:list"), orderApi.GetPage)
		orderRouter.GET("/stats", middleware.Permission("order:stats"), orderApi.GetStats)
		orderRouter.PUT("/auto-renew/config", orderApi.UpdateAutoRenewConfig)
		orderRouter.GET("/auto-renew/config", orderApi.GetAutoRenewConfig)

		refundRouter := orderRouter.Group("/refunds")
		{
			refundRouter.GET("/page", middleware.Permission("refund:list"), orderApi.ListRefunds)
			refundRouter.PUT("/:refundId/approve", middleware.Permission("refund:audit"), orderApi.ApproveRefund)
			refundRouter.PUT("/:refundId/reject", middleware.Permission("refund:audit"), orderApi.RejectRefund)
		}

		orderRouter.GET("/:orderNo", orderApi.GetDetail)
		orderRouter.PUT("/:orderNo/cancel", orderApi.Cancel)
		orderRouter.POST("/:orderNo/pay", orderApi.Pay)
		orderRouter.POST("/:orderNo/refund", orderApi.ApplyRefund)
	}
}
