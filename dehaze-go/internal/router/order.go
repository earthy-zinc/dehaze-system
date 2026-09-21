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

		// 余额域（python order.py 同口径）：充值/余额查询/余额退款申请与管理员审核
		orderRouter.GET("/balance", orderApi.GetBalance)
		orderRouter.POST("/balance-refund", orderApi.ApplyBalanceRefund)
		orderRouter.PUT("/balance-refunds/:refundId/audit", middleware.Permission("order:refund:approve"), orderApi.AuditBalanceRefund)
		orderRouter.POST("/recharge", orderApi.CreateRecharge)

		refundRouter := orderRouter.Group("/refunds")
		{
			// 权限标识与 python @require_permission 一致（order:refund:list / order:refund:approve）
			refundRouter.GET("/page", middleware.Permission("order:refund:list"), orderApi.ListRefunds)
			refundRouter.PUT("/:refundId/approve", middleware.Permission("order:refund:approve"), orderApi.ApproveRefund)
			refundRouter.PUT("/:refundId/reject", middleware.Permission("order:refund:approve"), orderApi.RejectRefund)
		}

		orderRouter.GET("/:orderNo", orderApi.GetDetail)
		orderRouter.PUT("/:orderNo/cancel", orderApi.Cancel)
		orderRouter.POST("/:orderNo/pay", orderApi.Pay)
		orderRouter.POST("/:orderNo/refund", orderApi.ApplyRefund)
	}
}
