package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterAiBillingRoutes 注册 AI 计费管理路由（/api/v1/ai-billing，18 端点）。
//
// 权限标识：ai:billing:stat（统计/异常/下钻他人数据）、ai:billing:adjust（手动调整积分）、
// ai:billing:refund（退款申请列表与审核）、ai:billing:cost（成本单价与成本-利润统计）。
// 权限在 handler 内于参数校验之后执行（与 python require_permission 的执行顺序一致）。
func RegisterAiBillingRoutes(rg *gin.RouterGroup, billingApi *api.AiBillingApi) {
	group := rg.Group("/ai-billing")
	{
		group.GET("/balance", billingApi.GetBalance)
		group.GET("/summary", billingApi.GetSummary)
		group.GET("/records", billingApi.ListRecords)
		group.GET("/credit-logs", billingApi.ListCreditLogs)
		group.GET("/bills/:month", billingApi.GetBill)
		group.GET("/bills/:month/download", billingApi.DownloadBill)

		group.POST("/refunds", billingApi.ApplyRefund)
		group.GET("/refunds", billingApi.ListRefunds)
		group.POST("/refunds/:id/audit", billingApi.AuditRefund)

		group.GET("/stats", billingApi.GetStats)
		group.POST("/adjust", billingApi.AdjustCredits)
		group.GET("/anomalies", billingApi.ListAnomalies)

		group.GET("/costs", billingApi.ListCosts)
		group.POST("/costs", billingApi.CreateCost)
		group.PUT("/costs/:id", billingApi.UpdateCost)
		group.DELETE("/costs/:id", billingApi.DeleteCost)
		group.GET("/cost-stats", billingApi.GetCostStats)
		group.POST("/reconcile/import", billingApi.ImportReconcile)
	}
}
