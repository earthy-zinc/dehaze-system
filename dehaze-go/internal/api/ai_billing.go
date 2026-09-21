package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// AiBillingApi AI 计费管理（用户端余额/明细/账单/退款申诉 + 管理端统计/调整/审核/异常/成本）。
//
// 权限校验一律放在 handler 内、参数绑定之后：FastAPI 先做请求体/查询/路径参数校验再执行
// 权限装饰器，非法参数 + 无权限的请求在 Python 端返回 A0400 而非 A0301。
type AiBillingApi struct {
	billing *aiservice.BillingService
	cost    *aiservice.CostService
}

func NewAiBillingApi(billing *aiservice.BillingService, cost *aiservice.CostService) *AiBillingApi {
	return &AiBillingApi{billing: billing, cost: cost}
}

// resolveQueryUser 解析查询目标用户：管理员可指定 userId 查询他人数据（需 ai:billing:stat），
// 普通用户仅可查本人（越权一律 A0301，与 python _resolve_query_user 一致）。
func (a *AiBillingApi) resolveQueryUser(c *gin.Context, rawUserID string) (int64, bool) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return 0, false
	}
	if rawUserID == "" {
		return userID, true
	}
	target, parseErr := strconv.ParseInt(rawUserID, 10, 64)
	if parseErr != nil || target < 1 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "userId 参数非法"))
		return 0, false
	}
	if target == userID {
		return userID, true
	}
	if permErr := middleware.CheckPermission(c, "ai:billing:stat"); permErr != nil {
		_ = c.Error(permErr)
		return 0, false
	}
	return target, true
}

// ==================== 用户端 ====================

// GetBalance 用户余额查询。
func (a *AiBillingApi) GetBalance(c *gin.Context) {
	target, ok := a.resolveQueryUser(c, c.Query("userId"))
	if !ok {
		return
	}
	result, err := a.billing.Balance(c.Request.Context(), target)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetSummary 消耗汇总查询（dimension: day/month）。
func (a *AiBillingApi) GetSummary(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.Summary(c.Request.Context(), userID, c.DefaultQuery("dimension", "day"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListRecords 计费明细查询。
func (a *AiBillingApi) ListRecords(c *gin.Context) {
	var query bo.BillingRecordQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	// 分页口径对齐 python `Query(default=1, ge=1)` / `Query(default=20, ge=1, le=100)`：
	// 缺省取 1/20，显式传非数字、<1 或 >100 一律 A0400（不得静默回退默认值）。
	page, size, ok := parsePaginationWithSize(c, 20)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	target, ok := a.resolveQueryUser(c, c.Query("userId"))
	if !ok {
		return
	}
	result, err := a.billing.Records(c.Request.Context(), target, &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListCreditLogs 余额流水查询。
func (a *AiBillingApi) ListCreditLogs(c *gin.Context) {
	var query bo.CreditLogQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 20)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	target, ok := a.resolveQueryUser(c, c.Query("userId"))
	if !ok {
		return
	}
	result, err := a.billing.CreditLogs(c.Request.Context(), target, &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetBill 月结账单查询。
func (a *AiBillingApi) GetBill(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.Bill(c.Request.Context(), userID, c.Param("month"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DownloadBill 账单下载（保持 JSON 信封，前端另存为文件）。
func (a *AiBillingApi) DownloadBill(c *gin.Context) {
	a.GetBill(c)
}

// ApplyRefund 退款申请。
func (a *AiBillingApi) ApplyRefund(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.BillingRefundApplyForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.ApplyRefund(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ==================== 管理端 ====================

// ListRefunds 退款申请列表。
func (a *AiBillingApi) ListRefunds(c *gin.Context) {
	var query bo.RefundQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 20)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	if err := middleware.CheckPermission(c, "ai:billing:refund"); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.ListRefunds(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// AuditRefund 退款审核。
func (a *AiBillingApi) AuditRefund(c *gin.Context) {
	refundID, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form bo.BillingRefundAuditForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:refund"); err != nil {
		_ = c.Error(err)
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.AuditRefund(
		c.Request.Context(), refundID, *form.Approved, form.AuditRemark, operatorID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetStats 管理员计费统计。
func (a *AiBillingApi) GetStats(c *gin.Context) {
	var query bo.BillingStatQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:stat"); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.Stats(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// AdjustCredits 管理员手动调整积分。
func (a *AiBillingApi) AdjustCredits(c *gin.Context) {
	var form bo.CreditAdjustForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:adjust"); err != nil {
		_ = c.Error(err)
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.Adjust(c.Request.Context(), operatorID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListAnomalies 异常计费记录查询。
func (a *AiBillingApi) ListAnomalies(c *gin.Context) {
	var query bo.AnomalyQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 20)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	if err := middleware.CheckPermission(c, "ai:billing:stat"); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.billing.Anomalies(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ==================== 成本单价 ====================

// ListCosts 成本单价列表。
func (a *AiBillingApi) ListCosts(c *gin.Context) {
	var query bo.ModelCostQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 20)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.cost.ListCosts(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateCost 新增成本单价。
func (a *AiBillingApi) CreateCost(c *gin.Context) {
	var form bo.ModelCostCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.cost.CreateCost(c.Request.Context(), operatorID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateCost 更新成本单价。
func (a *AiBillingApi) UpdateCost(c *gin.Context) {
	costID, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form bo.ModelCostUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.cost.UpdateCost(c.Request.Context(), costID, operatorID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteCost 删除成本单价。
func (a *AiBillingApi) DeleteCost(c *gin.Context) {
	costID, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.cost.DeleteCost(c.Request.Context(), costID, operatorID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// GetCostStats 成本-利润统计。
func (a *AiBillingApi) GetCostStats(c *gin.Context) {
	var query bo.CostStatQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.cost.CostStats(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ImportReconcile 供应商账单导入。
func (a *AiBillingApi) ImportReconcile(c *gin.Context) {
	var form bo.ReconcileImportForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, "ai:billing:cost"); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(gin.H{"imported": a.cost.ImportReconcile(&form)}, c)
}
