package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiObservabilityApi AI 可观测性查询（F-M08-013）。
//
// summary/traces/export/costs/trends 为管理端审计接口（ai:conversation:audit）；
// 过程链详情与时间线登录即可访问，普通用户仅可查自己会话的过程链（跨会话 A0401）。
// 权限校验放在参数绑定之后，对齐 FastAPI"先校验再执行权限装饰器"的顺序（非法参数报 A0400）。
type AiObservabilityApi struct {
	observability *aiservice.ObservabilityService
}

func NewAiObservabilityApi(observability *aiservice.ObservabilityService) *AiObservabilityApi {
	return &AiObservabilityApi{observability: observability}
}

// hasAuditPermission 是否具备管理端审计权限（ROOT 放行，否则需 ai:conversation:audit）。
func hasAuditPermission(c *gin.Context) bool {
	if security.IsRoot(c) {
		return true
	}
	has, err := security.HasAnyPermission(c, "ai:conversation:audit")
	return err == nil && has
}

// GetSummary 异常总览统计。
func (a *AiObservabilityApi) GetSummary(c *gin.Context) {
	if !requireConversationAudit(c) {
		return
	}
	result, err := a.observability.Summary(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListTraces 过程链检索。
func (a *AiObservabilityApi) ListTraces(c *gin.Context) {
	var query bo.TracePageQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	// 分页口径对齐 python BasePageQuery（pageNum default=1 ge=1 / pageSize default=10 ge=1 le=100）：
	// 缺省取 1/10，显式传非数字、<1 或 >100 一律 A0400（不得静默回退默认值）。
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	if !requireConversationAudit(c) {
		return
	}
	result, err := a.observability.ListTraces(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ExportTraces 过程链导出（CSV，UTF-8 BOM）。
func (a *AiObservabilityApi) ExportTraces(c *gin.Context) {
	var query bo.TracePageQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	// 导出不按页裁剪数据，但 python 的 TracePageQuery 仍会校验分页参数，故此处同样校验（口径一致）
	if _, _, ok := parsePaginationWithSize(c, 10); !ok {
		return
	}
	if !requireConversationAudit(c) {
		return
	}
	payload, err := a.observability.ExportTraces(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Header("Content-Disposition", "attachment; filename=\"ai_traces.csv\"")
	c.Data(http.StatusOK, "text/csv; charset=utf-8", payload)
}

// GetTrace 过程链详情（管理员全量，普通用户仅自己会话）。
func (a *AiObservabilityApi) GetTrace(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.observability.GetTrace(
		c.Request.Context(), c.Param("traceId"), userID, hasAuditPermission(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetConversationTimeline 会话审计时间线。
func (a *AiObservabilityApi) GetConversationTimeline(c *gin.Context) {
	conversationID, ok := parseID(c, "id")
	if !ok {
		return
	}
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	// 未传 include 视为含 raw 原始报文，传非 raw 值则省略供轻量预览
	includeRaw := true
	if include, exists := c.GetQuery("include"); exists {
		includeRaw = false
		for _, item := range strings.Split(include, ",") {
			if item == "raw" {
				includeRaw = true
				break
			}
		}
	}
	result, err := a.observability.ConversationTimeline(
		c.Request.Context(), conversationID, userID, hasAuditPermission(c), includeRaw)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ExportConversationTimeline 会话时间线导出（JSON 全量含 raw 报文）。
func (a *AiObservabilityApi) ExportConversationTimeline(c *gin.Context) {
	conversationID, ok := parseID(c, "id")
	if !ok {
		return
	}
	if !requireConversationAudit(c) {
		return
	}
	result, err := a.observability.ExportTimeline(c.Request.Context(), conversationID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	payload, err := json.Marshal(result)
	if err != nil {
		_ = c.Error(err)
		return
	}
	c.Header("Content-Disposition",
		fmt.Sprintf("attachment; filename=\"conversation_%d_timeline.json\"", conversationID))
	c.Data(http.StatusOK, "application/json", payload)
}

// GetCosts 资源消耗聚合。
func (a *AiObservabilityApi) GetCosts(c *gin.Context) {
	var query bo.CostsQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	query.PageNum, query.PageSize = page, size
	if !requireConversationAudit(c) {
		return
	}
	result, err := a.observability.Costs(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetTrends 性能趋势。
func (a *AiObservabilityApi) GetTrends(c *gin.Context) {
	var query bo.TrendsQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	if !requireConversationAudit(c) {
		return
	}
	result, err := a.observability.Trends(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}
