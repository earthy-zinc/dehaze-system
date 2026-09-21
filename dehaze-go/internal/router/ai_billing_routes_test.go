package router

import (
	"net/http"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestRegisterAiBillingAndObservabilityRoutes AI 计费 18 端点 + 可观测性 8 端点必须全部注册成功，
// 且与 go-proxy 的转发路径（/ai/conversations/:id/messages 等）、A 类会话路由共存不 panic
// （gin 同一路径位置的通配符名与静态段都不能冲突）。
func TestRegisterAiBillingAndObservabilityRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")

	require.NotPanics(t, func() {
		RegisterAiBillingRoutes(v1, api.NewAiBillingApi(nil, nil))
		RegisterAiObservabilityRoutes(v1, api.NewAiObservabilityApi(nil))
		RegisterAiConversationRoutes(v1, api.NewAiConversationApi(nil, nil, nil, nil))
		RegisterAIProxyRoutes(v1, api.NewAIProxyApi(nil))
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}

	billingRoutes := []string{
		http.MethodGet + " /api/v1/ai-billing/balance",
		http.MethodGet + " /api/v1/ai-billing/summary",
		http.MethodGet + " /api/v1/ai-billing/records",
		http.MethodGet + " /api/v1/ai-billing/credit-logs",
		http.MethodGet + " /api/v1/ai-billing/bills/:month",
		http.MethodGet + " /api/v1/ai-billing/bills/:month/download",
		http.MethodPost + " /api/v1/ai-billing/refunds",
		http.MethodGet + " /api/v1/ai-billing/refunds",
		http.MethodPost + " /api/v1/ai-billing/refunds/:id/audit",
		http.MethodGet + " /api/v1/ai-billing/stats",
		http.MethodPost + " /api/v1/ai-billing/adjust",
		http.MethodGet + " /api/v1/ai-billing/anomalies",
		http.MethodGet + " /api/v1/ai-billing/costs",
		http.MethodPost + " /api/v1/ai-billing/costs",
		http.MethodPut + " /api/v1/ai-billing/costs/:id",
		http.MethodDelete + " /api/v1/ai-billing/costs/:id",
		http.MethodGet + " /api/v1/ai-billing/cost-stats",
		http.MethodPost + " /api/v1/ai-billing/reconcile/import",
	}
	observabilityRoutes := []string{
		http.MethodGet + " /api/v1/ai/observability/summary",
		http.MethodGet + " /api/v1/ai/observability/traces",
		http.MethodGet + " /api/v1/ai/observability/traces/export",
		http.MethodGet + " /api/v1/ai/observability/traces/:traceId",
		http.MethodGet + " /api/v1/ai/observability/conversations/:id/timeline",
		http.MethodGet + " /api/v1/ai/observability/conversations/:id/timeline/export",
		http.MethodGet + " /api/v1/ai/observability/costs",
		http.MethodGet + " /api/v1/ai/observability/trends",
	}
	require.Len(t, billingRoutes, 18, "AI 计费端点数")
	require.Len(t, observabilityRoutes, 8, "可观测性端点数")

	for _, route := range append(billingRoutes, observabilityRoutes...) {
		require.True(t, registered[route], "未注册路由: %s", route)
	}
}
