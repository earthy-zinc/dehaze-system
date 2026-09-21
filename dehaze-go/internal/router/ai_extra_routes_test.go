package router

import (
	"net/http"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestRegisterAiA2AAndCompatRoutes A2A Agent Card 与兼容调用审计路由注册不得 panic，
// 且不与 go-proxy 的转发路径（/a2a、/.well-known/agent.json、/ai/agents/:id/test 等）重叠。
func TestRegisterAiA2AAndCompatRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")

	require.NotPanics(t, func() {
		RegisterAiA2ARoutes(v1, api.NewAiA2AApi(nil))
		RegisterAiCompatAuditRoutes(v1, api.NewAiCompatAuditApi(nil))
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	expected := []string{
		http.MethodGet + " /api/v1/ai/agents/:id/a2a/.well-known/agent.json",
		http.MethodGet + " /api/v1/ai/compat/calls",
	}
	for _, path := range expected {
		require.True(t, registered[path], "路由未注册: %s", path)
	}
}

// TestAgentCardRouteRegisteredOnce 原生 A2A 路由与转发白名单挂在同一路由组时必须能共存：
// Agent Card 发现端点由 Go 原生实现（ai_a2a.go），转发白名单（ai_proxy.go）若重复注册同一
// 「方法 + 路径」，gin 注册阶段直接 panic（服务启动失败）——这是启动的最后一道保险。
func TestAgentCardRouteRegisteredOnce(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")

	require.NotPanics(t, func() {
		RegisterAiA2ARoutes(v1, api.NewAiA2AApi(nil))
		RegisterAIProxyRoutes(v1, newTestProxyApi(t))
	})
}

// TestKbRoutesCoexistWithProxyWhitelist 原生知识库 A 类路由（ai_kb.go）与转发白名单（ai_proxy.go）
// 挂在同一 /api/v1 组下必须能共存：库级创建/删除/索引统计归转发，原生不得再声明（否则同一
// 「方法 + 路径」重复注册会让 gin 启动 panic）；同时 /kb/documents/:id 与 /kb/:id/index-stats
// 这类「静态兄弟 + 参数兄弟」形态也不得在注册阶段崩溃。
func TestKbRoutesCoexistWithProxyWhitelist(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")

	require.NotPanics(t, func() {
		RegisterAiKbRoutes(v1, api.NewAiKbApi(nil))
		RegisterAIProxyRoutes(v1, newTestProxyApi(t))
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	expected := []string{
		// 转发侧：库级三条
		http.MethodPost + " /api/v1/kb",
		http.MethodGet + " /api/v1/kb/:id/index-stats",
		http.MethodDelete + " /api/v1/kb/:id",
		// 原生侧：库列表/详情/编辑仍归 Go
		http.MethodGet + " /api/v1/kb",
		http.MethodGet + " /api/v1/kb/:id",
		http.MethodPut + " /api/v1/kb/:id",
	}
	for _, path := range expected {
		require.True(t, registered[path], "路由未注册: %s", path)
	}
}
