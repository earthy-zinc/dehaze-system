package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/aiclient"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/gin-gonic/gin"
)

func newTestProxyApi(t *testing.T) *api.AIProxyApi {
	t.Helper()
	client, err := aiclient.New(options.AI{ServiceURL: "http://127.0.0.1:8991"})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	return api.NewAIProxyApi(client)
}

func TestRegisterAIProxyRoutes_RegistersWhitelist(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	RegisterAIProxyRoutes(engine.Group("/api/v1"), newTestProxyApi(t))

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}

	// B 类端点白名单：与 dehaze-python/app/router 逐条对齐
	want := []string{
		"POST /api/v1/ai/conversations/:id/messages",
		"POST /api/v1/ai/messages/:id/regenerate",
		"POST /api/v1/ai/messages/:id/resume",
		"PUT /api/v1/ai/messages/:id",
		"POST /api/v1/ai/messages/:id/stop",
		"GET /api/v1/ai/conversations/:id/messages/stream/:stream_id",
		"POST /api/v1/ai/agents/:id/test",
		"POST /api/v1/ai/agents/:id/publish",
		"POST /api/v1/ai/skills/:id/test",
		"POST /api/v1/ai/skills/upload",
		"POST /api/v1/ai/models/:id/test",
		"POST /api/v1/ai/providers/:id/test-connection",
		"GET /api/v1/ai/mcp/servers/:id/health",
		"GET /api/v1/ai/mcp/servers/:id/tools",
		"POST /api/v1/ai/mcp/servers/:id/tools/test",
		"POST /api/v1/ai/agents/:id/eval/runs",
		"GET /api/v1/ai/agents/:id/eval/runs",
		"GET /api/v1/ai/agents/:id/eval/tasks/:task_id",
		"POST /api/v1/ai/scheduled-tasks/:id/run",
		"POST /api/v1/kb",
		"GET /api/v1/kb/:id/index-stats",
		"DELETE /api/v1/kb/:id",
		"POST /api/v1/kb/:id/documents",
		"POST /api/v1/kb/:id/documents/batch",
		"POST /api/v1/kb/:id/documents/text",
		"POST /api/v1/kb/:id/documents/import-url",
		"PUT /api/v1/kb/documents/:id",
		"DELETE /api/v1/kb/documents/:id",
		"POST /api/v1/kb/documents/:id/reprocess",
		"POST /api/v1/kb/documents/chunks/preview",
		"POST /api/v1/kb/search",
		"POST /api/v1/kb/:id/retrieve/test",
		"POST /api/v1/kb/:id/retrieve/test-sets/:test_set_id/run",
		"POST /api/v1/ai/a2a/endpoints/:id/refresh-card",
		"POST /api/v1/ai/agents/:id/a2a",
	}
	for _, path := range want {
		if !registered[path] {
			t.Errorf("缺少 B 类转发路由: %s", path)
		}
	}

	// 白名单外不做兜底转发：A 类 CRUD 路径必须 404（404 说明没有笼统的转发兜底）
	outside := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/v1/ai/conversations/1/export"},
		{http.MethodGet, "/api/v1/ai/agents/1"},
		{http.MethodGet, "/api/v1/kb/1/retrieve/test-sets"},
		{http.MethodPost, "/api/v1/ai/unknown/anything"},
	}
	for _, route := range outside {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if rec.Code != http.StatusNotFound {
			t.Errorf("%s %s 应 404（白名单外不转发），实际 %d", route.method, route.path, rec.Code)
		}
	}
}

// TestRegisterAIProxyRoutes_CoexistsWithCrudRoutes 用 A 类 CRUD 已落盘的路径形态锁定共节点安全：
// 静态与参数兄弟（skills/upload|market 与 skills/:id）、同节点不同 method（GET|DELETE/PUT kb/documents/:id、
// GET/DELETE|PUT ai/messages/:id）、节点 handler + 静态子节点（ai/conversations/:id/messages 与 .../stream/:id）。
// 注册 panic 会直接让本用例失败——这是服务启动的最后一道保险。
func TestRegisterAIProxyRoutes_CoexistsWithCrudRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	rg := engine.Group("/api/v1")
	crud := func(c *gin.Context) {
		c.Header("X-Crud", "1")
		c.Status(http.StatusOK)
	}
	rg.POST("/ai/skills/market", crud)
	rg.GET("/ai/skills/:id", crud)
	rg.POST("/ai/models/:id/prices", crud)
	rg.GET("/kb/documents/:id", crud)
	rg.GET("/ai/conversations/:id/messages", crud)
	rg.DELETE("/ai/messages/:id", crud)

	// 上游指向必然拒绝连接的端口：转发 handler 命中时返回 C0001 信封，可与 CRUD handler 区分
	client, err := aiclient.New(options.AI{ServiceURL: "http://127.0.0.1:1", Timeout: 1, ConnectTimeout: 1})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	RegisterAIProxyRoutes(rg, api.NewAIProxyApi(client))

	crudHits := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/v1/ai/skills/market"},
		{http.MethodGet, "/api/v1/ai/skills/2"},
		{http.MethodPost, "/api/v1/ai/models/3/prices"},
		{http.MethodGet, "/api/v1/kb/documents/5"},
		{http.MethodGet, "/api/v1/ai/conversations/9/messages"},
		{http.MethodDelete, "/api/v1/ai/messages/4"},
	}
	for _, route := range crudHits {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if rec.Header().Get("X-Crud") != "1" {
			t.Errorf("%s %s 应仍由 CRUD handler 处理，实际 status=%d", route.method, route.path, rec.Code)
		}
	}

	proxyHits := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/v1/ai/skills/upload"},
		{http.MethodPut, "/api/v1/kb/documents/5"},
		{http.MethodDelete, "/api/v1/kb/documents/5"},
		{http.MethodPost, "/api/v1/kb/documents/5/reprocess"},
		{http.MethodPut, "/api/v1/ai/messages/4"},
		{http.MethodGet, "/api/v1/ai/conversations/9/messages/stream/s-1"},
	}
	for _, route := range proxyHits {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if !strings.Contains(rec.Body.String(), common.CALL_THIRD_PARTY_SERVICE_ERROR.Code) {
			t.Errorf("%s %s 应由转发 handler 处理（预期上游不可达信封），实际 status=%d body=%s",
				route.method, route.path, rec.Code, rec.Body.String())
		}
	}
}

// TestRegisterAIProxyRoutes_KbLevelRoutesAreForwarded 锁定知识库库级三条的转发归属与共节点安全：
// 原生 A 类路由（ai_kb.go）与转发白名单挂在同一 /kb 节点下（静态 documents 与参数 :id 兄弟），
// 注册不得 panic；POST /kb 为**无尾段**路径，必须命中转发 handler 而不是落到原生列表 handler。
func TestRegisterAIProxyRoutes_KbLevelRoutesAreForwarded(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	rg := engine.Group("/api/v1")
	crud := func(c *gin.Context) {
		c.Header("X-Crud", "1")
		c.Status(http.StatusOK)
	}
	// 与 ai_kb.go 现存原生路由同形
	rg.GET("/kb", crud)
	rg.GET("/kb/:id", crud)
	rg.PUT("/kb/:id", crud)
	rg.GET("/kb/:id/documents", crud)
	rg.GET("/kb/documents/:id", crud)
	rg.GET("/kb/documents/:id/chunks", crud)
	rg.POST("/kb/:id/retrieve/test-sets", crud)
	rg.GET("/kb/:id/retrieve/test-sets", crud)
	rg.GET("/kb/:id/chunks/low-quality", crud)

	// 上游指向必然拒绝连接的端口：转发 handler 命中时返回 C0001 信封，可与原生 handler 区分
	client, err := aiclient.New(options.AI{ServiceURL: "http://127.0.0.1:1", Timeout: 1, ConnectTimeout: 1})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	RegisterAIProxyRoutes(rg, api.NewAIProxyApi(client))

	nativeHits := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/v1/kb"},
		{http.MethodGet, "/api/v1/kb/5"},
		{http.MethodPut, "/api/v1/kb/5"},
		{http.MethodGet, "/api/v1/kb/5/documents"},
		{http.MethodGet, "/api/v1/kb/documents/5"},
		{http.MethodGet, "/api/v1/kb/5/retrieve/test-sets"},
		{http.MethodGet, "/api/v1/kb/5/chunks/low-quality"},
	}
	for _, route := range nativeHits {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if rec.Header().Get("X-Crud") != "1" {
			t.Errorf("%s %s 应仍由原生 handler 处理（不得被转发截胡），实际 status=%d", route.method, route.path, rec.Code)
		}
	}

	forwardedHits := []struct {
		method string
		path   string
	}{
		{http.MethodPost, "/api/v1/kb"},
		{http.MethodGet, "/api/v1/kb/7/index-stats"},
		{http.MethodDelete, "/api/v1/kb/7"},
		{http.MethodDelete, "/api/v1/kb/documents/7"},
	}
	for _, route := range forwardedHits {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if !strings.Contains(rec.Body.String(), common.CALL_THIRD_PARTY_SERVICE_ERROR.Code) {
			t.Errorf("%s %s 应由转发 handler 处理（预期上游不可达信封），实际 status=%d body=%s",
				route.method, route.path, rec.Code, rec.Body.String())
		}
	}
}

// TestRegisterAICompatRoutes_CoexistsWithMessageModule 覆盖真实 app 的路由共存形态：
// 消息模块在 POST /api/v1/messages/send 上已有处理函数，兼容 API 的 POST /api/v1/messages
// 挂在同一节点上，gin 允许"节点处理函数 + 子路径"共存，注册不得 panic。
func TestRegisterAICompatRoutes_CoexistsWithMessageModule(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.POST("/api/v1/messages/send", func(*gin.Context) {})

	RegisterAICompatRoutes(engine, newTestProxyApi(t))

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/v1/messages/send", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("消息模块的 /api/v1/messages/send 应仍可命中，实际 %d", rec.Code)
	}
}

func TestRegisterAICompatRoutes_RegistersRootEndpoints(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	RegisterAICompatRoutes(engine, newTestProxyApi(t))

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	for _, path := range []string{
		"POST /api/v1/chat/completions",
		"POST /api/v1/messages",
		"GET /api/v1/models",
		"POST /a2a",
		"GET /.well-known/agent.json",
	} {
		if !registered[path] {
			t.Errorf("缺少兼容协议/A2A 入口路由: %s", path)
		}
	}
}
