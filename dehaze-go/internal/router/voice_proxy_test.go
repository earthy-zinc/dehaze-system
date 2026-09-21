package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestRegisterVoiceProxyRoutes_RegistersWhitelist 语音域白名单必须与
// dehaze-python/app/router/voice.py、voice_admin.py 逐条对齐：漏注册会让 SDK 在前端拿到 404，
// 多注册则意味着 Go 侧凭空多出 python 不存在的端点。
func TestRegisterVoiceProxyRoutes_RegistersWhitelist(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	RegisterVoiceProxyRoutes(engine.Group("/api/v1"), newTestProxyApi(t))
	RegisterVoiceWSProxyRoute(engine, newTestProxyApi(t))

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}

	// 用户端（voice.py）
	want := []string{
		"POST /api/v1/voice/asr/stream-session",
		"GET /api/v1/voice/asr/result/:id",
		"POST /api/v1/voice/asr/offline",
		"POST /api/v1/voice/tts",
		"GET /api/v1/voice/tts/voices",
		"GET /api/v1/voice/tts/audio/:id",
		"GET /api/v1/voice/hotwords",
		"POST /api/v1/voice/hotwords",
		"DELETE /api/v1/voice/hotwords/:id",
		"GET /api/v1/voice/hotwords/global",
		"POST /api/v1/voice/hotwords/global",
		"DELETE /api/v1/voice/hotwords/global/:id",
		"GET /api/v1/voice/service/status",
		// 管理端（voice_admin.py）
		"GET /api/v1/voice/providers",
		"GET /api/v1/voice/providers/enabled",
		"POST /api/v1/voice/providers",
		"PUT /api/v1/voice/providers/:id",
		"DELETE /api/v1/voice/providers/:id",
		"POST /api/v1/voice/providers/:id/test-connection",
		"GET /api/v1/voice/providers/:id/keys",
		"POST /api/v1/voice/providers/:id/keys",
		"PUT /api/v1/voice/providers/:id/keys/:key_id",
		"DELETE /api/v1/voice/providers/:id/keys/:key_id",
		"GET /api/v1/voice/models",
		"POST /api/v1/voice/models",
		"PUT /api/v1/voice/models/:id",
		"DELETE /api/v1/voice/models/:id",
		// 流式 ASR WebSocket
		"GET /ws/asr",
	}
	for _, path := range want {
		if !registered[path] {
			t.Errorf("缺少语音域转发路由: %s", path)
		}
	}

	// 白名单外不做兜底转发
	outside := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/api/v1/voice/asr/stream-session"},
		{http.MethodPost, "/api/v1/voice/tts/voices"},
		{http.MethodGet, "/api/v1/voice/providers/1"},
		{http.MethodPatch, "/api/v1/voice/models/1"},
	}
	for _, route := range outside {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(route.method, route.path, nil))
		if rec.Code != http.StatusNotFound {
			t.Errorf("%s %s 应 404（白名单外不转发），实际 %d", route.method, route.path, rec.Code)
		}
	}
}

// TestRegisterVoiceProxyRoutes_CoexistsWithNativeRoutes 语音域路由与其它模块共用 /api/v1 组、
// 且 /ws/asr 与既有 /ws 同节点，注册不得 panic（gin 同一位置只允许一个通配符名）。
// 静态与参数兄弟（hotwords/global 与 hotwords/:id、providers/enabled 与 providers/:id/keys）
// 是本次注册最容易踩 panic 的形态，此处用真实注册锁定。
func TestRegisterVoiceProxyRoutes_CoexistsWithNativeRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.GET("/ws", func(c *gin.Context) { c.Status(http.StatusOK) })

	require.NotPanics(t, func() {
		RegisterVoiceProxyRoutes(engine.Group("/api/v1"), newTestProxyApi(t))
		RegisterVoiceWSProxyRoute(engine, newTestProxyApi(t))
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	// 原生 /ws 不能被 /ws/asr 注册挤掉
	if !registered["GET /ws"] || !registered["GET /ws/asr"] {
		t.Errorf("/ws 与 /ws/asr 应共存，实际路由: %v", registered)
	}
}
