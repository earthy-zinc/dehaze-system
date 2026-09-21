package router

import (
	"net/http"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestRegisterAiKbRoutes KB A 类路由注册不得 panic，且路径参数统一 :id
// （与 go-proxy 的 KB 写路径转发路由同节点相遇时通配符名必须一致）。
func TestRegisterAiKbRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")
	require.NotPanics(t, func() {
		RegisterAiKbRoutes(v1, api.NewAiKbApi(nil))
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	expected := []string{
		http.MethodGet + " /api/v1/kb",
		http.MethodGet + " /api/v1/kb/:id",
		http.MethodPut + " /api/v1/kb/:id",
		http.MethodGet + " /api/v1/kb/:id/documents",
		http.MethodGet + " /api/v1/kb/documents/:id",
		http.MethodGet + " /api/v1/kb/documents/:id/chunks",
		http.MethodPost + " /api/v1/kb/:id/retrieve/test-sets",
		http.MethodGet + " /api/v1/kb/:id/retrieve/test-sets",
		http.MethodGet + " /api/v1/kb/:id/chunks/low-quality",
	}
	for _, path := range expected {
		require.True(t, registered[path], "路由未注册: %s", path)
	}
}
