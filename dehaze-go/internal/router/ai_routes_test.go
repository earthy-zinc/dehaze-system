package router

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAIRoutesRegistrationHasNoConflict A 类原生路由与 B 类转发路由共用一个 /ai 节点树：
// gin 在同一路径位置出现不一致的通配符名或重复注册会 panic，本用例以注册过程（含 panic）
// 作为回归护栏。
//
// 注意 POST /ai/agents/:id/publish 走 B 类转发（发布门禁需执行回归评测，行为事实源在 python），
// 若有人恢复 A 类原生注册，本用例会因同「方法+路径」重复注册而 panic。
func TestAIRoutesRegistrationHasNoConflict(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	protected := engine.Group("/api/v1")

	require.NotPanics(t, func() {
		RegisterAIProxyRoutes(protected, api.NewAIProxyApi(nil))
		RegisterAiConversationRoutes(protected, api.NewAiConversationApi(nil, nil, nil, nil))
		RegisterAiMemoryRoutes(protected, api.NewAiMemoryApi(nil))
		RegisterAiAgentRoutes(protected, api.NewAiAgentApi(nil, nil, nil))
		RegisterAiEvalRoutes(protected, api.NewAiEvalApi(nil, nil))
		RegisterAiScheduleRoutes(protected, api.NewAiScheduleApi(nil), api.NewAiUsageApi(nil))
	})

	routes := engine.Routes()
	assert.Greater(t, len(routes), 60, "AI 域路由数量异常")

	// 关键路径必须注册成功且方法正确
	expected := map[string]string{
		"POST /api/v1/ai/conversations":                           "",
		"GET /api/v1/ai/conversations":                            "",
		"GET /api/v1/ai/conversations/:id":                        "",
		"GET /api/v1/ai/conversations/:id/messages":               "",
		"PUT /api/v1/ai/conversations/:id/pin":                    "",
		"GET /api/v1/ai/memories":                                 "",
		"POST /api/v1/ai/memories/:id/unarchive":                  "",
		"GET /api/v1/ai/agents":                                   "",
		"GET /api/v1/ai/agents/:id/versions/diff":                 "",
		"POST /api/v1/ai/agents/:id/versions/:versionNo/rollback": "",
		"POST /api/v1/ai/a2a/endpoints":                           "",
		"GET /api/v1/ai/agents/:id/eval/datasets":                 "",
		"GET /api/v1/ai/eval-center/overview":                     "",
		"GET /api/v1/ai/scheduled-tasks/next-times":               "",
		"GET /api/v1/ai/usage/stats":                              "",
		"GET /api/v1/ai/agents/config-defaults":                   "",
	}
	registered := map[string]struct{}{}
	for _, route := range routes {
		registered[route.Method+" "+route.Path] = struct{}{}
	}
	for path := range expected {
		assert.Contains(t, registered, path, "路由缺失")
	}
}
