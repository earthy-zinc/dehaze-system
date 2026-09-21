package router

import (
	"net/http"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// aiProxyRoute 单条转发路由
type aiProxyRoute struct {
	method  string
	path    string
	handler gin.HandlerFunc
}

// RegisterAIProxyRoutes 注册 AI 域 B 类端点（强依赖 deepagents/LLM/ES）。
// 这些端点在 Go 侧只有一层 handler，报文原样转发给 dehaze-python（唯一行为事实源），
// 路径与 python 路由逐条对齐（dehaze-python/app/router）。
//
// 路径参数统一用 :id —— gin 在同一路径位置只允许一个通配符名，与其它模块路由共用节点时
// 名字必须完全一致，否则启动 panic（wildcard segment conflicts）。
func RegisterAIProxyRoutes(rg *gin.RouterGroup, api *api.AIProxyApi) {
	routes := []aiProxyRoute{
		// AI 对话：流式推理与中断控制（行为在 python 的 deepagents 推理循环内）
		{http.MethodPost, "/ai/conversations/:id/messages", api.ForwardSSE},
		{http.MethodPost, "/ai/messages/:id/regenerate", api.ForwardSSE},
		{http.MethodPost, "/ai/messages/:id/resume", api.ForwardSSE},
		{http.MethodPut, "/ai/messages/:id", api.ForwardSSE},
		{http.MethodPost, "/ai/messages/:id/stop", api.ForwardJSON},
		{http.MethodGet, "/ai/conversations/:id/messages/stream/:stream_id", api.ForwardSSE},

		// 真实调用下游能力的可用性测试
		{http.MethodPost, "/ai/agents/:id/test", api.ForwardJSON},
		// 发布门禁需执行回归评测（分钟级真实推理），与 rollback 不同——rollback 只重放快照，保持原生
		{http.MethodPost, "/ai/agents/:id/publish", api.ForwardJSON},
		{http.MethodPost, "/ai/skills/:id/test", api.ForwardJSON},
		// SKILL 压缩包上传：zip 解包 + manifest 解析 + 脚本命令策略校验 + 落盘，全在 python
		{http.MethodPost, "/ai/skills/upload", api.ForwardMultipart},
		{http.MethodPost, "/ai/models/:id/test", api.ForwardJSON},
		{http.MethodPost, "/ai/providers/:id/test-connection", api.ForwardJSON},

		// MCP Server 探测与工具试调用（需真实 MCP 会话）
		{http.MethodGet, "/ai/mcp/servers/:id/health", api.ForwardJSON},
		{http.MethodGet, "/ai/mcp/servers/:id/tools", api.ForwardJSON},
		{http.MethodPost, "/ai/mcp/servers/:id/tools/test", api.ForwardJSON},

		// 智能体评测：异步任务派发与进度查询
		{http.MethodPost, "/ai/agents/:id/eval/runs", api.ForwardJSON},
		{http.MethodGet, "/ai/agents/:id/eval/runs", api.ForwardJSON},
		{http.MethodGet, "/ai/agents/:id/eval/tasks/:task_id", api.ForwardJSON},

		// 定时任务手动触发（真实执行 Agent）
		{http.MethodPost, "/ai/scheduled-tasks/:id/run", api.ForwardJSON},

		// 知识库库级创建/删除/索引统计：依赖 ES 索引生命周期（python ensure_kb_index /
		// delete_kb_index / _stats），Go 原生实现会出现「库建了但 ES 索引不存在」的缺口，故一律转发。
		// 注意 POST /kb 为无尾段路径，与 /kb/:id/... 共节点，注册不得 panic
		{http.MethodPost, "/kb", api.ForwardJSON},
		{http.MethodGet, "/kb/:id/index-stats", api.ForwardJSON},
		{http.MethodDelete, "/kb/:id", api.ForwardJSON},

		// 知识库：文档入库（解析 + 向量化 + ES 写入）与检索召回调试。
		// 所有触发 python 侧 _process_document_guarded 的写路径都必须转发，
		// 否则 Go 侧只落库文档元数据、processingStatus 永远停在 pending（永不入库）。
		// 注意：文档上传是 JSON（body 传 fileId，文件本体走 /api/v1/files），不是 multipart
		{http.MethodPost, "/kb/:id/documents", api.ForwardJSON},
		{http.MethodPost, "/kb/:id/documents/batch", api.ForwardJSON},
		{http.MethodPost, "/kb/:id/documents/text", api.ForwardJSON},
		{http.MethodPost, "/kb/:id/documents/import-url", api.ForwardJSON},
		{http.MethodPut, "/kb/documents/:id", api.ForwardJSON},
		{http.MethodDelete, "/kb/documents/:id", api.ForwardJSON},
		{http.MethodPost, "/kb/documents/:id/reprocess", api.ForwardJSON},
		{http.MethodPost, "/kb/documents/chunks/preview", api.ForwardJSON},
		{http.MethodPost, "/kb/search", api.ForwardJSON},
		{http.MethodPost, "/kb/:id/retrieve/test", api.ForwardJSON},
		{http.MethodPost, "/kb/:id/retrieve/test-sets/:test_set_id/run", api.ForwardJSON},

		// 外部 A2A 端点 Agent Card 刷新（真实访问对端）
		{http.MethodPost, "/ai/a2a/endpoints/:id/refresh-card", api.ForwardJSON},

		// A2A 协议 JSON-RPC 入口（Agent Card 发现端点由 Go 原生实现，见 ai_a2a.go，勿重复注册）
		{http.MethodPost, "/ai/agents/:id/a2a", api.ForwardSSE},
	}

	for _, route := range routes {
		rg.Handle(route.method, route.path, route.handler)
	}
}

// RegisterAICompatRoutes 注册 OpenAI/Claude 兼容 API 与 A2A 标准入口。
// 这些端点必须挂在根引擎（不过 AuthMiddleware）：python 的 ApiKeyAuthMiddleware 是全局中间件，
// Claude 协议走 x-api-key、A2A 标准入口的鉴权也只有 python 认，Go 侧先拦会把这些调用方一律 401；
// 鉴权事实源保持 python 一处，Go 只透传 Authorization/Cookie/X-Session-Id/x-api-key。
func RegisterAICompatRoutes(engine *gin.Engine, api *api.AIProxyApi) {
	engine.POST("/api/v1/chat/completions", api.ForwardSSE)
	engine.POST("/api/v1/messages", api.ForwardSSE)
	engine.GET("/api/v1/models", api.ForwardJSON)
	engine.POST("/a2a", api.ForwardSSE)
	engine.GET("/.well-known/agent.json", api.ForwardJSON)
}
