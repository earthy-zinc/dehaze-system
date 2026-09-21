package router

import (
	"net/http"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterVoiceProxyRoutes 注册语音域端点（ASR / TTS / 热词 / 引擎注册表）。
// 与 AI 域 B 类端点同属「行为唯一实现在 dehaze-python」的转发端点：ASR 走本地 FunASR、
// TTS 走本地 Piper 引擎（均进程内懒加载，模型随 python 分发），Go 侧不做业务实现，
// 只做身份透传与报文搬运；权限（voice:hotword:edit / voice:service:monitor /
// voice:engine:manage）一律由 python 终审。
// 路径与 dehaze-python/app/router/voice.py、voice_admin.py 逐条对齐，增删接口时同步维护。
//
// 路径参数统一用 :id —— gin 在同一路径位置只允许一个通配符名，与其它模块路由共用节点时
// 名字必须完全一致，否则启动 panic（wildcard segment conflicts）。
func RegisterVoiceProxyRoutes(rg *gin.RouterGroup, api *api.AIProxyApi) {
	routes := []aiProxyRoute{
		// ASR：流式会话创建 / 结果查询 / 离线识别（音频文件 multipart 直传，不落盘）
		{http.MethodPost, "/voice/asr/stream-session", api.ForwardJSON},
		{http.MethodGet, "/voice/asr/result/:id", api.ForwardJSON},
		{http.MethodPost, "/voice/asr/offline", api.ForwardMultipart},

		// TTS：合成返回 audioUrl、音色列表、缓存音频下载（响应为音频二进制，原样透传不解包）
		{http.MethodPost, "/voice/tts", api.ForwardJSON},
		{http.MethodGet, "/voice/tts/voices", api.ForwardJSON},
		{http.MethodGet, "/voice/tts/audio/:id", api.ForwardJSON},

		// 热词：用户级 + 全局（全局写操作的管理员校验在 python）
		{http.MethodGet, "/voice/hotwords", api.ForwardJSON},
		{http.MethodPost, "/voice/hotwords", api.ForwardJSON},
		{http.MethodDelete, "/voice/hotwords/:id", api.ForwardJSON},
		{http.MethodGet, "/voice/hotwords/global", api.ForwardJSON},
		{http.MethodPost, "/voice/hotwords/global", api.ForwardJSON},
		{http.MethodDelete, "/voice/hotwords/global/:id", api.ForwardJSON},

		// 服务状态监控：ASR/TTS 引擎状态由 python 进程内实例上报
		{http.MethodGet, "/voice/service/status", api.ForwardJSON},

		// 引擎注册表管理：provider / key / model
		{http.MethodGet, "/voice/providers", api.ForwardJSON},
		{http.MethodGet, "/voice/providers/enabled", api.ForwardJSON},
		{http.MethodPost, "/voice/providers", api.ForwardJSON},
		{http.MethodPut, "/voice/providers/:id", api.ForwardJSON},
		{http.MethodDelete, "/voice/providers/:id", api.ForwardJSON},
		// 连通性测试要真实访问引擎（云端引擎走外网、local 引擎走进程内实例）
		{http.MethodPost, "/voice/providers/:id/test-connection", api.ForwardJSON},
		{http.MethodGet, "/voice/providers/:id/keys", api.ForwardJSON},
		{http.MethodPost, "/voice/providers/:id/keys", api.ForwardJSON},
		{http.MethodPut, "/voice/providers/:id/keys/:key_id", api.ForwardJSON},
		{http.MethodDelete, "/voice/providers/:id/keys/:key_id", api.ForwardJSON},
		{http.MethodGet, "/voice/models", api.ForwardJSON},
		{http.MethodPost, "/voice/models", api.ForwardJSON},
		{http.MethodPut, "/voice/models/:id", api.ForwardJSON},
		{http.MethodDelete, "/voice/models/:id", api.ForwardJSON},
	}

	for _, route := range routes {
		rg.Handle(route.method, route.path, route.handler)
	}
}

// RegisterVoiceWSProxyRoute 注册流式 ASR 的 WebSocket 转发（/ws/asr）。
// 挂在根引擎（不过 AuthMiddleware）：浏览器 WS 握手无法携带自定义头，登录会话凭证随
// query 的 sid 传递，鉴权与 ASR 会话归属校验统一由 python 完成
// （dehaze-python/app/service/voice/asr_service.handle_stream_websocket），
// Go 侧先拦只会把合法连接判成 401。
func RegisterVoiceWSProxyRoute(engine *gin.Engine, api *api.AIProxyApi) {
	engine.GET("/ws/asr", api.ForwardWebSocket)
}
