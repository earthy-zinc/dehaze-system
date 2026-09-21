package api

import (
	"github.com/earthyzinc/dehaze-go/pkg/aiclient"
	"github.com/gin-gonic/gin"
)

// AIProxyApi python 能力端点转发 API（AI 域 B 类 + 语音域：强依赖 deepagents/LLM/ES、
// 本地 ASR/TTS 引擎，行为唯一实现在 dehaze-python）
type AIProxyApi struct {
	client *aiclient.Client
}

func NewAIProxyApi(client *aiclient.Client) *AIProxyApi {
	return &AIProxyApi{client: client}
}

// ForwardJSON 普通 JSON 端点透传
func (api *AIProxyApi) ForwardJSON(c *gin.Context) {
	api.client.ForwardJSON(c)
}

// ForwardSSE 流式端点透传（含可能返回 SSE 的 a2a JSON-RPC）
func (api *AIProxyApi) ForwardSSE(c *gin.Context) {
	api.client.ForwardSSE(c)
}

// ForwardMultipart 文件上传端点透传
func (api *AIProxyApi) ForwardMultipart(c *gin.Context) {
	api.client.ForwardMultipart(c)
}

// ForwardWebSocket WebSocket 端点双向透传（语音流式 ASR）
func (api *AIProxyApi) ForwardWebSocket(c *gin.Context) {
	api.client.ForwardWebSocket(c)
}
