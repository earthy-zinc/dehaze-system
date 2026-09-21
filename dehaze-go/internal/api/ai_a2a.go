package api

import (
	"strings"

	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

// AiA2AApi A2A 协议 Agent Card 发现端点（A 类）
type AiA2AApi struct {
	service *aiservice.A2AService
}

func NewAiA2AApi(service *aiservice.A2AService) *AiA2AApi {
	return &AiA2AApi{service: service}
}

// AgentCard 动态生成 Agent Card（对齐 python GET {agent}/a2a/.well-known/agent.json）
func (a *AiA2AApi) AgentCard(c *gin.Context) {
	agentID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	card, err := a.service.AgentCard(c.Request.Context(), agentID, requestBaseURL(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(card, c)
}

// requestBaseURL 请求基地址（scheme://host），优先取反向代理透传的 X-Forwarded-Proto
func requestBaseURL(c *gin.Context) string {
	scheme := "http"
	if proto := c.GetHeader("X-Forwarded-Proto"); proto != "" {
		scheme = strings.TrimSpace(strings.Split(proto, ",")[0])
	} else if c.Request.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + c.Request.Host
}
