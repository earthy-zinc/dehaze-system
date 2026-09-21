package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiMcpApi 外部 MCP Server 管理接口（F-M08-006 §2.6.13）
type AiMcpApi struct {
	service *aiservice.McpServerService
}

func NewAiMcpApi(service *aiservice.McpServerService) *AiMcpApi {
	return &AiMcpApi{service: service}
}

// ListServers MCP Server 分页列表
func (a *AiMcpApi) ListServers(c *gin.Context) {
	// 分页字段（AiPageQuery）已 form:"-"，分页只由本 helper 解析：非数字/越界一律 A0400，
	// 对齐 python McpServerQuery(BasePageQuery)。
	pageNum, pageSize, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	var query bo.McpServerQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.PageNum, query.PageSize = pageNum, pageSize
	result, err := a.service.ListServers(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateServer 注册外部 MCP Server
func (a *AiMcpApi) CreateServer(c *gin.Context) {
	var form bo.McpServerCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.CreateServer(c.Request.Context(), &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetServer Server 详情
func (a *AiMcpApi) GetServer(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.GetServer(c.Request.Context(), serverID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateServer 更新 Server
func (a *AiMcpApi) UpdateServer(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.McpServerUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateServer(c.Request.Context(), serverID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteServer 删除 Server
func (a *AiMcpApi) DeleteServer(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.DeleteServer(c.Request.Context(), serverID, security.GetUserID(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// SwitchServerStatus 启停 Server
func (a *AiMcpApi) SwitchServerStatus(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.McpServerStatusForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.SwitchServerStatus(c.Request.Context(), serverID, *form.Status, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListNamespaces 命名空间列表
func (a *AiMcpApi) ListNamespaces(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.ListNamespaces(c.Request.Context(), serverID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateNamespaces 配置命名空间（覆盖式更新）
func (a *AiMcpApi) UpdateNamespaces(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var forms []bo.McpNamespaceForm
	if bindErr := c.ShouldBindJSON(&forms); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateNamespaces(c.Request.Context(), serverID, forms, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateCredentials 配置外部服务凭据（AES 加密存储，不回显）
func (a *AiMcpApi) UpdateCredentials(c *gin.Context) {
	serverID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.McpCredentialForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	if err := a.service.UpdateCredentials(c.Request.Context(), serverID, &form, security.GetUserID(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// GetMarket MCP 市场目录
func (a *AiMcpApi) GetMarket(c *gin.Context) {
	result, err := a.service.GetMarket(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListCalls MCP 调用审计
func (a *AiMcpApi) ListCalls(c *gin.Context) {
	// 分页校验先于绑定，对齐 python McpCallQuery(BasePageQuery) 的 A0400。
	pageNum, pageSize, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	var query bo.McpCallQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.PageNum, query.PageSize = pageNum, pageSize
	result, err := a.service.ListCalls(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}
