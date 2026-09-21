package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiProviderApi AI 模型供应商与 API Key 管理接口
type AiProviderApi struct {
	service *aiservice.ProviderService
}

func NewAiProviderApi(service *aiservice.ProviderService) *AiProviderApi {
	return &AiProviderApi{service: service}
}

// ListProviders 供应商分页列表
func (a *AiProviderApi) ListProviders(c *gin.Context) {
	// 分页字段（AiPageQuery）已 form:"-"，分页只由本 helper 解析：非数字/越界一律 A0400，
	// 对齐 python ProviderPageQuery(BasePageQuery)。
	pageNum, pageSize, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	var query bo.ProviderQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.PageNum, query.PageSize = pageNum, pageSize
	result, err := a.service.ListProviders(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListEnabledProviders 启用供应商精简列表
func (a *AiProviderApi) ListEnabledProviders(c *gin.Context) {
	result, err := a.service.ListEnabledProviders(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateProvider 新增供应商
func (a *AiProviderApi) CreateProvider(c *gin.Context) {
	var form bo.ProviderCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.CreateProvider(c.Request.Context(), &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateProvider 更新供应商
func (a *AiProviderApi) UpdateProvider(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.ProviderUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateProvider(c.Request.Context(), providerID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteProvider 删除供应商
func (a *AiProviderApi) DeleteProvider(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.DeleteProvider(c.Request.Context(), providerID, security.GetUserID(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// ListProviderKeys 供应商 API Key 列表
func (a *AiProviderApi) ListProviderKeys(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.ListProviderKeys(c.Request.Context(), providerID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateProviderKey 新增 API Key
func (a *AiProviderApi) CreateProviderKey(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.ProviderKeyCreateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.CreateProviderKey(c.Request.Context(), providerID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateProviderKey 更新 API Key
func (a *AiProviderApi) UpdateProviderKey(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	keyID, err := lastAiPathID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.ProviderKeyUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateProviderKey(c.Request.Context(), providerID, keyID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteProviderKey 删除 API Key
func (a *AiProviderApi) DeleteProviderKey(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	keyID, err := lastAiPathID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.DeleteProviderKey(c.Request.Context(), providerID, keyID); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// CloseProviderCircuit 手动解除供应商熔断
func (a *AiProviderApi) CloseProviderCircuit(c *gin.Context) {
	providerID, err := parseAiPathID(c, "id")
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.CloseProviderCircuit(c.Request.Context(), providerID); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}
