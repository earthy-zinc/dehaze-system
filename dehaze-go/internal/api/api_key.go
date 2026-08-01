package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	apikeyservice "github.com/earthyzinc/dehaze-go/internal/service/api_key"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type ApiKeyApi struct {
	apiKeyService apikeyservice.IApiKeyService
}

func NewApiKeyApi(apiKeyService apikeyservice.IApiKeyService) *ApiKeyApi {
	return &ApiKeyApi{
		apiKeyService: apiKeyService,
	}
}

func (a *ApiKeyApi) CreateApiKey(c *gin.Context) {
	userID := security.GetUserID(c)
	if userID == 0 {
		_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或登录已过期"))
		return
	}

	var req dto.ApiKeyCreateRequest
	if err := c.ShouldBind(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := a.apiKeyService.CreateApiKey(c.Request.Context(), userID, &req)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

func (a *ApiKeyApi) ListApiKeys(c *gin.Context) {
	userID := security.GetUserID(c)
	if userID == 0 {
		_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或登录已过期"))
		return
	}

	results, err := a.apiKeyService.ListApiKeys(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(results, c)
}

func (a *ApiKeyApi) DeleteApiKey(c *gin.Context) {
	userID := security.GetUserID(c)
	if userID == 0 {
		_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或登录已过期"))
		return
	}

	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "无效的ID"))
		return
	}

	if err := a.apiKeyService.Revoke(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage(common.SUCCESS.Msg, c)
}
