package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiUserPreferenceApi 用户偏好默认模型接口（登录即可，无权限码）
type AiUserPreferenceApi struct {
	service *aiservice.UserPreferenceService
}

func NewAiUserPreferenceApi(service *aiservice.UserPreferenceService) *AiUserPreferenceApi {
	return &AiUserPreferenceApi{service: service}
}

// Get 查询偏好默认模型（未设置返回 defaultModel=null）
func (a *AiUserPreferenceApi) Get(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.Get(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// Update 设置/清除偏好默认模型（defaultModel 为 null 或缺失即清除）
func (a *AiUserPreferenceApi) Update(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.UserPreferenceForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.Update(c.Request.Context(), userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}
