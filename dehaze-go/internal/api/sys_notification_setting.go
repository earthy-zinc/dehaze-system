package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type NotificationSettingApi struct {
	settingService msgservice.INotificationSettingService
}

func NewNotificationSettingApi(settingService msgservice.INotificationSettingService) *NotificationSettingApi {
	return &NotificationSettingApi{settingService: settingService}
}

func (api *NotificationSettingApi) Get(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.settingService.Get(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *NotificationSettingApi) Update(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.NotificationSettingForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.settingService.Update(c.Request.Context(), userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新成功", c)
}
