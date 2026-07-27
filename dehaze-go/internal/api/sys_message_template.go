package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type MessageTemplateApi struct {
	tplService msgservice.IMessageTemplateService
}

func NewMessageTemplateApi(tplService msgservice.IMessageTemplateService) *MessageTemplateApi {
	return &MessageTemplateApi{tplService: tplService}
}

func (api *MessageTemplateApi) GetPage(c *gin.Context) {
	q := &query.MessageTemplateQuery{
		Name:     c.Query("name"),
		Type:     c.Query("type"),
		PageNum:  1,
		PageSize: 20,
	}
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageSize = n
		}
	}
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Status = n
		}
	}

	result, err := api.tplService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageTemplateApi) GetDetail(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模板ID格式不正确"))
		return
	}

	result, err := api.tplService.GetDetail(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageTemplateApi) Update(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模板ID格式不正确"))
		return
	}

	var form bo.MessageTemplateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.tplService.Update(c.Request.Context(), id, userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新成功", c)
}
