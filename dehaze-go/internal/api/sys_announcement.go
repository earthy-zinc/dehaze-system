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

type AnnouncementApi struct {
	annService msgservice.IAnnouncementService
}

func NewAnnouncementApi(annService msgservice.IAnnouncementService) *AnnouncementApi {
	return &AnnouncementApi{annService: annService}
}

func (api *AnnouncementApi) GetPage(c *gin.Context) {
	q := &query.AnnouncementQuery{
		Title:    c.Query("title"),
		Type:     c.Query("type"),
		PageNum:  1,
		PageSize: 10,
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

	result, err := api.annService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *AnnouncementApi) Create(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.AnnouncementForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.annService.Create(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "创建成功", c)
}

func (api *AnnouncementApi) GetDetail(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "公告ID格式不正确"))
		return
	}

	result, err := api.annService.GetDetail(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *AnnouncementApi) Update(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "公告ID格式不正确"))
		return
	}

	var form bo.AnnouncementForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.annService.Update(c.Request.Context(), id, userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新成功", c)
}

func (api *AnnouncementApi) Delete(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "公告ID格式不正确"))
		return
	}

	if err := api.annService.Delete(c.Request.Context(), id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除成功", c)
}

func (api *AnnouncementApi) Send(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "公告ID格式不正确"))
		return
	}

	result, err := api.annService.Send(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "发送成功", c)
}

func (api *AnnouncementApi) Cancel(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "公告ID格式不正确"))
		return
	}

	if err := api.annService.Cancel(c.Request.Context(), id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("取消成功", c)
}
