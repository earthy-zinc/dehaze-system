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

type MessageApi struct {
	msgService msgservice.IMessageService
}

func NewMessageApi(msgService msgservice.IMessageService) *MessageApi {
	return &MessageApi{msgService: msgService}
}

func (api *MessageApi) GetPage(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	q := &query.MessageQuery{
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
	if v := c.Query("readStatus"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.ReadStatus = &n
		}
	}

	result, err := api.msgService.GetPage(c.Request.Context(), userID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageApi) GetUnreadCount(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.msgService.GetUnreadCount(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageApi) GetDetail(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "消息ID格式不正确"))
		return
	}

	result, err := api.msgService.GetDetail(c.Request.Context(), id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageApi) MarkRead(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "消息ID格式不正确"))
		return
	}

	if err := api.msgService.MarkRead(c.Request.Context(), id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("标记已读成功", c)
}

func (api *MessageApi) MarkAllRead(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	msgType := c.Query("type")
	result, err := api.msgService.MarkAllRead(c.Request.Context(), userID, msgType)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "全部标记已读成功", c)
}

func (api *MessageApi) Delete(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	ids, err := parseIDsFromCSV(c.Param("ids"))
	if err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.msgService.Delete(c.Request.Context(), ids, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除成功", c)
}

func (api *MessageApi) Search(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	q := &query.MessageSearchQuery{
		Keyword:  c.Query("keyword"),
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

	result, err := api.msgService.Search(c.Request.Context(), userID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MessageApi) Send(c *gin.Context) {
	var form bo.MessageSendForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.msgService.Send(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "发送成功", c)
}
