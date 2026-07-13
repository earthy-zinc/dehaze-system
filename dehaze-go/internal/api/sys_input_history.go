package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	ihservice "github.com/earthyzinc/dehaze-go/internal/service/input_history"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

// SysInputHistoryApi 图像输入历史记录 API
type SysInputHistoryApi struct {
	service *ihservice.InputHistoryService
}

func NewSysInputHistoryApi(service *ihservice.InputHistoryService) *SysInputHistoryApi {
	return &SysInputHistoryApi{service: service}
}

// ListHistory 分页查询历史记录
func (api *SysInputHistoryApi) ListHistory(c *gin.Context) {
	ctx := c.Request.Context()
	userID := getCurrentUserID(c)
	pageNum, pageSize := getPageParams(c)
	inputSource := c.Query("inputSource")
	keyword := c.Query("keywords")
	favoriteOnly := c.Query("favoriteOnly") == "true"

	// status 处理状态筛选（1=成功，2=失败，3=处理中），0 表示不筛选
	status := 0
	if statusStr := c.Query("status"); statusStr != "" {
		if n, err := strconv.Atoi(statusStr); err == nil {
			status = n
		}
	}

	result, err := api.service.GetPage(ctx, userID, pageNum, pageSize, inputSource, keyword, favoriteOnly, status)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetHistory 历史记录详情
func (api *SysInputHistoryApi) GetHistory(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}
	history, err := api.service.GetByID(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(history, c)
}

// CreateHistory 创建历史记录
func (api *SysInputHistoryApi) CreateHistory(c *gin.Context) {
	ctx := c.Request.Context()
	userID := getCurrentUserID(c)

	var history model.SysInputHistory
	if err := c.ShouldBindJSON(&history); err != nil {
		_ = c.Error(err)
		return
	}
	history.UserID = userID

	if err := api.service.Create(ctx, &history); err != nil {
		_ = c.Error(err)
		return
	}
	// 与 Java 后端一致，返回创建的历史记录 ID
	common.OkWithData(history.ID, c)
}

// UpdateHistory 更新历史记录
func (api *SysInputHistoryApi) UpdateHistory(c *gin.Context) {
	ctx := c.Request.Context()
	userID := getCurrentUserID(c)
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.service.Update(ctx, id, userID, updates); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新成功", c)
}

// DeleteHistory 删除单条历史记录
func (api *SysInputHistoryApi) DeleteHistory(c *gin.Context) {
	ctx := c.Request.Context()
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}
	if err := api.service.Delete(ctx, id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除成功", c)
}

// BatchDeleteHistory 批量删除历史记录
func (api *SysInputHistoryApi) BatchDeleteHistory(c *gin.Context) {
	ctx := c.Request.Context()
	userID := getCurrentUserID(c)

	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.service.BatchDelete(ctx, req.IDs, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(int64(len(req.IDs)), c)
}

// ClearHistory 清空历史记录
func (api *SysInputHistoryApi) ClearHistory(c *gin.Context) {
	ctx := c.Request.Context()
	userID := getCurrentUserID(c)

	count, err := api.service.ClearAll(ctx, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(count, c)
}

// SyncHistory 同步历史记录
func (api *SysInputHistoryApi) SyncHistory(c *gin.Context) {
	// Go后端无需从Python同步，直接返回0
	common.OkWithData(int64(0), c)
}
