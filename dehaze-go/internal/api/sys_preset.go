package api

import (
	"strconv"

	presetservice "github.com/earthyzinc/dehaze-go/internal/service/preset"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type SysPresetApi struct {
	service *presetservice.PresetService
}

func NewSysPresetApi(service *presetservice.PresetService) *SysPresetApi {
	return &SysPresetApi{service: service}
}

// ListPresets 获取预设列表
func (api *SysPresetApi) ListPresets(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var algorithmID int64
	if v := c.Query("algorithmId"); v != "" {
		algorithmID, err = strconv.ParseInt(v, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "algorithmId格式不正确"))
			return
		}
	}

	pageNum := 1
	if v := c.Query("pageNum"); v != "" {
		pageNum, err = strconv.Atoi(v)
		if err != nil || pageNum < 1 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "pageNum格式不正确"))
			return
		}
	}

	pageSize := 10
	if v := c.Query("pageSize"); v != "" {
		pageSize, err = strconv.Atoi(v)
		if err != nil || pageSize < 1 || pageSize > 100 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "pageSize格式不正确"))
			return
		}
	}

	var isSystem *bool
	if v := c.Query("isSystem"); v != "" {
		b := v == "true"
		isSystem = &b
	}

	result, err := api.service.ListPresets(ctx, algorithmID, userID, isSystem, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreatePreset 创建自定义预设
func (api *SysPresetApi) CreatePreset(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form presetservice.PresetForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.CreatePreset(ctx, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdatePreset 更新自定义预设
func (api *SysPresetApi) UpdatePreset(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form presetservice.PresetForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.UpdatePreset(ctx, id, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeletePreset 删除自定义预设
func (api *SysPresetApi) DeletePreset(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	if err := api.service.DeletePreset(ctx, id, userID); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}
