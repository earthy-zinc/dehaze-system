package api

import (
	"strconv"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiScheduleApi AI 对话定时任务（A 类，用户级无独立权限标识，走归属校验）。
type AiScheduleApi struct {
	schedules *aidomain.ScheduleService
}

// NewAiScheduleApi 构造 AiScheduleApi。
func NewAiScheduleApi(schedules *aidomain.ScheduleService) *AiScheduleApi {
	return &AiScheduleApi{schedules: schedules}
}

// CreateSchedule 创建定时任务。
func (a *AiScheduleApi) CreateSchedule(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form aidomain.ScheduleCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.schedules.Create(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListSchedules 定时任务列表。
func (a *AiScheduleApi) ListSchedules(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.schedules.List(c.Request.Context(), userID, pageNum, pageSize, c.Query("keyword"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// PreviewNextTimes Cron 解释与下次执行时间预览。
func (a *AiScheduleApi) PreviewNextTimes(c *gin.Context) {
	cron := c.Query("cron")
	if cron == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "cron 不能为空"))
		return
	}
	count := 5
	if raw := c.Query("count"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 20 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "count 取值非法"))
			return
		}
		count = parsed
	}
	result, err := a.schedules.PreviewNextTimes(c.Request.Context(), cron, count)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetSchedule 定时任务详情。
func (a *AiScheduleApi) GetSchedule(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.schedules.GetDetail(c.Request.Context(), userID, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateSchedule 更新定时任务。
func (a *AiScheduleApi) UpdateSchedule(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.ScheduleUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.schedules.Update(c.Request.Context(), userID, id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SetScheduleStatus 启停定时任务。
func (a *AiScheduleApi) SetScheduleStatus(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form struct {
		Enabled *int `json:"enabled"`
	}
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if form.Enabled == nil || (*form.Enabled != 0 && *form.Enabled != 1) {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "enabled 取值非法"))
		return
	}
	if err := a.schedules.SetEnabled(c.Request.Context(), userID, id, *form.Enabled); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// DeleteSchedule 删除定时任务。
func (a *AiScheduleApi) DeleteSchedule(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := a.schedules.Delete(c.Request.Context(), userID, id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// ListRunHistory 执行历史。
func (a *AiScheduleApi) ListRunHistory(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.schedules.ListHistory(c.Request.Context(), userID, id, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}
