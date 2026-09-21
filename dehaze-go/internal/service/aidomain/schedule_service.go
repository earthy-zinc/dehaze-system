package aidomain

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

const (
	// maxSchedulesPerUser 单用户定时任务上限
	maxSchedulesPerUser = 20
	// defaultScheduleTimezone 默认任务时区（与配额重置时区一致）
	defaultScheduleTimezone = "Asia/Shanghai"
)

// memberLevelValues 会员等级编码 → 数值等级（定时调度仅 VIP2+ 可用）。
var memberLevelValues = map[string]int{"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}

// ScheduleDetailVO 定时任务详情。
type ScheduleDetailVO struct {
	ID              int64           `json:"id"`
	UserID          int64           `json:"userId"`
	Name            string          `json:"name"`
	Cron            string          `json:"cron"`
	Timezone        string          `json:"timezone"`
	Input           json.RawMessage `json:"input,omitempty"`
	Output          json.RawMessage `json:"output,omitempty"`
	Enabled         int             `json:"enabled"`
	Status          int             `json:"status"`
	CircuitStreak   int             `json:"circuitStreak"`
	NextTriggerTime string          `json:"nextTriggerTime,omitempty"`
	CreateTime      string          `json:"createTime,omitempty"`
}

// RunSummaryVO 最近一次执行摘要。
type RunSummaryVO struct {
	Status         int      `json:"status"`
	SkipReason     string   `json:"skipReason,omitempty"`
	Credits        *float64 `json:"credits,omitempty"`
	DurationMs     *int     `json:"durationMs,omitempty"`
	ErrorMsg       string   `json:"errorMsg,omitempty"`
	ConversationID *int64   `json:"conversationId,omitempty"`
	CreateTime     string   `json:"createTime,omitempty"`
}

// ScheduleListItemVO 定时任务列表项（含最近执行摘要）。
type ScheduleListItemVO struct {
	ScheduleDetailVO
	LastRun *RunSummaryVO `json:"lastRun,omitempty"`
}

// RunHistoryVO 执行历史项。
type RunHistoryVO struct {
	ID             int64    `json:"id"`
	ScheduleID     int64    `json:"scheduleId"`
	Status         int      `json:"status"`
	SkipReason     string   `json:"skipReason,omitempty"`
	Credits        *float64 `json:"credits,omitempty"`
	DurationMs     *int     `json:"durationMs,omitempty"`
	ErrorMsg       string   `json:"errorMsg,omitempty"`
	ConversationID *int64   `json:"conversationId,omitempty"`
	RequestID      string   `json:"requestId,omitempty"`
	WindowStart    string   `json:"windowStart,omitempty"`
	CreateTime     string   `json:"createTime,omitempty"`
}

// NextTimesPreviewVO Cron 解释与下次执行时间预览。
type NextTimesPreviewVO struct {
	Description string   `json:"description"`
	NextTimes   []string `json:"nextTimes"`
}

// ScheduleCreateForm 创建定时任务表单。
type ScheduleCreateForm struct {
	Name     string         `json:"name"`
	Cron     string         `json:"cron"`
	Timezone string         `json:"timezone"`
	Input    map[string]any `json:"input"`
	Output   map[string]any `json:"output"`
}

// ScheduleUpdateForm 更新定时任务表单。
type ScheduleUpdateForm struct {
	Name     *string        `json:"name"`
	Cron     *string        `json:"cron"`
	Timezone *string        `json:"timezone"`
	Input    map[string]any `json:"input"`
	Output   map[string]any `json:"output"`
	Enabled  *int           `json:"enabled" binding:"omitempty,oneof=0 1"`
}

// ScheduleService 定时任务业务逻辑（任务执行本身为 B 类，由 python 承接）。
type ScheduleService struct {
	schedules *repo.ScheduleRepository
}

// NewScheduleService 构造 ScheduleService。
func NewScheduleService(schedules *repo.ScheduleRepository) *ScheduleService {
	return &ScheduleService{schedules: schedules}
}

// Create 创建定时任务（VIP2+、单用户上限、Cron 合法性校验）。
func (s *ScheduleService) Create(ctx context.Context, userID int64, form *ScheduleCreateForm) (*ScheduleDetailVO, error) {
	if form.Name == "" || form.Cron == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "任务名称与触发规则不能为空")
	}
	if err := s.ensureVip2(ctx, userID); err != nil {
		return nil, err
	}
	count, err := s.schedules.CountByUser(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询定时任务失败", err)
	}
	if count >= maxSchedulesPerUser {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "定时任务数量已达上限(20个)")
	}
	if err := validateTypedConfig(form.Input, []string{"fixed", "dynamic"}, "input"); err != nil {
		return nil, err
	}
	if err := validateTypedConfig(form.Output, []string{"message", "callback"}, "output"); err != nil {
		return nil, err
	}

	timezone := form.Timezone
	if timezone == "" {
		timezone = defaultScheduleTimezone
	}
	cron, err := normalizeCron(form.Cron)
	if err != nil {
		return nil, err
	}
	nextTrigger, err := computeNextTrigger(cron, timezone)
	if err != nil {
		return nil, err
	}
	task := &model.SysAiSchedule{
		UserID:          userID,
		Name:            form.Name,
		Cron:            cron,
		Timezone:        timezone,
		Input:           marshalJSON(form.Input),
		Output:          marshalJSON(form.Output),
		Enabled:         1,
		Status:          1,
		NextTriggerTime: &nextTrigger,
	}
	if err := s.schedules.Create(ctx, task); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建定时任务失败", err)
	}
	return toScheduleVO(task), nil
}

// List 定时任务列表（含最近执行摘要）。
func (s *ScheduleService) List(ctx context.Context, userID int64, page, size int, keyword string) (*vo.PageResult[ScheduleListItemVO], error) {
	items, total, err := s.schedules.PaginateByUser(ctx, userID, page, size, keyword)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询定时任务失败", err)
	}
	ids := make([]int64, 0, len(items))
	for _, item := range items {
		ids = append(ids, item.ID)
	}
	latest, err := s.schedules.LatestRunsByScheduleIDs(ctx, ids)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询执行历史失败", err)
	}
	result := make([]ScheduleListItemVO, 0, len(items))
	for i := range items {
		entry := ScheduleListItemVO{ScheduleDetailVO: *toScheduleVO(&items[i])}
		if run, ok := latest[items[i].ID]; ok {
			entry.LastRun = &RunSummaryVO{
				Status:         run.Status,
				SkipReason:     derefString(run.SkipReason),
				Credits:        run.Credits,
				DurationMs:     run.DurationMs,
				ErrorMsg:       derefString(run.ErrorMsg),
				ConversationID: run.ConversationID,
				CreateTime:     formatTime(run.CreateTime),
			}
		}
		result = append(result, entry)
	}
	return &vo.PageResult[ScheduleListItemVO]{List: result, Total: total}, nil
}

// GetDetail 定时任务详情。
func (s *ScheduleService) GetDetail(ctx context.Context, userID, scheduleID int64) (*ScheduleDetailVO, error) {
	task, err := s.owned(ctx, userID, scheduleID)
	if err != nil {
		return nil, err
	}
	return toScheduleVO(task), nil
}

// Update 更新定时任务（变更后重算下次触发时间）。
func (s *ScheduleService) Update(ctx context.Context, userID, scheduleID int64, form *ScheduleUpdateForm) (*ScheduleDetailVO, error) {
	task, err := s.owned(ctx, userID, scheduleID)
	if err != nil {
		return nil, err
	}
	if form.Input != nil {
		if err := validateTypedConfig(form.Input, []string{"fixed", "dynamic"}, "input"); err != nil {
			return nil, err
		}
	}
	if form.Output != nil {
		if err := validateTypedConfig(form.Output, []string{"message", "callback"}, "output"); err != nil {
			return nil, err
		}
	}

	fields := map[string]any{}
	recompute := false
	if form.Name != nil {
		fields["name"] = *form.Name
		task.Name = *form.Name
	}
	if form.Cron != nil {
		cron, err := normalizeCron(*form.Cron)
		if err != nil {
			return nil, err
		}
		if _, err := parseCron(cron); err != nil {
			return nil, err
		}
		fields["cron"] = cron
		task.Cron = cron
		recompute = true
	}
	if form.Timezone != nil {
		fields["timezone"] = *form.Timezone
		task.Timezone = *form.Timezone
		recompute = true
	}
	if form.Input != nil {
		fields["input"] = marshalJSON(form.Input)
		task.Input = marshalJSON(form.Input)
	}
	if form.Output != nil {
		fields["output"] = marshalJSON(form.Output)
		task.Output = marshalJSON(form.Output)
	}
	if form.Enabled != nil {
		fields["enabled"] = *form.Enabled
		task.Enabled = *form.Enabled
		if *form.Enabled == 1 {
			recompute = true
		}
	}
	if recompute {
		nextTrigger, err := computeNextTrigger(task.Cron, task.Timezone)
		if err != nil {
			return nil, err
		}
		fields["next_trigger_time"] = nextTrigger
		task.NextTriggerTime = &nextTrigger
	}
	if len(fields) > 0 {
		fields["update_by"] = userID
		if err := s.schedules.UpdateFields(ctx, scheduleID, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新定时任务失败", err)
		}
	}
	return toScheduleVO(task), nil
}

// SetEnabled 启停任务（熔断停用后重新启用会重置熔断计数并重算下次触发时间）。
func (s *ScheduleService) SetEnabled(ctx context.Context, userID, scheduleID int64, enabled int) error {
	if _, err := s.owned(ctx, userID, scheduleID); err != nil {
		return err
	}
	if err := s.schedules.SetEnabled(ctx, scheduleID, enabled, userID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "启停定时任务失败", err)
	}
	return nil
}

// Delete 删除定时任务（软删，不可恢复）。
func (s *ScheduleService) Delete(ctx context.Context, userID, scheduleID int64) error {
	if _, err := s.owned(ctx, userID, scheduleID); err != nil {
		return err
	}
	if err := s.schedules.SoftDelete(ctx, scheduleID, userID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除定时任务失败", err)
	}
	return nil
}

// PreviewNextTimes Cron 解释与接下来 N 次触发时间预览。
func (s *ScheduleService) PreviewNextTimes(ctx context.Context, cron string, count int) (*NextTimesPreviewVO, error) {
	normalized, err := normalizeCron(cron)
	if err != nil {
		return nil, err
	}
	schedule, err := parseCron(normalized)
	if err != nil {
		return nil, err
	}
	location, err := resolveLocation(defaultScheduleTimezone)
	if err != nil {
		return nil, err
	}
	times := schedule.nextTimes(time.Now().In(location), count)
	formatted := make([]string, 0, len(times))
	for _, item := range times {
		formatted = append(formatted, item.Format(time.RFC3339))
	}
	return &NextTimesPreviewVO{Description: describeCron(normalized), NextTimes: formatted}, nil
}

// ListHistory 执行历史分页。
func (s *ScheduleService) ListHistory(ctx context.Context, userID, scheduleID int64, page, size int) (*vo.PageResult[RunHistoryVO], error) {
	if _, err := s.owned(ctx, userID, scheduleID); err != nil {
		return nil, err
	}
	items, total, err := s.schedules.PaginateHistory(ctx, scheduleID, page, size)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询执行历史失败", err)
	}
	result := make([]RunHistoryVO, 0, len(items))
	for _, item := range items {
		result = append(result, RunHistoryVO{
			ID:             item.ID,
			ScheduleID:     item.ScheduleID,
			Status:         item.Status,
			SkipReason:     derefString(item.SkipReason),
			Credits:        item.Credits,
			DurationMs:     item.DurationMs,
			ErrorMsg:       derefString(item.ErrorMsg),
			ConversationID: item.ConversationID,
			RequestID:      derefString(item.RequestID),
			WindowStart:    formatTime(item.WindowStart),
			CreateTime:     formatTime(item.CreateTime),
		})
	}
	return &vo.PageResult[RunHistoryVO]{List: result, Total: total}, nil
}

// ensureVip2 校验用户为 VIP2 及以上（无会员记录视为 level_0）。
func (s *ScheduleService) ensureVip2(ctx context.Context, userID int64) error {
	levelCode, err := s.schedules.GetMemberLevelCode(ctx, userID)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会员信息失败", err)
	}
	if memberLevelValues[levelCode] < 2 {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "定时调度功能需 VIP2 及以上会员，请升级会员后使用")
	}
	return nil
}

// owned 取归属当前用户且未删除的任务，越权或不存在抛 404。
func (s *ScheduleService) owned(ctx context.Context, userID, scheduleID int64) (*model.SysAiSchedule, error) {
	task, err := s.schedules.GetByID(ctx, scheduleID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询定时任务失败", err)
	}
	if task == nil || task.UserID != userID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "定时任务不存在")
	}
	return task, nil
}

// validateTypedConfig 校验 JSON 配置的 type 枚举合法性（不深度校验内部结构）。
func validateTypedConfig(config map[string]any, allowed []string, field string) error {
	if config == nil {
		return nil
	}
	configType, _ := config["type"].(string)
	for _, item := range allowed {
		if configType == item {
			return nil
		}
	}
	return common.NewBizError(common.PARAM_ERROR, field+".type 取值非法，应为 "+strings.Join(allowed, "/"))
}

func toScheduleVO(task *model.SysAiSchedule) *ScheduleDetailVO {
	return &ScheduleDetailVO{
		ID:              task.ID,
		UserID:          task.UserID,
		Name:            task.Name,
		Cron:            task.Cron,
		Timezone:        task.Timezone,
		Input:           rawJSON(task.Input),
		Output:          rawJSON(task.Output),
		Enabled:         task.Enabled,
		Status:          task.Status,
		CircuitStreak:   task.CircuitStreak,
		NextTriggerTime: formatTimePtr(task.NextTriggerTime),
		CreateTime:      formatTime(task.CreateTime),
	}
}
