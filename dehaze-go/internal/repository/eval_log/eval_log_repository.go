package eval_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IEvalLogRepository 评估日志仓储接口
type IEvalLogRepository interface {
	Create(ctx context.Context, log *model.SysEvalLog) error
	FindByID(ctx context.Context, id int64) (*model.SysEvalLog, error)

	FindPageByUser(ctx context.Context, userID int64, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error)
	// FindLogPageByUser 评估日志列表分页（**当前用户全部状态**的评估日志，python `list_logs` 口径）
	FindLogPageByUser(ctx context.Context, userID int64, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error)
	UpdateResult(ctx context.Context, id int64, status model.LogStatus, result string, time int) error
	UpdateStatus(ctx context.Context, id int64, status model.LogStatus, errorMessage string, time int) error
	MarkStuckAsFailed(ctx context.Context, threshold time.Time) (int, error)
}

type evalLogRepository struct {
	db *gorm.DB
}

func NewEvalLogRepository(db *gorm.DB) IEvalLogRepository {
	return &evalLogRepository{db: db}
}

func (r *evalLogRepository) Create(ctx context.Context, log *model.SysEvalLog) error {
	return r.db.WithContext(ctx).Create(log).Error
}

func (r *evalLogRepository) FindByID(ctx context.Context, id int64) (*model.SysEvalLog, error) {
	var log model.SysEvalLog
	err := r.db.WithContext(ctx).First(&log, id).Error
	if err != nil {
		return nil, err
	}
	return &log, nil
}

func (r *evalLogRepository) UpdateResult(ctx context.Context, id int64, status model.LogStatus, result string, time int) error {
	return r.db.WithContext(ctx).Model(&model.SysEvalLog{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"status": status,
			"result": result,
			"time":   time,
		}).Error
}

func (r *evalLogRepository) UpdateStatus(ctx context.Context, id int64, status model.LogStatus, errorMessage string, time int) error {
	updates := map[string]any{
		"status": status,
		"time":   time,
	}
	if errorMessage != "" {
		updates["error_message"] = errorMessage
	}
	return r.db.WithContext(ctx).Model(&model.SysEvalLog{}).
		Where("id = ?", id).
		Updates(updates).Error
}

func (r *evalLogRepository) FindPageByUser(ctx context.Context, userID int64, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error) {
	var list []model.SysEvalLog
	var total int64
	// 指标历史仅含效果评估任务，排除同表存储的对比报告行
	query := r.db.WithContext(ctx).Model(&model.SysEvalLog{}).
		Where("create_by = ? AND status = ? AND task_type = ?", userID, model.LogStatusCompleted, "evaluation")
	if algorithmID > 0 {
		query = query.Where("algorithm_id = ?", algorithmID)
	}
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	offset := (pageNum - 1) * pageSize
	if err := query.Order("id DESC").Offset(offset).Limit(pageSize).Find(&list).Error; err != nil {
		return nil, 0, err
	}
	return list, total, nil
}

// FindLogPageByUser 评估日志列表（`GET /evaluation/logs`）分页。
// 对齐 python `eval_log_repository.get_paginated`：**仅按 create_by 隔离**（不限状态、不限 task_type）、按 id 倒序；
// 而"指标历史"（`GET /evaluation/metrics`）另有 completed + task_type='evaluation' 的口径，见 FindPageByUser。
func (r *evalLogRepository) FindLogPageByUser(ctx context.Context, userID int64, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error) {
	var list []model.SysEvalLog
	var total int64
	query := r.db.WithContext(ctx).Model(&model.SysEvalLog{}).Where("create_by = ?", userID)
	if algorithmID > 0 {
		query = query.Where("algorithm_id = ?", algorithmID)
	}
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	offset := (pageNum - 1) * pageSize
	if err := query.Order("id DESC").Offset(offset).Limit(pageSize).Find(&list).Error; err != nil {
		return nil, 0, err
	}
	return list, total, nil
}

func (r *evalLogRepository) MarkStuckAsFailed(ctx context.Context, threshold time.Time) (int, error) {
	result := r.db.WithContext(ctx).Model(&model.SysEvalLog{}).
		Where("status = ? AND update_time < ?", model.LogStatusProcessing, threshold).
		Updates(map[string]any{
			"status":        model.LogStatusFailed,
			"error_message": "任务执行超时，服务可能已重启",
		})
	return int(result.RowsAffected), result.Error
}
