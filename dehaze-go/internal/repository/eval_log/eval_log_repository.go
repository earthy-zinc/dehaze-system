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
	FindPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error)
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

func (r *evalLogRepository) FindPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) ([]model.SysEvalLog, int64, error) {
	var list []model.SysEvalLog
	var total int64
	query := r.db.WithContext(ctx).Model(&model.SysEvalLog{})
	if algorithmID > 0 {
		query = query.Where("algorithm_id = ?", algorithmID)
	}
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	offset := (pageNum - 1) * pageSize
	if err := query.Order("create_time DESC").Offset(offset).Limit(pageSize).Find(&list).Error; err != nil {
		return nil, 0, err
	}
	return list, total, nil
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

func (r *evalLogRepository) MarkStuckAsFailed(ctx context.Context, threshold time.Time) (int, error) {
	result := r.db.WithContext(ctx).Model(&model.SysEvalLog{}).
		Where("status = ? AND update_time < ?", model.LogStatusProcessing, threshold).
		Updates(map[string]any{
			"status":        model.LogStatusFailed,
			"error_message": "任务执行超时，服务可能已重启",
		})
	return int(result.RowsAffected), result.Error
}
