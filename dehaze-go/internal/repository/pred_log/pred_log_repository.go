package pred_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IPredLogRepository 图像任务日志仓储接口
type IPredLogRepository interface {
	Create(ctx context.Context, log *model.SysPredLog) error
	FindByID(ctx context.Context, id int64) (*model.SysPredLog, error)
	FindPage(ctx context.Context, modelID string, userID int64, pageNum, pageSize int) ([]model.SysPredLog, int64, error)
	UpdateResult(ctx context.Context, id int64, status model.LogStatus, predURL, predMD5, originMD5 string, time int) error
	UpdateStatus(ctx context.Context, id int64, status model.LogStatus, errorMessage string, time int) error
	MarkStuckAsFailed(ctx context.Context, threshold time.Time) (int, error)
	// ExistsByID 检查图像任务记录是否存在
	ExistsByID(ctx context.Context, id int64) (bool, error)
	// MarkCancelled 将"处理中"任务置为已取消，返回是否真的流转成功（带 processing 前置，与其它终态写入同口径）
	MarkCancelled(ctx context.Context, id int64) (bool, error)
}

type predLogRepository struct {
	db *gorm.DB
}

func NewPredLogRepository(db *gorm.DB) IPredLogRepository {
	return &predLogRepository{db: db}
}

func (r *predLogRepository) Create(ctx context.Context, log *model.SysPredLog) error {
	return r.db.WithContext(ctx).Create(log).Error
}

func (r *predLogRepository) FindByID(ctx context.Context, id int64) (*model.SysPredLog, error) {
	var log model.SysPredLog
	err := r.db.WithContext(ctx).First(&log, id).Error
	if err != nil {
		return nil, err
	}
	return &log, nil
}

func (r *predLogRepository) FindPage(ctx context.Context, modelID string, userID int64, pageNum, pageSize int) ([]model.SysPredLog, int64, error) {
	var list []model.SysPredLog
	var total int64
	query := r.db.WithContext(ctx).Model(&model.SysPredLog{})
	if modelID != "" {
		query = query.Where("model_id = ?", modelID)
	}
	if userID > 0 {
		// 归属过滤：仅返回该用户的图像任务日志
		query = query.Where("create_by = ?", userID)
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

// UpdateResult 将日志从 processing 流转为 completed 并写入结果。
// 带 processing 前置条件，终态不可被覆盖（防取消/失败后结果复活）。
// originMD5 由算法服务返回（Go 侧只有图片 URL，拿不到原图内容 MD5），为空时不覆盖。
func (r *predLogRepository) UpdateResult(ctx context.Context, id int64, status model.LogStatus, predURL, predMD5, originMD5 string, time int) error {
	updates := map[string]any{
		"status":   status,
		"pred_url": predURL,
		"pred_md5": predMD5,
		"time":     time,
	}
	if originMD5 != "" {
		updates["origin_md5"] = originMD5
	}
	return r.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("id = ? AND status = ?", id, model.LogStatusProcessing).
		Updates(updates).Error
}

// UpdateStatus 将日志从 processing 流转为终态并写入错误信息。
// 带 processing 前置条件，保证终态写入不被并发覆盖。
func (r *predLogRepository) UpdateStatus(ctx context.Context, id int64, status model.LogStatus, errorMessage string, time int) error {
	updates := map[string]any{
		"status": status,
		"time":   time,
	}
	if errorMessage != "" {
		updates["error_message"] = errorMessage
	}
	return r.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("id = ? AND status = ?", id, model.LogStatusProcessing).
		Updates(updates).Error
}

// MarkCancelled 将"处理中"任务置为已取消（python `cancel_task` 的终态写入）。
// 带 processing 前置：并发取消只有一个生效；同时保证已被取消的任务不会被后续异步推理结果覆盖。
func (r *predLogRepository) MarkCancelled(ctx context.Context, id int64) (bool, error) {
	res := r.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("id = ? AND status = ?", id, model.LogStatusProcessing).
		Updates(map[string]any{
			"status":        model.LogStatusCancelled,
			"error_message": "任务已取消",
			"time":          0,
		})
	return res.RowsAffected > 0, res.Error
}

func (r *predLogRepository) MarkStuckAsFailed(ctx context.Context, threshold time.Time) (int, error) {
	result := r.db.WithContext(ctx).Model(&model.SysPredLog{}).
		Where("status = ? AND update_time < ?", model.LogStatusProcessing, threshold).
		Updates(map[string]any{
			"status":        model.LogStatusFailed,
			"error_message": "任务执行超时，服务可能已重启",
		})
	return int(result.RowsAffected), result.Error
}

// ExistsByID 检查图像任务记录是否存在
func (r *predLogRepository) ExistsByID(ctx context.Context, id int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysPredLog{}).
		Where("id = ?", id).
		Count(&count).Error
	return count > 0, err
}
