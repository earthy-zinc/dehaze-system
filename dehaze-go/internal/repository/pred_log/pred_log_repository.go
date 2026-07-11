package pred_log

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IPredLogRepository 预测日志仓储接口
type IPredLogRepository interface {
	Create(ctx context.Context, log *model.SysPredLog) error
	FindByID(ctx context.Context, id int64) (*model.SysPredLog, error)
	FindByAlgorithmAndMD5(ctx context.Context, algorithmID int64, originMD5 string) (*model.SysPredLog, error)
	FindPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) ([]model.SysPredLog, int64, error)
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

func (r *predLogRepository) FindByAlgorithmAndMD5(ctx context.Context, algorithmID int64, originMD5 string) (*model.SysPredLog, error) {
	var log model.SysPredLog
	err := r.db.WithContext(ctx).
		Where("algorithm_id = ? AND origin_md5 = ?", algorithmID, originMD5).
		Order("create_time DESC").
		First(&log).Error
	if err != nil {
		return nil, err
	}
	return &log, nil
}

func (r *predLogRepository) FindPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) ([]model.SysPredLog, int64, error) {
	var list []model.SysPredLog
	var total int64
	query := r.db.WithContext(ctx).Model(&model.SysPredLog{})
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
