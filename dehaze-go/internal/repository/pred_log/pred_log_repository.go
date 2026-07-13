package pred_log

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IPredLogRepository 预测日志仓储接口
type IPredLogRepository interface {
	Create(ctx context.Context, log *model.SysPredLog) error
	FindByID(ctx context.Context, id int64) (*model.SysPredLog, error)
	FindByAlgorithmAndMD5(ctx context.Context, algorithmID int64, originMD5 string) (*model.SysPredLog, error)
	FindPage(ctx context.Context, algorithmID int64, pageNum, pageSize int) ([]model.SysPredLog, int64, error)
	GetMonitorStats(ctx context.Context, algorithmID int64) (*MonitorStats, error)
}

// MonitorStats 算法监控统计原始数据
type MonitorStats struct {
	CallCount      int64
	TodayCallCount int64
	AvgTime        float64
	SuccessCount   int64
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

// GetMonitorStats 获取算法监控统计数据
// 对齐 Java SysAlgorithmServiceImpl#getMonitorData 的统计口径：
//   - callCount: 该算法的总调用次数
//   - todayCallCount: 今日（>= 今日 0 点）的调用次数
//   - avgTime: time IS NOT NULL 记录的平均处理时间
//   - successCount: time IS NOT NULL 且 pred_url 非空 的记录数（用于计算成功率）
func (r *predLogRepository) GetMonitorStats(ctx context.Context, algorithmID int64) (*MonitorStats, error) {
	stats := &MonitorStats{}

	// 总调用次数
	if err := r.db.WithContext(ctx).
		Model(&model.SysPredLog{}).
		Where("algorithm_id = ?", algorithmID).
		Count(&stats.CallCount).Error; err != nil {
		return nil, err
	}

	// 今日调用次数
	now := time.Now()
	todayStart := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	if err := r.db.WithContext(ctx).
		Model(&model.SysPredLog{}).
		Where("algorithm_id = ? AND create_time >= ?", algorithmID, todayStart).
		Count(&stats.TodayCallCount).Error; err != nil {
		return nil, err
	}

	// 平均处理时间（time IS NOT NULL 的记录）
	row := r.db.WithContext(ctx).
		Model(&model.SysPredLog{}).
		Where("algorithm_id = ? AND time IS NOT NULL", algorithmID).
		Select("COALESCE(AVG(time), 0)").
		Row()
	if err := row.Scan(&stats.AvgTime); err != nil {
		return nil, err
	}

	// 成功数（time IS NOT NULL 且 pred_url 非空，对齐 Java 实现）
	if err := r.db.WithContext(ctx).
		Model(&model.SysPredLog{}).
		Where("algorithm_id = ? AND time IS NOT NULL AND pred_url IS NOT NULL AND pred_url <> ''", algorithmID).
		Count(&stats.SuccessCount).Error; err != nil {
		return nil, err
	}

	return stats, nil
}
