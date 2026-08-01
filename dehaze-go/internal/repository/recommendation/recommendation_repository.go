package recommendation

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type recommendationRepository struct {
	db *gorm.DB
}

func NewRecommendationRepository(db *gorm.DB) RecommendationRepository {
	return &recommendationRepository{db: db}
}

func (r *recommendationRepository) Create(ctx context.Context, rec *model.SysRecommendation) error {
	return r.db.WithContext(ctx).Create(rec).Error
}

func (r *recommendationRepository) FindByID(ctx context.Context, id int64) (*model.SysRecommendation, error) {
	var rec model.SysRecommendation
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&rec).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rec, err
}

func (r *recommendationRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysRecommendation{}).
		Where("id = ?", id).
		Updates(updates).Error
}

func (r *recommendationRepository) CountTotal(ctx context.Context, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysRecommendation{})
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var count int64
	err := db.Count(&count).Error
	return count, err
}

func (r *recommendationRepository) CountUseful(ctx context.Context, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysRecommendation{}).Where("feedback = 1")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var count int64
	err := db.Count(&count).Error
	return count, err
}

func (r *recommendationRepository) CountFeedbackTotal(ctx context.Context, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysRecommendation{}).Where("feedback IN (1, 2)")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var count int64
	err := db.Count(&count).Error
	return count, err
}

func (r *recommendationRepository) CountAdoptedAlgorithmDistinct(ctx context.Context, startTime, endTime string) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysRecommendation{}).Where("adopted_algorithm_id IS NOT NULL")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var count int64
	err := db.Distinct("adopted_algorithm_id").Count(&count).Error
	return count, err
}

func (r *recommendationRepository) FindDailyAdoptionRate(ctx context.Context, startTime, endTime string) ([]DailyAdoptionRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_recommendation").
		Select("DATE(create_time) AS date, IF(COUNT(*) > 0, SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) / COUNT(*), 0) AS adoption_rate").
		Where("feedback IN (1, 2)")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	var rows []DailyAdoptionRow
	err := db.Group("DATE(create_time)").Order("date").Scan(&rows).Error
	return rows, err
}
