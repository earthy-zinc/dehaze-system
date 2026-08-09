package recommendation

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
)

type RecommendationRepository interface {
	Create(ctx context.Context, r *model.SysRecommendation) error
	FindByID(ctx context.Context, id int64) (*model.SysRecommendation, error)
	FindLatestByImageMd5(ctx context.Context, imageMd5 string) (*model.SysRecommendation, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	CountTotal(ctx context.Context, startTime, endTime string) (int64, error)
	CountUseful(ctx context.Context, startTime, endTime string) (int64, error)
	CountFeedbackTotal(ctx context.Context, startTime, endTime string) (int64, error)
	CountAdoptedAlgorithmDistinct(ctx context.Context, startTime, endTime string) (int64, error)
	FindDailyAdoptionRate(ctx context.Context, startTime, endTime string) ([]DailyAdoptionRow, error)
}

type DailyAdoptionRow struct {
	Date         string  `gorm:"column:date" json:"date"`
	AdoptionRate float64 `gorm:"column:adoptionRate" json:"adoptionRate"`
}

type RuleRepository interface {
	FindAll(ctx context.Context) ([]model.SysRecommendationRule, error)
	FindEnabled(ctx context.Context) ([]model.SysRecommendationRule, error)
	FindByID(ctx context.Context, id int64) (*model.SysRecommendationRule, error)
	Create(ctx context.Context, r *model.SysRecommendationRule) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
}
