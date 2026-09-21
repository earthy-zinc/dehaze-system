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
	// CountRecommended 统计带推荐来源（sys_pred_log.recommended_by 非空）的预测记录数
	CountRecommended(ctx context.Context, startTime, endTime string) (int64, error)
	// FindDailyTotal 按日统计推荐总数
	FindDailyTotal(ctx context.Context, startTime, endTime string) ([]DailyCountRow, error)
	// FindDailyRecommended 按日统计带推荐来源的预测记录数
	FindDailyRecommended(ctx context.Context, startTime, endTime string) ([]DailyCountRow, error)
}

type DailyCountRow struct {
	Date  string `gorm:"column:date" json:"date"`
	Count int64  `gorm:"column:cnt" json:"cnt"`
}

type RuleRepository interface {
	FindAll(ctx context.Context) ([]model.SysRecommendationRule, error)
	FindEnabled(ctx context.Context) ([]model.SysRecommendationRule, error)
	FindByID(ctx context.Context, id int64) (*model.SysRecommendationRule, error)
	Create(ctx context.Context, r *model.SysRecommendationRule) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
}
