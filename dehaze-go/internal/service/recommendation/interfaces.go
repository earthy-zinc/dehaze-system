package recommendation

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IRecommendationService interface {
	Analyze(ctx context.Context, form *bo.AnalyzeForm) (*vo.ImageFeatureAnalysisVO, error)
	GetAlgorithmRecommendations(ctx context.Context, userID int64, analysisID *int64, imageMd5 string) ([]vo.RecommendedAlgorithmVO, error)
	SubmitFeedback(ctx context.Context, form *bo.FeedbackForm) (int64, error)
	GetRules(ctx context.Context) ([]vo.RecommendationRuleVO, error)
	UpdateRule(ctx context.Context, id int64, form *bo.RuleForm) (int64, error)
	GetReport(ctx context.Context, startDate, endDate string) (*vo.RecommendationReportVO, error)
}
