package feedback

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IRatingService interface {
	CreateRating(ctx context.Context, userID int64, form *bo.RatingCreateForm) (int64, error)
	UpdateRating(ctx context.Context, userID, ratingID int64, form *bo.RatingCreateForm) error
	ListMyRatings(ctx context.Context, userID int64, pageNum, pageSize int) (*vo.PageResult[vo.MyRatingVO], error)
	GetRatingByPrediction(ctx context.Context, predLogID int64) (*vo.RatingDetailVO, error)
	ListPagedRatings(ctx context.Context, q *query.RatingPageQuery) (*vo.PageResult[vo.RatingPageVO], error)
	HideRating(ctx context.Context, id int64) error
	ReplyRating(ctx context.Context, id int64, content string) error
	GetRatingStats(ctx context.Context, startTime, endTime string) (*vo.RatingStatsVO, error)
}

type IFeedbackService interface {
	CreateFeedback(ctx context.Context, userID int64, form *bo.FeedbackCreateForm) (int64, error)
	ListMyFeedback(ctx context.Context, userID int64, pageNum, pageSize int) (*vo.PageResult[vo.FeedbackPageVO], error)
	GetFeedbackDetail(ctx context.Context, id int64) (*vo.FeedbackDetailVO, error)
	SupplementFeedback(ctx context.Context, userID, feedbackID int64, form *bo.FeedbackSupplementForm) error
	ListPagedFeedback(ctx context.Context, q *query.FeedbackPageQuery) (*vo.PageResult[vo.FeedbackPageVO], error)
	AssignFeedback(ctx context.Context, id, assigneeID int64) error
	ReplyFeedback(ctx context.Context, adminID, feedbackID int64, form *bo.FeedbackReplyForm) error
	CloseFeedback(ctx context.Context, id int64, reason string) error
	UpdateFeedbackTags(ctx context.Context, id int64, tags []string) error
	GetFeedbackStats(ctx context.Context, startTime, endTime string) (*vo.FeedbackStatsVO, error)
}

type ILowRatingAlertService interface {
	PublishRatingEvent(ctx context.Context, rating *model.SysRating) error
	HandleMessage(ctx context.Context, body []byte) error
	CheckAndAlert(ctx context.Context, ratingID int64) error
}
