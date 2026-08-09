package feedback

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type RatingWithAlgorithm struct {
	model.SysRating
	AlgorithmName string `gorm:"column:algorithm_name" json:"algorithmName"`
}

type RatingWithUserAndAlgorithm struct {
	model.SysRating
	Username      string `gorm:"column:username" json:"username"`
	UserAvatar    string `gorm:"column:user_avatar" json:"userAvatar"`
	AlgorithmName string `gorm:"column:algorithm_name" json:"algorithmName"`
}

type FeedbackWithUser struct {
	model.SysFeedback
	Username     string `gorm:"column:username" json:"username"`
	AssigneeName string `gorm:"column:assignee_name" json:"assigneeName"`
}

type AlgorithmRatingStatRow struct {
	AlgorithmID   int64   `gorm:"column:algorithm_id" json:"algorithmId"`
	AlgorithmName string  `gorm:"column:algorithm_name" json:"algorithmName"`
	AverageRating float64 `gorm:"column:average_rating" json:"averageRating"`
	TotalRatings  int64   `gorm:"column:total_ratings" json:"totalRatings"`
	LowRatingCount int64  `gorm:"column:low_rating_count" json:"lowRatingCount"`
}

type ModuleCountRow struct {
	Module string `gorm:"column:module" json:"module"`
	Count  int64  `gorm:"column:count" json:"count"`
}

type TagCountRow struct {
	Tag   string `gorm:"column:tag" json:"tag"`
	Count int64  `gorm:"column:count" json:"count"`
}

type TypeCountRow struct {
	Type  string `gorm:"column:type" json:"type"`
	Count int64  `gorm:"column:count" json:"count"`
}

type StatusCountRow struct {
	Status int8  `gorm:"column:status" json:"status"`
	Count  int64 `gorm:"column:count" json:"count"`
}

type ResponseTimeRow struct {
	Hours float64 `gorm:"column:hours" json:"hours"`
}

type IRatingRepository interface {
	Create(ctx context.Context, r *model.SysRating) error
	FindByID(ctx context.Context, id int64) (*model.SysRating, error)
	FindByPredLogID(ctx context.Context, predLogID int64) (*model.SysRating, error)
	FindPageMy(ctx context.Context, userID int64, pageNum, pageSize int) ([]RatingWithAlgorithm, int64, error)
	FindPage(ctx context.Context, q *query.RatingPageQuery) ([]RatingWithUserAndAlgorithm, int64, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	GetStats(ctx context.Context, startTime, endTime string) (total int64, avgRating float64, distribution map[int]int64, err error)
	GetAlgorithmStats(ctx context.Context, startTime, endTime string) ([]AlgorithmRatingStatRow, error)
	GetTagRanking(ctx context.Context, startTime, endTime string) ([]TagCountRow, error)
	GetRatingCountByValue(ctx context.Context, rating int, startTime, endTime string) (int64, error)
	CountLowRatingsByAlgorithmSince(ctx context.Context, algorithmID int64, since time.Time) (int64, error)
	GetTodayLowRatingCounts(ctx context.Context) (lowCount int64, totalCount int64, err error)
	GetStatsByAlgorithmID(ctx context.Context, algorithmID int64) (totalCount int64, avgRating float64, distribution map[int8]int64, err error)
}

type IFeedbackRepository interface {
	Create(ctx context.Context, f *model.SysFeedback) error
	FindByID(ctx context.Context, id int64) (*model.SysFeedback, error)
	FindPageMy(ctx context.Context, userID int64, pageNum, pageSize int) ([]FeedbackWithUser, int64, error)
	FindPage(ctx context.Context, q *query.FeedbackPageQuery) ([]FeedbackWithUser, int64, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	CountTodayByUserID(ctx context.Context, userID int64) (int64, error)
	GetTypeDistribution(ctx context.Context, startTime, endTime string) ([]TypeCountRow, error)
	GetStatusDistribution(ctx context.Context, startTime, endTime string) ([]StatusCountRow, error)
	GetModuleDistribution(ctx context.Context, startTime, endTime string) ([]ModuleCountRow, error)
	GetTotalCount(ctx context.Context, startTime, endTime string) (int64, error)
	GetAvgResponseTime(ctx context.Context, startTime, endTime string) (float64, error)
	GetAvgCloseTime(ctx context.Context, startTime, endTime string) (float64, error)
	GetTopKeywords(ctx context.Context, startTime, endTime string, limit int) ([]KeywordCountRow, error)
}

type KeywordCountRow struct {
	Keyword string `gorm:"column:keyword" json:"keyword"`
	Count   int64  `gorm:"column:count" json:"count"`
}

type IFeedbackReplyRepository interface {
	Create(ctx context.Context, r *model.SysFeedbackReply) error
	FindByFeedbackID(ctx context.Context, feedbackID int64) ([]model.SysFeedbackReply, error)
	FindFirstAdminReply(ctx context.Context, feedbackID int64) (*model.SysFeedbackReply, error)
}
