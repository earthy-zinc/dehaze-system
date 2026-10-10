package feedback

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type RatingWithModel struct {
	model.SysRating
	ModelName string `gorm:"column:model_name" json:"modelName"`
}

type RatingWithUserAndModel struct {
	model.SysRating
	Username   string `gorm:"column:username" json:"username"`
	UserAvatar string `gorm:"column:user_avatar" json:"userAvatar"`
	ModelName  string `gorm:"column:model_name" json:"modelName"`
}

type FeedbackWithUser struct {
	model.SysFeedback
	Username     string `gorm:"column:username" json:"username"`
	AssigneeName string `gorm:"column:assignee_name" json:"assigneeName"`
}

type ModuleCountRow struct {
	Module string `gorm:"column:module" json:"module"`
	Count  int64  `gorm:"column:count" json:"count"`
}

type TagCountRow struct {
	Tag   string `gorm:"column:tag" json:"tag"`
	Count int64  `gorm:"column:count" json:"count"`
}

// ModelStatRow 模型维度评价聚合行（avg_rating 为原始均值，精度处理在 service 层）
type ModelStatRow struct {
	ModelID   string  `gorm:"column:model_id" json:"modelId"`
	ModelName string  `gorm:"column:model_name" json:"modelName"`
	AvgRating float64 `gorm:"column:avg_rating" json:"avgRating"`
	Total     int64   `gorm:"column:total" json:"total"`
	LowCount  int64   `gorm:"column:low_count" json:"lowCount"`
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
	FindPageMy(ctx context.Context, userID int64, pageNum, pageSize int) ([]RatingWithModel, int64, error)
	FindPage(ctx context.Context, q *query.RatingPageQuery) ([]RatingWithUserAndModel, int64, error)
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	GetStats(ctx context.Context, startTime, endTime string) (total int64, avgRating float64, distribution map[int]int64, err error)
	GetTagRanking(ctx context.Context, startTime, endTime string) ([]TagCountRow, error)
	GetModelStats(ctx context.Context, startTime, endTime string) ([]ModelStatRow, error)
	GetRatingCountByValue(ctx context.Context, rating int, startTime, endTime string) (int64, error)
	CountLowRatingsByModelSince(ctx context.Context, modelID string, since time.Time) (int64, error)
	GetTodayLowRatingCounts(ctx context.Context) (lowCount int64, totalCount int64, err error)
	GetStatsByModelID(ctx context.Context, modelID string) (totalCount int64, avgRating float64, distribution map[int8]int64, err error)
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
