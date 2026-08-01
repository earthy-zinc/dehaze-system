package favorite

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

// FavoriteWithAlgorithm 收藏记录+算法名联合结果
type FavoriteWithAlgorithm struct {
	model.SysFavorite
	AlgorithmName string `gorm:"column:algorithm_name" json:"algorithmName"`
}

// IFavoriteRepository 收藏数据访问接口
type IFavoriteRepository interface {
	Create(ctx context.Context, f *model.SysFavorite) error
	FindByUserAndTarget(ctx context.Context, userID int64, targetType string, targetID int64) (*model.SysFavorite, error)
	Upsert(ctx context.Context, f *model.SysFavorite) error
	FindPage(ctx context.Context, userID int64, q *query.FavoritePageQuery) ([]FavoriteWithAlgorithm, int64, error)
	CountByUserID(ctx context.Context, userID int64) (int64, error)
	CountByUserAndType(ctx context.Context, userID int64, targetType string) (int64, error)
	CountGroupByType(ctx context.Context, userID int64, targetType string) ([]CountByTypeRow, error)
	DeleteByIDs(ctx context.Context, userID int64, ids []int64) error
	UpdateByID(ctx context.Context, id int64, updates map[string]any) error
	MarkInvalid(ctx context.Context, targetType string, targetIDs []int64) error
}

// CountByTypeRow 按类型计数行
type CountByTypeRow struct {
	TargetType string `gorm:"column:target_type" json:"targetType"`
	Count      int64  `gorm:"column:count" json:"count"`
}
