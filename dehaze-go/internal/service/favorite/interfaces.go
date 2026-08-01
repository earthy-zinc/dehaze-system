package favorite

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IFavoriteService interface {
	Add(ctx context.Context, userID int64, form *bo.FavoriteForm) (int64, error)
	DeleteByIDs(ctx context.Context, userID int64, ids []int64) error
	GetPage(ctx context.Context, userID int64, q *query.FavoritePageQuery) (*vo.PageResult[vo.FavoriteVO], error)
	GetStatus(ctx context.Context, userID int64, targetType string, targetID int64) (*vo.FavoriteStatusVO, error)
	GetCount(ctx context.Context, userID int64, targetType string) ([]vo.FavoriteCountVO, error)
	MarkInvalid(ctx context.Context, targetType string, targetIDs []int64) error
}
