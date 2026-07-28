package pkgsale

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IPackageService interface {
	ListOnSale(ctx context.Context) ([]vo.PackageDetailVO, error)
	GetDetail(ctx context.Context, id int64) (*vo.PackageDetailVO, error)
	CalculatePrice(ctx context.Context, userID, packageID int64, userCouponID *int64) (*vo.PriceResult, error)
	GetPage(ctx context.Context, q *query.PackagePageQuery) (*vo.PageResult[vo.PackagePageVO], error)
	GetForm(ctx context.Context, id int64) (*bo.PackageForm, error)
	Create(ctx context.Context, form *bo.PackageForm) error
	Update(ctx context.Context, id int64, form *bo.PackageForm) error
	UpdateStatus(ctx context.Context, id int64, status int) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	GetSalesStats(ctx context.Context) (*vo.SalesStatsVO, error)
}

type ICouponService interface {
	ListMy(ctx context.Context, userID int64, status *int) ([]vo.UserCouponVO, error)
	Receive(ctx context.Context, userID, couponID int64) (*vo.CouponReceiveResult, error)
	GetPage(ctx context.Context, q *query.CouponPageQuery) (*vo.PageResult[vo.CouponVO], error)
	Create(ctx context.Context, form *bo.CouponForm) (*vo.CouponCreateResult, error)
	Update(ctx context.Context, id int64, form *bo.CouponForm) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	BatchDistribute(ctx context.Context, form *bo.CouponBatchDistributeForm) (*vo.CouponBatchDistributeResult, error)
}
