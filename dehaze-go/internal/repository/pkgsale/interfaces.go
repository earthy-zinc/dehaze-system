package pkgsale

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type IPackageRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysPackage, error)
	FindByIDs(ctx context.Context, ids []int64) ([]model.SysPackage, error)
	FindAllOnSale(ctx context.Context) ([]model.SysPackage, error)
	FindPage(ctx context.Context, q *query.PackagePageQuery) ([]model.SysPackage, int64, error)
	Create(ctx context.Context, p *model.SysPackage) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	UpdateStatus(ctx context.Context, id int64, status int8) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	IncrementSalesCount(ctx context.Context, id int64, delta int64) error
	CountOrders(ctx context.Context, packageID int64) (int64, error)
}

type ICouponRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysCoupon, error)
	FindPage(ctx context.Context, q *query.CouponPageQuery) ([]model.SysCoupon, int64, error)
	Create(ctx context.Context, c *model.SysCoupon) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	IncrementIssuedQty(ctx context.Context, id int64) error
	IncrementUsedQty(ctx context.Context, id int64) error
	CountIssued(ctx context.Context) (int64, error)
	CountUsed(ctx context.Context) (int64, error)
}

type IUserCouponRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysUserCoupon, error)
	FindByUserIDAndCouponID(ctx context.Context, userID, couponID int64) (*model.SysUserCoupon, error)
	FindByUserID(ctx context.Context, userID int64, status *int) ([]model.SysUserCoupon, error)
	FindByUserIDAndStatusForUpdate(ctx context.Context, userID, userCouponID int64) (*model.SysUserCoupon, error)
	Create(ctx context.Context, uc *model.SysUserCoupon) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	CountByUserIDAndCouponID(ctx context.Context, userID, couponID int64) (int64, error)
	FindExpired(ctx context.Context, before time.Time) ([]model.SysUserCoupon, error)
	BatchMarkExpired(ctx context.Context, ids []int64) error
}
