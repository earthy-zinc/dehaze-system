package pkgsale

import (
	"context"
	"database/sql"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type PromotionWithPackage struct {
	PromotionID   int64
	Name          string
	Type          string
	Description   string
	DiscountType  string
	DiscountValue int64
	Status        int8
	StartTime     time.Time
	EndTime       time.Time
	ActivityRules sql.NullString
	NewUserOnly   int8
}

type PackageOrderStatRow struct {
	PackageID   int64
	PackageName string
	Count       int64
	Revenue     int64
}

type LevelOrderStatRow struct {
	PackageLevel string
	Count        int64
	Revenue      int64
}

type PeriodOrderStatRow struct {
	Period  string
	Count   int64
	Revenue int64
}

type IPackageRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysPackage, error)
	FindByIDs(ctx context.Context, ids []int64) ([]model.SysPackage, error)
	ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error)
	FindAllOnSale(ctx context.Context) ([]model.SysPackage, error)
	FindPage(ctx context.Context, q *query.PackagePageQuery) ([]model.SysPackage, int64, error)
	Create(ctx context.Context, p *model.SysPackage) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	UpdateStatus(ctx context.Context, id int64, status int8) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	IncrementSalesCount(ctx context.Context, id int64, delta int64) error
	CountOrders(ctx context.Context, packageID int64) (int64, error)
	CountPaidOrdersByUser(ctx context.Context, userID int64) (int64, error)
	FindActiveVipByLevelCode(ctx context.Context, levelCode string) (*model.SysPackage, error)
	FindActivePromotionsByPackageID(ctx context.Context, packageID int64) ([]PromotionWithPackage, error)
	SumPaidAmountByStatus(ctx context.Context, statuses []int8) (int64, error)
	CountOrdersByStatus(ctx context.Context, statuses []int8) (int64, error)
	GetPackageOrderStats(ctx context.Context, statuses []int8) ([]PackageOrderStatRow, error)
	GetLevelOrderStats(ctx context.Context, statuses []int8) ([]LevelOrderStatRow, error)
	GetPeriodOrderStats(ctx context.Context, statuses []int8) ([]PeriodOrderStatRow, error)
}

type IPromotionRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysPromotion, error)
	FindPage(ctx context.Context, q *query.PromotionPageQuery) ([]model.SysPromotion, int64, error)
	Create(ctx context.Context, p *model.SysPromotion) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	UpdateStatus(ctx context.Context, id int64, status int8) error
	DeleteByID(ctx context.Context, id int64) error
	ListPackageIDs(ctx context.Context, promotionID int64) ([]int64, error)
	RebindPackages(ctx context.Context, promotionID int64, discountType string, discountValue int64, packageIDs []int64) error
}

type ICouponRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysCoupon, error)
	FindByIDsIncludeDeleted(ctx context.Context, ids []int64) ([]model.SysCoupon, error)
	FindPage(ctx context.Context, q *query.CouponPageQuery) ([]model.SysCoupon, int64, error)
	Create(ctx context.Context, c *model.SysCoupon) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	DeleteByIDs(ctx context.Context, ids []int64) error
	IncrementIssuedQtyWithLimit(ctx context.Context, id int64, n int64) (bool, error)
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
	DeleteByCouponIDs(ctx context.Context, couponIDs []int64) error
	CountUsedByCouponIDs(ctx context.Context, couponIDs []int64) (int64, error)
	FindActiveTrialCouponExpireTime(ctx context.Context, userID int64) (*time.Time, error)
}
