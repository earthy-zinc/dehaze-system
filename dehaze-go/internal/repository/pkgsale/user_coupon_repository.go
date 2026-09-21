package pkgsale

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type UserCouponRepository struct {
	db *gorm.DB
}

func NewUserCouponRepository(db *gorm.DB) *UserCouponRepository {
	return &UserCouponRepository{db: db}
}

func (r *UserCouponRepository) FindByID(ctx context.Context, id int64) (*model.SysUserCoupon, error) {
	var uc model.SysUserCoupon
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&uc).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &uc, err
}

func (r *UserCouponRepository) FindByUserIDAndCouponID(ctx context.Context, userID, couponID int64) (*model.SysUserCoupon, error) {
	var uc model.SysUserCoupon
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND coupon_id = ? AND deleted = 0", userID, couponID).
		Order("id DESC").
		First(&uc).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &uc, err
}

func (r *UserCouponRepository) FindByUserID(ctx context.Context, userID int64, status *int) ([]model.SysUserCoupon, error) {
	db := r.db.WithContext(ctx).
		Where("user_id = ? AND deleted = 0", userID)
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	var list []model.SysUserCoupon
	err := db.Order("id DESC").Find(&list).Error
	return list, err
}

func (r *UserCouponRepository) FindByUserIDAndStatusForUpdate(ctx context.Context, userID, userCouponID int64) (*model.SysUserCoupon, error) {
	var uc model.SysUserCoupon
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ? AND deleted = 0", userCouponID, userID).
		First(&uc).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &uc, err
}

func (r *UserCouponRepository) Create(ctx context.Context, uc *model.SysUserCoupon) error {
	return r.db.WithContext(ctx).Create(uc).Error
}

func (r *UserCouponRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *UserCouponRepository) CountByUserIDAndCouponID(ctx context.Context, userID, couponID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("user_id = ? AND coupon_id = ? AND deleted = 0", userID, couponID).
		Count(&count).Error
	return count, err
}

func (r *UserCouponRepository) FindExpired(ctx context.Context, before time.Time) ([]model.SysUserCoupon, error) {
	var list []model.SysUserCoupon
	err := r.db.WithContext(ctx).
		Where("status = 1 AND expire_time IS NOT NULL AND expire_time < ? AND deleted = 0", before).
		Find(&list).Error
	return list, err
}

func (r *UserCouponRepository) BatchMarkExpired(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("id IN ? AND deleted = 0", ids).
		Update("status", 3).Error
}

func (r *UserCouponRepository) DeleteByCouponIDs(ctx context.Context, couponIDs []int64) error {
	if len(couponIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("coupon_id IN ? AND status = 1 AND deleted = 0", couponIDs).
		Update("deleted", gorm.Expr("id")).Error
}

func (r *UserCouponRepository) CountUsedByCouponIDs(ctx context.Context, couponIDs []int64) (int64, error) {
	if len(couponIDs) == 0 {
		return 0, nil
	}
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("coupon_id IN ? AND status = 2 AND deleted = 0", couponIDs).
		Count(&count).Error
	return count, err
}

// FindActiveTrialCouponExpireTime 有效体验券（未使用且未过期）的最晚到期时间；无券返回 nil。
// 对齐 python `user_coupon_repository.get_active_trial_coupon`：join sys_coupon 限定 type='trial'，
// user_coupon.status=1（未使用）、未删除，到期时间为空或晚于当前，按到期时间倒序取一张。
func (r *UserCouponRepository) FindActiveTrialCouponExpireTime(ctx context.Context, userID int64) (*time.Time, error) {
	var row struct {
		ExpireTime *time.Time `gorm:"column:expire_time"`
	}
	err := r.db.WithContext(ctx).
		Table("sys_user_coupon").
		Select("sys_user_coupon.expire_time").
		Joins("JOIN sys_coupon ON sys_coupon.id = sys_user_coupon.coupon_id AND sys_coupon.deleted = 0").
		Where("sys_user_coupon.user_id = ? AND sys_user_coupon.deleted = 0 AND sys_user_coupon.status = 1 AND sys_coupon.type = ?",
			userID, "trial").
		Where("sys_user_coupon.expire_time IS NULL OR sys_user_coupon.expire_time > ?", time.Now()).
		Order("sys_user_coupon.expire_time DESC").
		Limit(1).
		Scan(&row).Error
	if err != nil {
		return nil, err
	}
	return row.ExpireTime, nil
}

var _ IUserCouponRepository = (*UserCouponRepository)(nil)
