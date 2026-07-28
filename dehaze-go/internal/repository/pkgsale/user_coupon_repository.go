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

var _ IUserCouponRepository = (*UserCouponRepository)(nil)
