package pkgsale

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type CouponRepository struct {
	db *gorm.DB
}

func NewCouponRepository(db *gorm.DB) *CouponRepository {
	return &CouponRepository{db: db}
}

func (r *CouponRepository) FindByID(ctx context.Context, id int64) (*model.SysCoupon, error) {
	var c model.SysCoupon
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&c).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &c, err
}

func (r *CouponRepository) FindByIDsIncludeDeleted(ctx context.Context, ids []int64) ([]model.SysCoupon, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var list []model.SysCoupon
	err := r.db.WithContext(ctx).Unscoped().
		Where("id IN ?", ids).
		Find(&list).Error
	return list, err
}

func (r *CouponRepository) FindPage(ctx context.Context, q *query.CouponPageQuery) ([]model.SysCoupon, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysCoupon{}).Where("deleted = 0")
	if q.Name != "" {
		db = db.Where("name LIKE ?", "%"+q.Name+"%")
	}
	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []model.SysCoupon
	err := db.Order("id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&list).Error
	return list, total, err
}

func (r *CouponRepository) Create(ctx context.Context, c *model.SysCoupon) error {
	return r.db.WithContext(ctx).Create(c).Error
}

func (r *CouponRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysCoupon{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *CouponRepository) DeleteByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysCoupon{}).
		Where("id IN ? AND deleted = 0", ids).
		Update("deleted", gorm.Expr("id")).Error
}

// IncrementIssuedQtyWithLimit 条件原子递增发放量：total_qty=-1 不限量，否则 issued_qty+n 不得超过库存。
// 返回 false 表示库存不足或券已删除，调用方据此拒绝领取（防并发超发）。
func (r *CouponRepository) IncrementIssuedQtyWithLimit(ctx context.Context, id int64, n int64) (bool, error) {
	res := r.db.WithContext(ctx).
		Model(&model.SysCoupon{}).
		Where("id = ? AND deleted = 0 AND (total_qty = -1 OR issued_qty + ? <= total_qty)", id, n).
		UpdateColumn("issued_qty", gorm.Expr("issued_qty + ?", n))
	if res.Error != nil {
		return false, res.Error
	}
	return res.RowsAffected > 0, nil
}

func (r *CouponRepository) IncrementUsedQty(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).
		Model(&model.SysCoupon{}).
		Where("id = ? AND deleted = 0", id).
		UpdateColumn("used_qty", gorm.Expr("used_qty + ?", 1)).Error
}

func (r *CouponRepository) CountIssued(ctx context.Context) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("deleted = 0").
		Count(&count).Error
	return count, err
}

func (r *CouponRepository) CountUsed(ctx context.Context) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserCoupon{}).
		Where("status = 2 AND deleted = 0").
		Count(&count).Error
	return count, err
}

var _ ICouponRepository = (*CouponRepository)(nil)
