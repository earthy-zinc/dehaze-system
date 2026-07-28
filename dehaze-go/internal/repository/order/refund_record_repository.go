package order

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type RefundRecordRepository struct {
	db *gorm.DB
}

func NewRefundRecordRepository(db *gorm.DB) *RefundRecordRepository {
	return &RefundRecordRepository{db: db}
}

func (r *RefundRecordRepository) FindByID(ctx context.Context, id int64) (*model.SysRefundRecord, error) {
	var rr model.SysRefundRecord
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&rr).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rr, err
}

func (r *RefundRecordRepository) FindByOrderID(ctx context.Context, orderID int64) (*model.SysRefundRecord, error) {
	var rr model.SysRefundRecord
	err := r.db.WithContext(ctx).
		Where("order_id = ? AND deleted = 0", orderID).
		Order("id DESC").
		First(&rr).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &rr, err
}

func (r *RefundRecordRepository) Create(ctx context.Context, rr *model.SysRefundRecord) error {
	return r.db.WithContext(ctx).Create(rr).Error
}

func (r *RefundRecordRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysRefundRecord{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *RefundRecordRepository) FindPage(ctx context.Context, q *query.RefundPageQuery) ([]RefundWithOrder, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_refund_record rr").
		Select("rr.*, o.order_no, u.username").
		Joins("LEFT JOIN sys_order o ON rr.order_id = o.id").
		Joins("LEFT JOIN sys_user u ON rr.user_id = u.id").
		Where("rr.deleted = 0")

	if q.OrderNo != "" {
		db = db.Where("o.order_no = ?", q.OrderNo)
	}
	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("o.order_no LIKE ? OR u.username LIKE ? OR rr.refund_no LIKE ?", kw, kw, kw)
	}
	if q.Status != "" {
		db = db.Where("rr.status = ?", refundStatusToInt(q.Status))
	}
	if q.ApplyTimeStart != "" {
		db = db.Where("rr.apply_time >= ?", q.ApplyTimeStart)
	}
	if q.ApplyTimeEnd != "" {
		db = db.Where("rr.apply_time <= ?", q.ApplyTimeEnd)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []RefundWithOrder
	err := db.Order("rr.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

var _ IRefundRecordRepository = (*RefundRecordRepository)(nil)
