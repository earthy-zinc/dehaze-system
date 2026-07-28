package order

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type OrderRepository struct {
	db *gorm.DB
}

func NewOrderRepository(db *gorm.DB) *OrderRepository {
	return &OrderRepository{db: db}
}

func (r *OrderRepository) FindByID(ctx context.Context, id int64) (*model.SysOrder, error) {
	var o model.SysOrder
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&o).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &o, err
}

func (r *OrderRepository) FindByOrderNo(ctx context.Context, orderNo string) (*model.SysOrder, error) {
	var o model.SysOrder
	err := r.db.WithContext(ctx).
		Where("order_no = ? AND deleted = 0", orderNo).
		First(&o).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &o, err
}

func (r *OrderRepository) FindByIDs(ctx context.Context, ids []int64) ([]model.SysOrder, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var list []model.SysOrder
	err := r.db.WithContext(ctx).
		Where("id IN ? AND deleted = 0", ids).
		Find(&list).Error
	return list, err
}

func (r *OrderRepository) Create(ctx context.Context, o *model.SysOrder) error {
	return r.db.WithContext(ctx).Create(o).Error
}

func (r *OrderRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *OrderRepository) UpdateByOrderNo(ctx context.Context, orderNo string, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("order_no = ? AND deleted = 0", orderNo).
		Updates(updates).Error
}

func (r *OrderRepository) FindPageMyOrders(ctx context.Context, userID int64, q *query.MyOrderQuery) ([]OrderWithUser, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_order o").
		Select("o.*, u.username").
		Joins("LEFT JOIN sys_user u ON o.user_id = u.id").
		Where("o.user_id = ? AND o.deleted = 0", userID)

	if q.Status != "" {
		db = db.Where("o.status = ?", orderStatusToInt(q.Status))
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []OrderWithUser
	err := db.Order("o.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *OrderRepository) FindPage(ctx context.Context, q *query.OrderPageQuery) ([]OrderWithUser, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_order o").
		Select("o.*, u.username").
		Joins("LEFT JOIN sys_user u ON o.user_id = u.id").
		Where("o.deleted = 0")

	if q.OrderNo != "" {
		db = db.Where("o.order_no = ?", q.OrderNo)
	}
	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("o.order_no LIKE ? OR o.package_name LIKE ? OR u.username LIKE ?", kw, kw, kw)
	}
	if q.Status != "" {
		db = db.Where("o.status = ?", orderStatusToInt(q.Status))
	}
	if q.PayMethod != "" {
		db = db.Where("o.pay_method = ?", q.PayMethod)
	}
	if q.AmountMin != nil {
		db = db.Where("o.payable_amount >= ?", *q.AmountMin)
	}
	if q.AmountMax != nil {
		db = db.Where("o.payable_amount <= ?", *q.AmountMax)
	}
	if q.PaidTimeStart != "" {
		db = db.Where("o.paid_time >= ?", q.PaidTimeStart)
	}
	if q.PaidTimeEnd != "" {
		db = db.Where("o.paid_time <= ?", q.PaidTimeEnd)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []OrderWithUser
	err := db.Order("o.id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *OrderRepository) FindPendingExpired(ctx context.Context, before time.Time) ([]model.SysOrder, error) {
	var list []model.SysOrder
	err := r.db.WithContext(ctx).
		Where("status = 1 AND expire_time < ? AND deleted = 0", before).
		Find(&list).Error
	return list, err
}

func (r *OrderRepository) CountByStatus(ctx context.Context, status int8) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("status = ? AND deleted = 0", status).
		Count(&count).Error
	return count, err
}

func (r *OrderRepository) SumRevenue(ctx context.Context, startTime, endTime string) (int64, error) {
	var total int64
	db := r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("status IN ? AND deleted = 0", []int8{2, 3})
	if startTime != "" {
		db = db.Where("paid_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("paid_time <= ?", endTime)
	}
	err := db.Select("COALESCE(SUM(paid_amount), 0)").Scan(&total).Error
	return total, err
}

func (r *OrderRepository) SumRefund(ctx context.Context, startTime, endTime string) (int64, error) {
	var total int64
	db := r.db.WithContext(ctx).
		Table("sys_refund_record").
		Where("status = 2 AND deleted = 0")
	if startTime != "" {
		db = db.Where("refund_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("refund_time <= ?", endTime)
	}
	err := db.Select("COALESCE(SUM(refund_amount), 0)").Scan(&total).Error
	return total, err
}

func (r *OrderRepository) CountTotalOrders(ctx context.Context, startTime, endTime string) (int64, error) {
	var count int64
	db := r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("deleted = 0")
	if startTime != "" {
		db = db.Where("create_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("create_time <= ?", endTime)
	}
	err := db.Count(&count).Error
	return count, err
}

func (r *OrderRepository) CountByPayMethod(ctx context.Context, payMethod string) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysOrder{}).
		Where("pay_method = ? AND status IN ? AND deleted = 0", payMethod, []int8{2, 3}).
		Count(&count).Error
	return count, err
}

func (r *OrderRepository) GetPackageDistribution(ctx context.Context, startTime, endTime string) ([]PackageStatRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_order o").
		Select("o.package_id, o.package_name, COUNT(*) as count, COALESCE(SUM(o.paid_amount), 0) as revenue").
		Where("o.status IN ? AND o.deleted = 0", []int8{2, 3})
	if startTime != "" {
		db = db.Where("o.paid_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("o.paid_time <= ?", endTime)
	}

	var rows []PackageStatRow
	err := db.Group("o.package_id, o.package_name").Scan(&rows).Error
	return rows, err
}

func (r *OrderRepository) GetDailyStats(ctx context.Context, startTime, endTime string) ([]DailyStatRow, error) {
	db := r.db.WithContext(ctx).
		Table("sys_order").
		Select("DATE_FORMAT(paid_time, '%Y-%m-%d') as date, COUNT(*) as count, COALESCE(SUM(paid_amount), 0) as revenue").
		Where("status IN ? AND deleted = 0", []int8{2, 3})
	if startTime != "" {
		db = db.Where("paid_time >= ?", startTime)
	}
	if endTime != "" {
		db = db.Where("paid_time <= ?", endTime)
	}

	var rows []DailyStatRow
	err := db.Group("DATE_FORMAT(paid_time, '%Y-%m-%d')").Order("date ASC").Scan(&rows).Error
	return rows, err
}

var _ IOrderRepository = (*OrderRepository)(nil)
