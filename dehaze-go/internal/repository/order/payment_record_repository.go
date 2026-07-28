package order

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type PaymentRecordRepository struct {
	db *gorm.DB
}

func NewPaymentRecordRepository(db *gorm.DB) *PaymentRecordRepository {
	return &PaymentRecordRepository{db: db}
}

func (r *PaymentRecordRepository) Create(ctx context.Context, p *model.SysPaymentRecord) error {
	return r.db.WithContext(ctx).Create(p).Error
}

func (r *PaymentRecordRepository) FindByOrderID(ctx context.Context, orderID int64) ([]model.SysPaymentRecord, error) {
	var list []model.SysPaymentRecord
	err := r.db.WithContext(ctx).
		Where("order_id = ?", orderID).
		Order("id DESC").
		Find(&list).Error
	return list, err
}

func (r *PaymentRecordRepository) UpdateStatus(ctx context.Context, id int64, status int8, callbackContent, errMsg string) error {
	updates := map[string]interface{}{
		"status":           status,
		"callback_time":    gorm.Expr("CURRENT_TIMESTAMP"),
		"callback_content": callbackContent,
		"error_message":    errMsg,
	}
	return r.db.WithContext(ctx).
		Model(&model.SysPaymentRecord{}).
		Where("id = ?", id).
		Updates(updates).Error
}

var _ IPaymentRecordRepository = (*PaymentRecordRepository)(nil)
