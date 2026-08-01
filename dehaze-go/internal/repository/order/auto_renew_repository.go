package order

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type AutoRenewRepository struct {
	db *gorm.DB
}

func NewAutoRenewRepository(db *gorm.DB) *AutoRenewRepository {
	return &AutoRenewRepository{db: db}
}

func (r *AutoRenewRepository) FindByUserIDAndPackageID(ctx context.Context, userID, packageID int64) (*model.SysAutoRenew, error) {
	var ar model.SysAutoRenew
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND package_id = ? AND deleted = 0", userID, packageID).
		First(&ar).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &ar, err
}

func (r *AutoRenewRepository) Create(ctx context.Context, ar *model.SysAutoRenew) error {
	return r.db.WithContext(ctx).Create(ar).Error
}

func (r *AutoRenewRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysAutoRenew{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *AutoRenewRepository) Upsert(ctx context.Context, ar *model.SysAutoRenew) error {
	now := time.Now()
	return r.db.Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "user_id"}, {Name: "package_id"}},
		DoUpdates: clause.Assignments(map[string]any{
			"status":         ar.Status,
			"pay_method":     ar.PayMethod,
			"next_renew_time": ar.NextRenewTime,
			"fail_count":     0,
			"close_reason":   ar.CloseReason,
			"deleted":        0,
			"update_time":    now,
		}),
	}).Create(ar).Error
}

func (r *AutoRenewRepository) FindDueRenewals(ctx context.Context, before interface{}) ([]model.SysAutoRenew, error) {
	var list []model.SysAutoRenew
	err := r.db.WithContext(ctx).
		Where("status = 1 AND next_renew_time IS NOT NULL AND next_renew_time <= ? AND deleted = 0", before).
		Find(&list).Error
	return list, err
}

var _ IAutoRenewRepository = (*AutoRenewRepository)(nil)
