package order

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// ============ 余额账户 ============

type BalanceAccountRepository struct {
	db *gorm.DB
}

func NewBalanceAccountRepository(db *gorm.DB) *BalanceAccountRepository {
	return &BalanceAccountRepository{db: db}
}

// GetOrCreate 查询或初始化用户余额账户（并发初始化靠 uk_user_id 幂等）
func (r *BalanceAccountRepository) GetOrCreate(ctx context.Context, userID int64) (*model.SysBalance, error) {
	var account model.SysBalance
	err := r.db.WithContext(ctx).Where("user_id = ?", userID).First(&account).Error
	if err == nil {
		return &account, nil
	}
	if err != gorm.ErrRecordNotFound {
		return nil, err
	}
	seed := model.SysBalance{UserID: userID}
	if err := r.db.WithContext(ctx).Clauses(clause.OnConflict{DoNothing: true}).Create(&seed).Error; err != nil {
		return nil, err
	}
	err = r.db.WithContext(ctx).Where("user_id = ?", userID).First(&account).Error
	return &account, err
}

// AdjustBalance 乐观锁 CAS 调整余额（delta 可负），balance + delta < 0 时失败。
// 返回 false 表示版本冲突或余额不足，调用方重试或拒绝。
func (r *BalanceAccountRepository) AdjustBalance(ctx context.Context, userID int64, delta int64, expectedVersion int) (bool, error) {
	res := r.db.WithContext(ctx).
		Model(&model.SysBalance{}).
		Where("user_id = ? AND version = ? AND balance + ? >= 0", userID, expectedVersion, delta).
		Updates(map[string]interface{}{
			"balance": gorm.Expr("balance + ?", delta),
			"version": gorm.Expr("version + 1"),
		})
	return res.RowsAffected > 0, res.Error
}

// ============ 余额流水 ============

type BalanceLogRepository struct {
	db *gorm.DB
}

func NewBalanceLogRepository(db *gorm.DB) *BalanceLogRepository {
	return &BalanceLogRepository{db: db}
}

func (r *BalanceLogRepository) Create(ctx context.Context, log *model.SysBalanceLog) error {
	return r.db.WithContext(ctx).Create(log).Error
}

// ============ 余额退款 ============

type BalanceRefundRepository struct {
	db *gorm.DB
}

func NewBalanceRefundRepository(db *gorm.DB) *BalanceRefundRepository {
	return &BalanceRefundRepository{db: db}
}

func (r *BalanceRefundRepository) Create(ctx context.Context, record *model.SysBalanceRefund) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *BalanceRefundRepository) FindByID(ctx context.Context, id int64) (*model.SysBalanceRefund, error) {
	var record model.SysBalanceRefund
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&record).Error
	if err == gorm.ErrRecordNotFound {
		return nil, nil
	}
	return &record, err
}

// FindPendingByUserID 查询用户待审核的余额退款申请（每用户仅一条）
func (r *BalanceRefundRepository) FindPendingByUserID(ctx context.Context, userID int64) (*model.SysBalanceRefund, error) {
	var record model.SysBalanceRefund
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND status = 1 AND deleted = 0", userID).
		First(&record).Error
	if err == gorm.ErrRecordNotFound {
		return nil, nil
	}
	return &record, err
}

func (r *BalanceRefundRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysBalanceRefund{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

// ============ 充值 ============

type RechargeRepository struct {
	db *gorm.DB
}

func NewRechargeRepository(db *gorm.DB) *RechargeRepository {
	return &RechargeRepository{db: db}
}

func (r *RechargeRepository) Create(ctx context.Context, record *model.SysRecharge) error {
	return r.db.WithContext(ctx).Create(record).Error
}
