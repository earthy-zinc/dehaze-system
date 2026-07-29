package member

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type MemberRepository struct {
	db *gorm.DB
}

func NewMemberRepository(db *gorm.DB) *MemberRepository {
	return &MemberRepository{db: db}
}

func (r *MemberRepository) FindByUserID(ctx context.Context, userID int64) (*model.SysMember, error) {
	var m model.SysMember
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND deleted = 0", userID).
		First(&m).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &m, err
}

func (r *MemberRepository) FindWithUserByUserID(ctx context.Context, userID int64) (*MemberWithUser, error) {
	var result MemberWithUser
	err := r.db.WithContext(ctx).
		Table("sys_member m").
		Select("m.*, u.username, u.nickname, u.avatar").
		Joins("LEFT JOIN sys_user u ON m.user_id = u.id").
		Where("m.user_id = ? AND m.deleted = 0", userID).
		Scan(&result).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if result.UserID == 0 {
		return nil, nil
	}
	return &result, err
}

func (r *MemberRepository) FindPageWithUser(ctx context.Context, q *query.MemberPageQuery) ([]MemberWithUser, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Table("sys_member m").
		Select("m.*, u.username, u.nickname, u.avatar").
		Joins("LEFT JOIN sys_user u ON m.user_id = u.id").
		Where("m.deleted = 0 AND u.deleted = 0")

	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", kw, kw, kw)
	}
	if q.LevelCode != "" {
		db = db.Where("m.level_code = ?", q.LevelCode)
	}
	if q.Status != nil {
		db = db.Where("m.status = ?", *q.Status)
	}
	if q.ExpireTimeStart != "" {
		db = db.Where("m.expire_time >= ?", q.ExpireTimeStart)
	}
	if q.ExpireTimeEnd != "" {
		db = db.Where("m.expire_time <= ?", q.ExpireTimeEnd)
	}
	if q.GrowthMin != nil {
		db = db.Where("m.growth_value >= ?", *q.GrowthMin)
	}
	if q.GrowthMax != nil {
		db = db.Where("m.growth_value <= ?", *q.GrowthMax)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []MemberWithUser
	err := db.Order("m.become_member_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *MemberRepository) FindAllActive(ctx context.Context, excludeQuotaResetMonth *int, limit int) ([]model.SysMember, error) {
	var list []model.SysMember
	db := r.db.WithContext(ctx).Where("deleted = 0")
	if excludeQuotaResetMonth != nil {
		db = db.Where("quota_reset_month IS NULL OR quota_reset_month != ?", *excludeQuotaResetMonth)
	}
	if limit > 0 {
		db = db.Limit(limit)
	}
	err := db.Find(&list).Error
	return list, err
}

func (r *MemberRepository) FindExpiredNonGrowth(ctx context.Context, now time.Time) ([]model.SysMember, error) {
	var list []model.SysMember
	err := r.db.WithContext(ctx).
		Where("deleted = 0 AND expire_time IS NOT NULL AND expire_time < ? AND level_source != ?", now, "growth").
		Find(&list).Error
	return list, err
}

func (r *MemberRepository) FindExpiringBetween(ctx context.Context, start, end time.Time) ([]model.SysMember, error) {
	var list []model.SysMember
	err := r.db.WithContext(ctx).
		Where("deleted = 0 AND expire_time IS NOT NULL AND expire_time >= ? AND expire_time < ? AND level_source != ?", start, end, "growth").
		Find(&list).Error
	return list, err
}

func (r *MemberRepository) Create(ctx context.Context, m *model.SysMember) error {
	return r.db.WithContext(ctx).Create(m).Error
}

func (r *MemberRepository) UpdateLevel(ctx context.Context, userID int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Updates(updates).Error
}

func (r *MemberRepository) UpdateGrowth(ctx context.Context, userID int64, growthValue int64) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Update("growth_value", growthValue).Error
}

func (r *MemberRepository) UpdateStatus(ctx context.Context, userID int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Updates(updates).Error
}

func (r *MemberRepository) Update(ctx context.Context, userID int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Updates(updates).Error
}

func (r *MemberRepository) IncrementQuotaUsed(ctx context.Context, userID int64, quotaType string, delta int) error {
	column := "monthly_dehaze_used"
	if quotaType == "evaluate" {
		column = "monthly_evaluate_used"
	}
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Update(column, gorm.Expr(column+" + ?", delta)).Error
}

func (r *MemberRepository) ResetMonthlyQuota(ctx context.Context, userID int64, dehazeQuota, evaluateQuota, quotaMonth int) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Updates(map[string]interface{}{
			"monthly_dehaze_quota":   dehazeQuota,
			"monthly_dehaze_used":    0,
			"monthly_evaluate_quota": evaluateQuota,
			"monthly_evaluate_used":  0,
			"quota_reset_month":      quotaMonth,
		}).Error
}

func (r *MemberRepository) CreateQuotaArchive(ctx context.Context, quota *model.SysMemberQuota) error {
	return r.db.WithContext(ctx).Create(quota).Error
}

func (r *MemberRepository) Transaction(ctx context.Context, fn func(repo IMemberRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return fn(&MemberRepository{db: tx})
	})
}

var _ IMemberRepository = (*MemberRepository)(nil)
