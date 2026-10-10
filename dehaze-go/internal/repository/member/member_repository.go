package member

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
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

func (r *MemberRepository) FindByUserIDs(ctx context.Context, userIDs []int64) ([]model.SysMember, error) {
	if len(userIDs) == 0 {
		return nil, nil
	}
	var list []model.SysMember
	err := r.db.WithContext(ctx).
		Where("user_id IN ? AND deleted = 0", userIDs).
		Find(&list).Error
	return list, err
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
		// 先转义字面关键字，再包裹通配符（python escape_like 同口径）
		kw := "%" + esc(q.Keywords) + "%"
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

// FindActiveUserIDsByLevelCode 分页扫描指定等级的活跃会员（未删除、未冻结），用于 VIP 月度赠送等批量任务
func (r *MemberRepository) FindActiveUserIDsByLevelCode(ctx context.Context, levelCode string, offset, limit int) ([]int64, error) {
	var userIDs []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("level_code = ? AND deleted = 0 AND status = 1", levelCode).
		Order("user_id ASC").
		Offset(offset).Limit(limit).
		Pluck("user_id", &userIDs).Error
	return userIDs, err
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

func (r *MemberRepository) FindUserIDsByLevelCodes(ctx context.Context, levelCodes []string) ([]int64, error) {
	if len(levelCodes) == 0 {
		return nil, nil
	}
	var userIDs []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("level_code IN ? AND deleted = 0", levelCodes).
		Pluck("user_id", &userIDs).Error
	return userIDs, err
}

func (r *MemberRepository) Upsert(ctx context.Context, m *model.SysMember) error {
	return r.db.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "user_id"}},
		DoUpdates: clause.AssignmentColumns([]string{"deleted", "update_time"}),
	}).Create(m).Error
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

func (r *MemberRepository) Update(ctx context.Context, userID int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysMember{}).
		Where("user_id = ? AND deleted = 0", userID).
		Updates(updates).Error
}

func (r *MemberRepository) Transaction(ctx context.Context, fn func(repo IMemberRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return fn(&MemberRepository{db: tx})
	})
}

var _ IMemberRepository = (*MemberRepository)(nil)

// esc 转义 LIKE 通配符（% _ \），配合 ESCAPE 子句使用
func esc(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "%", "\\%")
	s = strings.ReplaceAll(s, "_", "\\_")
	return s
}
