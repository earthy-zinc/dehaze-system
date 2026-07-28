package member

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type MemberGrowthLogRepository struct {
	db *gorm.DB
}

func NewMemberGrowthLogRepository(db *gorm.DB) *MemberGrowthLogRepository {
	return &MemberGrowthLogRepository{db: db}
}

func (r *MemberGrowthLogRepository) Create(ctx context.Context, log *model.SysMemberGrowthLog) error {
	return r.db.WithContext(ctx).Create(log).Error
}

func (r *MemberGrowthLogRepository) FindPageByUserID(ctx context.Context, userID int64, q *query.GrowthLogQuery) ([]model.SysMemberGrowthLog, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).
		Model(&model.SysMemberGrowthLog{}).
		Where("user_id = ?", userID)

	if q.ChangeType != "" {
		db = db.Where("change_type = ?", q.ChangeType)
	}
	if q.StartTime != "" {
		db = db.Where("create_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		db = db.Where("create_time <= ?", q.EndTime)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []model.SysMemberGrowthLog
	err := db.Order("create_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&list).Error
	return list, total, err
}

var _ IMemberGrowthLogRepository = (*MemberGrowthLogRepository)(nil)
