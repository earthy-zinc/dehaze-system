package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// ScheduleRepository AI 对话定时任务数据访问。
type ScheduleRepository struct {
	db *gorm.DB
}

func NewScheduleRepository(db *gorm.DB) *ScheduleRepository {
	return &ScheduleRepository{db: db}
}

func (r *ScheduleRepository) Create(ctx context.Context, s *model.SysAiSchedule) error {
	return r.db.WithContext(ctx).Create(s).Error
}

func (r *ScheduleRepository) GetByID(ctx context.Context, id int64) (*model.SysAiSchedule, error) {
	var s model.SysAiSchedule
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&s).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &s, err
}

func (r *ScheduleRepository) CountByUser(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiSchedule{}).
		Where("user_id = ? AND deleted = 0", userID).Count(&count).Error
	return count, err
}

// PaginateByUser 任务列表：启用优先 → 下次触发时间（NULL 排后）→ id。
func (r *ScheduleRepository) PaginateByUser(ctx context.Context, userID int64, page, size int, keyword string) ([]model.SysAiSchedule, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiSchedule{}).
		Where("user_id = ? AND deleted = 0", userID)
	if keyword != "" {
		db = db.Where("name LIKE ? ESCAPE '\\\\'", "%"+escapeLike(keyword)+"%")
	}
	db = db.Order("enabled DESC, ISNULL(next_trigger_time) ASC, next_trigger_time ASC, id ASC")
	return countAndFind[model.SysAiSchedule](db, page, size)
}

func (r *ScheduleRepository) UpdateFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiSchedule{}).
		Where("id = ?", id).Updates(fields).Error
}

// SetEnabled 启停任务；启用时重置熔断计数与状态。
func (r *ScheduleRepository) SetEnabled(ctx context.Context, id int64, enabled int, updateBy int64) error {
	fields := map[string]any{"enabled": enabled, "update_time": time.Now(), "update_by": updateBy}
	if enabled == 1 {
		fields["status"] = 1
		fields["circuit_streak"] = 0
	}
	return r.db.WithContext(ctx).Model(&model.SysAiSchedule{}).
		Where("id = ?", id).Updates(fields).Error
}

func (r *ScheduleRepository) SoftDelete(ctx context.Context, id, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiSchedule{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// LatestRunRow 最近一次执行摘要。
type LatestRunRow struct {
	ScheduleID     int64     `gorm:"column:schedule_id"`
	Status         int       `gorm:"column:status"`
	SkipReason     *string   `gorm:"column:skip_reason"`
	Credits        *float64  `gorm:"column:credits"`
	DurationMs     *int      `gorm:"column:duration_ms"`
	ErrorMsg       *string   `gorm:"column:error_msg"`
	ConversationID *int64    `gorm:"column:conversation_id"`
	CreateTime     time.Time `gorm:"column:create_time"`
	Rownum         int       `gorm:"column:rn"`
}

// LatestRunsByScheduleIDs 批量取各任务最近一次执行（窗口函数，避免 N+1）。
func (r *ScheduleRepository) LatestRunsByScheduleIDs(ctx context.Context, scheduleIDs []int64) (map[int64]LatestRunRow, error) {
	result := make(map[int64]LatestRunRow)
	if len(scheduleIDs) == 0 {
		return result, nil
	}
	var rows []LatestRunRow
	err := r.db.WithContext(ctx).Raw(`
		SELECT * FROM (
			SELECT r.schedule_id, r.status, r.skip_reason, r.credits, r.duration_ms,
			       r.error_msg, r.conversation_id, r.create_time,
			       ROW_NUMBER() OVER (PARTITION BY r.schedule_id ORDER BY r.id DESC) AS rn
			FROM sys_ai_schedule_run r WHERE r.schedule_id IN (?)
		) t WHERE t.rn = 1`, scheduleIDs).Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		result[row.ScheduleID] = row
	}
	return result, nil
}

func (r *ScheduleRepository) PaginateHistory(ctx context.Context, scheduleID int64, page, size int) ([]model.SysAiScheduleRun, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiScheduleRun{}).
		Where("schedule_id = ?", scheduleID).
		Order("id DESC")
	return countAndFind[model.SysAiScheduleRun](db, page, size)
}

// MemberLevelRow 会员等级查询行。
type MemberLevelRow struct {
	LevelCode string `gorm:"column:level_code"`
}

// GetMemberLevelCode 取用户会员等级编码（无会员记录返回空串）。
func (r *ScheduleRepository) GetMemberLevelCode(ctx context.Context, userID int64) (string, error) {
	var row MemberLevelRow
	err := r.db.WithContext(ctx).Table("sys_member").
		Select("level_code").Where("user_id = ? AND deleted = 0", userID).
		Limit(1).Scan(&row).Error
	if err != nil {
		return "", err
	}
	return row.LevelCode, nil
}
