package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// MemoryRecoveryWindowDays 记忆软删恢复窗口（天）。
const MemoryRecoveryWindowDays = 30

// MemoryRepository AI 长期记忆数据访问。
type MemoryRepository struct {
	db *gorm.DB
}

func NewMemoryRepository(db *gorm.DB) *MemoryRepository {
	return &MemoryRepository{db: db}
}

func (r *MemoryRepository) Create(ctx context.Context, m *model.SysAiMemory) error {
	return r.db.WithContext(ctx).Create(m).Error
}

func (r *MemoryRepository) GetByIDAndUser(ctx context.Context, id, userID int64) (*model.SysAiMemory, error) {
	var m model.SysAiMemory
	err := r.db.WithContext(ctx).
		Where("id = ? AND user_id = ? AND deleted = 0", id, userID).
		First(&m).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &m, err
}

// ListActive 活跃记忆分页（status=1 且未归档，importance 倒序）。
func (r *MemoryRepository) ListActive(ctx context.Context, userID int64, memoryType, source string, page, size int) ([]model.SysAiMemory, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("user_id = ? AND deleted = 0 AND status = 1 AND archived = 0", userID)
	if memoryType != "" {
		db = db.Where("memory_type = ?", memoryType)
	}
	if source != "" {
		db = db.Where("source = ?", source)
	}
	db = db.Order("importance DESC, create_time DESC")
	return countAndFind[model.SysAiMemory](db, page, size)
}

// ListArchived 归档记忆分页。
func (r *MemoryRepository) ListArchived(ctx context.Context, userID int64, memoryType string, page, size int) ([]model.SysAiMemory, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("user_id = ? AND deleted = 0 AND archived = 1", userID)
	if memoryType != "" {
		db = db.Where("memory_type = ?", memoryType)
	}
	db = db.Order("importance DESC, create_time DESC")
	return countAndFind[model.SysAiMemory](db, page, size)
}

func (r *MemoryRepository) SearchByKeyword(ctx context.Context, userID int64, keyword string, limit int) ([]model.SysAiMemory, error) {
	var items []model.SysAiMemory
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND deleted = 0 AND status = 1 AND archived = 0 AND content LIKE ? ESCAPE '\\\\'",
			userID, "%"+escapeLike(keyword)+"%").
		Order("importance DESC").Limit(limit).Find(&items).Error
	return items, err
}

// ListForExport 导出用活跃记忆（按 importance 排序）。
func (r *MemoryRepository) ListForExport(ctx context.Context, userID int64, limit int) ([]model.SysAiMemory, error) {
	var items []model.SysAiMemory
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND deleted = 0 AND status = 1 AND archived = 0", userID).
		Order("importance DESC, last_accessed_at DESC").Limit(limit).Find(&items).Error
	return items, err
}

// Touch 检索命中重激活：access_count+1、重置衰减计时器、importance+5（上限 100）。
func (r *MemoryRepository) Touch(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("id = ?", id).
		Updates(map[string]any{
			"access_count":     gorm.Expr("access_count + 1"),
			"last_accessed_at": time.Now(),
			"importance":       gorm.Expr("LEAST(100, importance + 5)"),
		}).Error
}

// BatchClear 批量清空（软删 + delete_time），返回受影响条数。
// create_time 的区间闭区间与 python `batch_clear` 一致；delete_time 截断到秒
// （列为 DATETIME 秒精度，直接写 time.Now() 会被 MySQL 进位成下一刻）。
func (r *MemoryRepository) BatchClear(ctx context.Context, userID int64, memoryType string, start, end *time.Time) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("user_id = ? AND deleted = 0", userID)
	if memoryType != "" {
		db = db.Where("memory_type = ?", memoryType)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	res := db.Updates(map[string]any{"deleted": gorm.Expr("id"), "delete_time": time.Now().Truncate(time.Second)})
	return res.RowsAffected, res.Error
}

// ListDeletedForRestore 查询恢复窗口内已软删记忆。
// 需 Unscoped：全局软删回调会给所有含 Deleted 字段的模型追加 deleted = 0，与 deleted <> 0 互斥，
// 不加则恢复窗口内的软删记忆恒为空。
func (r *MemoryRepository) ListDeletedForRestore(ctx context.Context, userID int64, memoryType string, start, end *time.Time) ([]model.SysAiMemory, error) {
	windowStart := time.Now().AddDate(0, 0, -MemoryRecoveryWindowDays)
	db := r.db.WithContext(ctx).Unscoped().
		Where("user_id = ? AND deleted <> 0 AND delete_time >= ?", userID, windowStart)
	if memoryType != "" {
		db = db.Where("memory_type = ?", memoryType)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var items []model.SysAiMemory
	err := db.Find(&items).Error
	return items, err
}

func (r *MemoryRepository) RestoreDeleted(ctx context.Context, ids []int64) (int64, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	res := r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("id IN ?", ids).
		Updates(map[string]any{"deleted": 0, "delete_time": nil})
	return res.RowsAffected, res.Error
}

// SoftDeleteWithTime 单条软删（含 delete_time，30 天内可恢复）。
func (r *MemoryRepository) SoftDeleteWithTime(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("id = ?", id).
		Updates(map[string]any{"deleted": gorm.Expr("id"), "delete_time": time.Now().Truncate(time.Second)}).Error
}

// UpdateFields 更新记忆可变字段。
func (r *MemoryRepository) UpdateFields(ctx context.Context, id int64, fields map[string]any) error {
	return r.db.WithContext(ctx).Model(&model.SysAiMemory{}).
		Where("id = ?", id).Updates(fields).Error
}
