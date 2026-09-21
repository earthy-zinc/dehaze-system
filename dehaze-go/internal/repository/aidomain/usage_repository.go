package aidomain

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// UsageRepository AI 运营统计（供应商/模型/计费流水）只读聚合。
type UsageRepository struct {
	db *gorm.DB
}

func NewUsageRepository(db *gorm.DB) *UsageRepository {
	return &UsageRepository{db: db}
}

// ListProviders 全部未删供应商。
func (r *UsageRepository) ListProviders(ctx context.Context) ([]model.SysAiProvider, error) {
	var items []model.SysAiProvider
	err := r.db.WithContext(ctx).Where("deleted = 0").Order("sort_order ASC, id ASC").Find(&items).Error
	return items, err
}

// ModelUsageRow 模型用量聚合行。
type ModelUsageRow struct {
	Model        string `gorm:"column:model"`
	CallCount    int64  `gorm:"column:call_count"`
	InputTokens  int64  `gorm:"column:input_tokens"`
	OutputTokens int64  `gorm:"column:output_tokens"`
	Credits      int64  `gorm:"column:credits"`
}

// ModelUsageByModel 按模型聚合计费流水。
func (r *UsageRepository) ModelUsageByModel(ctx context.Context, startTime, endTime *time.Time) ([]ModelUsageRow, error) {
	db := r.db.WithContext(ctx).Table("sys_ai_billing").
		Select("model, COUNT(*) AS call_count, COALESCE(SUM(input_tokens), 0) AS input_tokens, " +
			"COALESCE(SUM(output_tokens), 0) AS output_tokens, COALESCE(SUM(credits), 0) AS credits").
		Group("model")
	if startTime != nil {
		db = db.Where("create_time >= ?", *startTime)
	}
	if endTime != nil {
		db = db.Where("create_time <= ?", *endTime)
	}
	var rows []ModelUsageRow
	err := db.Scan(&rows).Error
	return rows, err
}

// DowngradeRow 降级发生次数聚合行（actual_model 非空即发生降级）。
type DowngradeRow struct {
	ModelID string `gorm:"column:model_id"`
	Count   int64  `gorm:"column:cnt"`
}

func (r *UsageRepository) DowngradeByModel(ctx context.Context, startTime, endTime *time.Time) ([]DowngradeRow, error) {
	db := r.db.WithContext(ctx).Table("sys_ai_billing").
		Select("actual_model AS model_id, COUNT(*) AS cnt").
		Where("actual_model IS NOT NULL").
		Group("actual_model")
	if startTime != nil {
		db = db.Where("create_time >= ?", *startTime)
	}
	if endTime != nil {
		db = db.Where("create_time <= ?", *endTime)
	}
	var rows []DowngradeRow
	err := db.Scan(&rows).Error
	return rows, err
}

// ModelDisplayNameRow 模型展示名。
type ModelDisplayNameRow struct {
	ModelID     string `gorm:"column:model_id"`
	DisplayName string `gorm:"column:display_name"`
}

func (r *UsageRepository) ModelDisplayNames(ctx context.Context, modelIDs []string) (map[string]string, error) {
	result := make(map[string]string)
	if len(modelIDs) == 0 {
		return result, nil
	}
	var rows []ModelDisplayNameRow
	err := r.db.WithContext(ctx).Table("sys_ai_model").
		Select("model_id, display_name").
		Where("model_id IN ?", modelIDs).Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		result[row.ModelID] = row.DisplayName
	}
	return result, nil
}
