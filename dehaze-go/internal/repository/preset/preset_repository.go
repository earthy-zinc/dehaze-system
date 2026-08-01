package preset

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type IPresetRepository interface {
	FindPage(ctx context.Context, algorithmID int64, userID int64, isSystem *bool, page, pageSize int) ([]model.SysPreset, int64, error)
	FindByID(ctx context.Context, id int64) (*model.SysPreset, error)
	Create(ctx context.Context, preset *model.SysPreset) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	Delete(ctx context.Context, id int64) error
	CountByUser(ctx context.Context, userID int64) (int64, error)
}

type presetRepository struct {
	db *gorm.DB
}

func NewPresetRepository(db *gorm.DB) IPresetRepository {
	return &presetRepository{db: db}
}

func (r *presetRepository) FindPage(ctx context.Context, algorithmID int64, userID int64, isSystem *bool, page, pageSize int) ([]model.SysPreset, int64, error) {
	query := r.db.WithContext(ctx).Model(&model.SysPreset{})

	if isSystem != nil && *isSystem {
		// 仅系统预设
		query = query.Where("type = ?", "system")
	} else if isSystem != nil && !*isSystem {
		// 仅用户自定义预设
		query = query.Where("type = ? AND user_id = ?", "custom", userID)
	} else {
		// 系统预设 + 当前用户自定义
		query = query.Where("type = ? OR (type = ? AND user_id = ?)", "system", "custom", userID)
	}

	if algorithmID > 0 {
		query = query.Where("algorithm_id = ?", algorithmID)
	}

	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []model.SysPreset
	if err := query.Order("type ASC, create_time DESC").
		Offset((page - 1) * pageSize).Limit(pageSize).
		Find(&list).Error; err != nil {
		return nil, 0, err
	}
	return list, total, nil
}

func (r *presetRepository) FindByID(ctx context.Context, id int64) (*model.SysPreset, error) {
	var preset model.SysPreset
	err := r.db.WithContext(ctx).First(&preset, id).Error
	if err != nil {
		return nil, err
	}
	return &preset, nil
}

func (r *presetRepository) Create(ctx context.Context, preset *model.SysPreset) error {
	return r.db.WithContext(ctx).Create(preset).Error
}

func (r *presetRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).Model(&model.SysPreset{}).Where("id = ?", id).Updates(updates).Error
}

func (r *presetRepository) Delete(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Delete(&model.SysPreset{}, id).Error
}

func (r *presetRepository) CountByUser(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysPreset{}).
		Where("type = ? AND user_id = ?", "custom", userID).
		Count(&count).Error
	return count, err
}
