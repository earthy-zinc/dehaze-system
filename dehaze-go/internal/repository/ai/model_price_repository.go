package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type ModelPriceRepository struct {
	db *gorm.DB
}

func NewModelPriceRepository(db *gorm.DB) *ModelPriceRepository {
	return &ModelPriceRepository{db: db}
}

// NextPriceVersion 取同模型同供应商的下一价格版本号（含软删历史，版本号不复用）
func (r *ModelPriceRepository) NextPriceVersion(ctx context.Context, modelID string, providerID int64) (int, error) {
	var maxVersion *int
	err := r.db.WithContext(ctx).Unscoped().Model(&model.SysAiModelPrice{}).
		Where("model_id = ? AND provider_id = ?", modelID, providerID).
		Select("MAX(price_version)").Scan(&maxVersion).Error
	if err != nil {
		return 0, err
	}
	if maxVersion == nil {
		return 1, nil
	}
	return *maxVersion + 1, nil
}

func (r *ModelPriceRepository) Create(ctx context.Context, p *model.SysAiModelPrice) error {
	return r.db.WithContext(ctx).Create(p).Error
}

func (r *ModelPriceRepository) CreateDetails(ctx context.Context, details []model.SysAiModelPriceDetail) error {
	if len(details) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Create(&details).Error
}

func (r *ModelPriceRepository) GetByID(ctx context.Context, id int64) (*model.SysAiModelPrice, error) {
	var p model.SysAiModelPrice
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &p, err
}

func (r *ModelPriceRepository) Paginate(ctx context.Context, page, size int, modelID string, providerID *int64) ([]model.SysAiModelPrice, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiModelPrice{}).Where("deleted = 0")
	if modelID != "" {
		db = db.Where("model_id = ?", modelID)
	}
	if providerID != nil {
		db = db.Where("provider_id = ?", *providerID)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiModelPrice
	err := db.Order("id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *ModelPriceRepository) ListDetails(ctx context.Context, priceIDs []int64) ([]model.SysAiModelPriceDetail, error) {
	if len(priceIDs) == 0 {
		return []model.SysAiModelPriceDetail{}, nil
	}
	var items []model.SysAiModelPriceDetail
	err := r.db.WithContext(ctx).
		Where("price_id IN ? AND deleted = 0", priceIDs).
		Order("id ASC").Find(&items).Error
	return items, err
}

func (r *ModelPriceRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiModelPrice{}).Where("id = ?", id).Updates(updates).Error
}

func (r *ModelPriceRepository) SoftDelete(ctx context.Context, id int64) error {
	now := time.Now()
	if err := r.db.WithContext(ctx).Model(&model.SysAiModelPrice{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{"deleted": gorm.Expr("id"), "update_time": now}).Error; err != nil {
		return err
	}
	return r.db.WithContext(ctx).Model(&model.SysAiModelPriceDetail{}).
		Where("price_id = ?", id).
		Updates(map[string]interface{}{"deleted": gorm.Expr("id"), "update_time": now}).Error
}
