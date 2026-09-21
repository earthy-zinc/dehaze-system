package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type ProviderRepository struct {
	db *gorm.DB
}

func NewProviderRepository(db *gorm.DB) *ProviderRepository {
	return &ProviderRepository{db: db}
}

// PaginateProviders 分页查询供应商（keyword 匹配 display_name/provider_code）
func (r *ProviderRepository) PaginateProviders(ctx context.Context, page, size int, keyword string) ([]model.SysAiProvider, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiProvider{}).Where("deleted = 0")
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("(display_name LIKE ? ESCAPE '\\\\' OR provider_code LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiProvider
	err := db.Order("sort_order ASC, id ASC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

func (r *ProviderRepository) GetByID(ctx context.Context, id int64) (*model.SysAiProvider, error) {
	var p model.SysAiProvider
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &p, err
}

// GetByCode 按业务键查询；includeDeleted=true 时含软删行（provider_code 白名单：软删后不可复用）
func (r *ProviderRepository) GetByCode(ctx context.Context, code string, includeDeleted bool) (*model.SysAiProvider, error) {
	db := r.db.WithContext(ctx).Where("provider_code = ?", code)
	if !includeDeleted {
		db = db.Where("deleted = 0")
	} else {
		db = db.Unscoped()
	}
	var p model.SysAiProvider
	err := db.First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &p, err
}

func (r *ProviderRepository) ListEnabled(ctx context.Context) ([]model.SysAiProvider, error) {
	var items []model.SysAiProvider
	err := r.db.WithContext(ctx).
		Where("status = 1 AND deleted = 0").
		Order("sort_order ASC, id ASC").Find(&items).Error
	return items, err
}

// CountModels 统计该供应商下的模型数（含禁用，防悬挂引用）
func (r *ProviderRepository) CountModels(ctx context.Context, providerID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Table("sys_ai_model").
		Where("provider_id = ? AND deleted = 0", providerID).
		Count(&count).Error
	return count, err
}

func (r *ProviderRepository) Create(ctx context.Context, p *model.SysAiProvider) error {
	return r.db.WithContext(ctx).Create(p).Error
}

func (r *ProviderRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiProvider{}).Where("id = ?", id).Updates(updates).Error
}

func (r *ProviderRepository) SoftDelete(ctx context.Context, id, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiProvider{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// ==================== 供应商 API Key ====================

type ProviderKeyRepository struct {
	db *gorm.DB
}

func NewProviderKeyRepository(db *gorm.DB) *ProviderKeyRepository {
	return &ProviderKeyRepository{db: db}
}

func (r *ProviderKeyRepository) ListByProvider(ctx context.Context, providerID int64) ([]model.SysAiProviderKey, error) {
	var items []model.SysAiProviderKey
	err := r.db.WithContext(ctx).
		Where("provider_id = ?", providerID).
		Order("priority ASC, id ASC").Find(&items).Error
	return items, err
}

func (r *ProviderKeyRepository) GetByID(ctx context.Context, id int64) (*model.SysAiProviderKey, error) {
	var k model.SysAiProviderKey
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&k).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &k, err
}

func (r *ProviderKeyRepository) GetByHash(ctx context.Context, hash string) (*model.SysAiProviderKey, error) {
	var k model.SysAiProviderKey
	err := r.db.WithContext(ctx).Where("key_hash = ?", hash).First(&k).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &k, err
}

func (r *ProviderKeyRepository) CountEnabledByProvider(ctx context.Context, providerID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiProviderKey{}).
		Where("provider_id = ? AND status = 1", providerID).
		Count(&count).Error
	return count, err
}

func (r *ProviderKeyRepository) Create(ctx context.Context, k *model.SysAiProviderKey) error {
	return r.db.WithContext(ctx).Create(k).Error
}

func (r *ProviderKeyRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	updates["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiProviderKey{}).Where("id = ?", id).Updates(updates).Error
}

// DeleteByID Key 管理为状态控制，删除即物理删除
func (r *ProviderKeyRepository) DeleteByID(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Where("id = ?", id).Delete(&model.SysAiProviderKey{}).Error
}
