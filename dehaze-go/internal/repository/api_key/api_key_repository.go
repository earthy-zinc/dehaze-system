package api_key

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type IApiKeyRepository interface {
	Create(ctx context.Context, key *model.SysApiKey) error
	FindByHash(ctx context.Context, keyHash string) (*model.SysApiKey, error)
	FindByUserID(ctx context.Context, userID int64) ([]model.SysApiKey, error)
	RevokeByID(ctx context.Context, id int64, userID int64) error
	UpdateLastUsed(ctx context.Context, id int64) error
}

type ApiKeyRepository struct {
	db *gorm.DB
}

func NewApiKeyRepository(db *gorm.DB) *ApiKeyRepository {
	return &ApiKeyRepository{db: db}
}

func (r *ApiKeyRepository) Create(ctx context.Context, key *model.SysApiKey) error {
	return r.db.WithContext(ctx).Create(key).Error
}

func (r *ApiKeyRepository) FindByHash(ctx context.Context, keyHash string) (*model.SysApiKey, error) {
	var key model.SysApiKey
	err := r.db.WithContext(ctx).
		Where("key_hash = ? AND revoked_at IS NULL", keyHash).
		First(&key).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &key, nil
}

func (r *ApiKeyRepository) FindByUserID(ctx context.Context, userID int64) ([]model.SysApiKey, error) {
	var keys []model.SysApiKey
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND revoked_at IS NULL", userID).
		Order("create_time DESC").
		Find(&keys).Error
	return keys, err
}

func (r *ApiKeyRepository) RevokeByID(ctx context.Context, id int64, userID int64) error {
	now := time.Now()
	result := r.db.WithContext(ctx).
		Model(&model.SysApiKey{}).
		Where("id = ? AND user_id = ?", id, userID).
		Updates(map[string]any{
			"revoked_at":  now,
			"update_time": now,
		})
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return errors.New("api key not found or unauthorized")
	}
	return nil
}

func (r *ApiKeyRepository) UpdateLastUsed(ctx context.Context, id int64) error {
	now := time.Now()
	return r.db.WithContext(ctx).
		Model(&model.SysApiKey{}).
		Where("id = ?", id).
		Update("last_used_at", now).Error
}

var _ IApiKeyRepository = (*ApiKeyRepository)(nil)
