package algorithm_favorite

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IAlgorithmFavoriteRepository 算法收藏仓储接口
type IAlgorithmFavoriteRepository interface {
	Create(ctx context.Context, favorite *model.SysAlgorithmFavorite) error
	Delete(ctx context.Context, userID, algorithmID int64) error
	FindByUserAndAlgorithm(ctx context.Context, userID, algorithmID int64) (*model.SysAlgorithmFavorite, error)
	FindByUserID(ctx context.Context, userID int64) ([]model.SysAlgorithmFavorite, error)
	IsFavorited(ctx context.Context, userID, algorithmID int64) (bool, error)
}

type repository struct {
	db *gorm.DB
}

func NewRepository(db *gorm.DB) IAlgorithmFavoriteRepository {
	return &repository{db: db}
}

func (r *repository) Create(ctx context.Context, f *model.SysAlgorithmFavorite) error {
	return r.db.WithContext(ctx).Create(f).Error
}

func (r *repository) Delete(ctx context.Context, userID, algorithmID int64) error {
	return r.db.WithContext(ctx).Where("user_id = ? AND algorithm_id = ?", userID, algorithmID).
		Delete(&model.SysAlgorithmFavorite{}).Error
}

func (r *repository) FindByUserAndAlgorithm(ctx context.Context, userID, algorithmID int64) (*model.SysAlgorithmFavorite, error) {
	var f model.SysAlgorithmFavorite
	err := r.db.WithContext(ctx).Where("user_id = ? AND algorithm_id = ?", userID, algorithmID).First(&f).Error
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func (r *repository) FindByUserID(ctx context.Context, userID int64) ([]model.SysAlgorithmFavorite, error) {
	var list []model.SysAlgorithmFavorite
	err := r.db.WithContext(ctx).Where("user_id = ?", userID).Order("create_time DESC").Find(&list).Error
	return list, err
}

func (r *repository) IsFavorited(ctx context.Context, userID, algorithmID int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAlgorithmFavorite{}).
		Where("user_id = ? AND algorithm_id = ?", userID, algorithmID).Count(&count).Error
	return count > 0, err
}
