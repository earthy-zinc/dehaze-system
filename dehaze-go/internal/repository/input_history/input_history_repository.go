package input_history

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IInputHistoryRepository 图像输入历史记录仓储接口
type IInputHistoryRepository interface {
	FindPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, favoriteOnly bool) ([]model.SysInputHistory, int64, error)
	FindByID(ctx context.Context, id int64) (*model.SysInputHistory, error)
	Create(ctx context.Context, history *model.SysInputHistory) error
	Update(ctx context.Context, history *model.SysInputHistory) error
	Delete(ctx context.Context, ids []int64) error
	DeleteByUserID(ctx context.Context, userID int64) (int64, error)
}

type inputHistoryRepository struct {
	db *gorm.DB
}

func NewInputHistoryRepository(db *gorm.DB) IInputHistoryRepository {
	return &inputHistoryRepository{db: db}
}

func (r *inputHistoryRepository) FindPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, favoriteOnly bool) ([]model.SysInputHistory, int64, error) {
	var list []model.SysInputHistory
	var total int64

	query := r.db.WithContext(ctx).Model(&model.SysInputHistory{}).Where("user_id = ?", userID)
	if inputSource != "" {
		query = query.Where("input_source = ?", inputSource)
	}
	if keyword != "" {
		like := "%" + keyword + "%"
		query = query.Where("original_image_url LIKE ? OR algorithm_name LIKE ?", like, like)
	}
	if favoriteOnly {
		query = query.Where("is_favorite = ?", 1)
	}

	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	offset := (pageNum - 1) * pageSize
	if err := query.Order("create_time DESC").Offset(offset).Limit(pageSize).Find(&list).Error; err != nil {
		return nil, 0, err
	}
	return list, total, nil
}

func (r *inputHistoryRepository) FindByID(ctx context.Context, id int64) (*model.SysInputHistory, error) {
	var history model.SysInputHistory
	err := r.db.WithContext(ctx).First(&history, id).Error
	if err != nil {
		return nil, err
	}
	return &history, nil
}

func (r *inputHistoryRepository) Create(ctx context.Context, history *model.SysInputHistory) error {
	return r.db.WithContext(ctx).Create(history).Error
}

func (r *inputHistoryRepository) Update(ctx context.Context, history *model.SysInputHistory) error {
	return r.db.WithContext(ctx).Save(history).Error
}

func (r *inputHistoryRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Delete(&model.SysInputHistory{}, ids).Error
}

func (r *inputHistoryRepository) DeleteByUserID(ctx context.Context, userID int64) (int64, error) {
	result := r.db.WithContext(ctx).Where("user_id = ?", userID).Delete(&model.SysInputHistory{})
	return result.RowsAffected, result.Error
}
