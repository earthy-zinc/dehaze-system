package input_history

import (
	"context"
	"errors"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// IInputHistoryRepository 图像输入历史记录仓储接口
type IInputHistoryRepository interface {
	FindPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, status int) ([]model.SysInputHistory, int64, error)
	FindByID(ctx context.Context, id int64) (*model.SysInputHistory, error)
	Create(ctx context.Context, history *model.SysInputHistory) error
	CountByUserID(ctx context.Context, userID int64) (int64, error)
	DeleteOldest(ctx context.Context, userID int64) error
	DeleteByUserAndIDs(ctx context.Context, userID int64, ids []int64) (int64, error)
	DeleteByUserID(ctx context.Context, userID int64) (int64, error)
}

type inputHistoryRepository struct {
	db *gorm.DB
}

func NewInputHistoryRepository(db *gorm.DB) IInputHistoryRepository {
	return &inputHistoryRepository{db: db}
}

func (r *inputHistoryRepository) FindPage(ctx context.Context, userID int64, pageNum, pageSize int, inputSource, keyword string, status int) ([]model.SysInputHistory, int64, error) {
	var list []model.SysInputHistory
	var total int64

	query := r.db.WithContext(ctx).Model(&model.SysInputHistory{}).Where("user_id = ?", userID)
	if inputSource != "" {
		query = query.Where("input_source = ?", inputSource)
	}
	if keyword != "" {
		like := "%" + escapeLike(keyword) + "%"
		query = query.Where("original_image_url LIKE ? OR algorithm_name LIKE ?", like, like)
	}
	if status > 0 {
		query = query.Where("status = ?", status)
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
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &history, nil
}

func (r *inputHistoryRepository) Create(ctx context.Context, history *model.SysInputHistory) error {
	return r.db.WithContext(ctx).Create(history).Error
}

func (r *inputHistoryRepository) CountByUserID(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysInputHistory{}).Where("user_id = ?", userID).Count(&count).Error
	return count, err
}

func (r *inputHistoryRepository) DeleteOldest(ctx context.Context, userID int64) error {
	var oldest model.SysInputHistory
	err := r.db.WithContext(ctx).
		Where("user_id = ?", userID).
		Order("create_time ASC, id ASC").
		First(&oldest).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	return r.db.WithContext(ctx).Delete(&oldest).Error
}

func (r *inputHistoryRepository) DeleteByUserAndIDs(ctx context.Context, userID int64, ids []int64) (int64, error) {
	result := r.db.WithContext(ctx).Where("user_id = ? AND id IN ?", userID, ids).Delete(&model.SysInputHistory{})
	return result.RowsAffected, result.Error
}

func (r *inputHistoryRepository) DeleteByUserID(ctx context.Context, userID int64) (int64, error) {
	result := r.db.WithContext(ctx).Where("user_id = ?", userID).Delete(&model.SysInputHistory{})
	return result.RowsAffected, result.Error
}

// escapeLike 转义 LIKE 通配符，防止用户关键词注入 %/_ 通配
func escapeLike(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "%", "\\%")
	s = strings.ReplaceAll(s, "_", "\\_")
	return s
}
