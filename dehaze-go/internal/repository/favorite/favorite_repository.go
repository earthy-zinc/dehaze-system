package favorite

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

type FavoriteRepository struct {
	db *gorm.DB
}

func NewFavoriteRepository(db *gorm.DB) *FavoriteRepository {
	return &FavoriteRepository{db: db}
}

func (r *FavoriteRepository) Create(ctx context.Context, f *model.SysFavorite) error {
	return r.db.WithContext(ctx).Create(f).Error
}

func (r *FavoriteRepository) FindByUserAndTarget(ctx context.Context, userID int64, targetType string, targetID int64) (*model.SysFavorite, error) {
	var f model.SysFavorite
	err := r.db.WithContext(ctx).
		Where("user_id = ? AND target_type = ? AND target_id = ? AND deleted = 0", userID, targetType, targetID).
		First(&f).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &f, err
}

func (r *FavoriteRepository) Upsert(ctx context.Context, f *model.SysFavorite) error {
	if err := r.db.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "user_id"}, {Name: "target_type"}, {Name: "target_id"}},
		DoUpdates: clause.AssignmentColumns([]string{"deleted", "is_invalid", "update_time"}),
	}).Create(f).Error; err != nil {
		return err
	}
	// ON CONFLICT 走 UPDATE 分支时 GORM Create 不会回填已有行的 id，需重查拿回真实 id
	if f.ID == 0 {
		existing, err := r.FindByUserAndTarget(ctx, f.UserID, f.TargetType, f.TargetID)
		if err != nil {
			return err
		}
		if existing != nil {
			f.ID = existing.ID
		}
	}
	return nil
}

func (r *FavoriteRepository) FindPage(ctx context.Context, userID int64, q *query.FavoritePageQuery) ([]FavoriteWithAlgorithm, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}

	db := r.db.WithContext(ctx).
		Table("sys_favorite f").
		Select("f.*, COALESCE(a.name, '') as algorithm_name").
		Joins("LEFT JOIN sys_algorithm a ON f.target_type = 'algorithm' AND f.target_id = a.id").
		Where("f.user_id = ? AND f.deleted = 0", userID)

	if q.TargetType != "" {
		db = db.Where("f.target_type = ?", q.TargetType)
	}
	if q.Keywords != "" {
		kw := "%" + q.Keywords + "%"
		db = db.Where("(f.target_type = 'algorithm' AND a.name LIKE ?) OR (f.target_type != 'algorithm' AND 1=0)", kw)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	sortField := "f.create_time"
	sortOrder := "DESC"
	if q.SortBy == "create_time" {
		sortField = "f.create_time"
	}
	if q.SortOrder == "asc" || q.SortOrder == "ASC" {
		sortOrder = "ASC"
	}

	var list []FavoriteWithAlgorithm
	err := db.Order(sortField + " " + sortOrder).
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Scan(&list).Error
	return list, total, err
}

func (r *FavoriteRepository) CountByUserID(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Where("user_id = ? AND deleted = 0", userID).
		Count(&count).Error
	return count, err
}

func (r *FavoriteRepository) CountByUserAndType(ctx context.Context, userID int64, targetType string) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Where("user_id = ? AND target_type = ? AND deleted = 0", userID, targetType).
		Count(&count).Error
	return count, err
}

func (r *FavoriteRepository) CountGroupByType(ctx context.Context, userID int64, targetType string) ([]CountByTypeRow, error) {
	db := r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Select("target_type, COUNT(*) as count").
		Where("user_id = ? AND deleted = 0", userID)

	if targetType != "" {
		db = db.Where("target_type = ?", targetType)
	}

	var rows []CountByTypeRow
	err := db.Group("target_type").Scan(&rows).Error
	return rows, err
}

func (r *FavoriteRepository) DeleteByIDs(ctx context.Context, userID int64, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Where("user_id = ? AND id IN ? AND deleted = 0", userID, ids).
		Update("deleted", 1).Error
}

func (r *FavoriteRepository) UpdateByID(ctx context.Context, id int64, updates map[string]any) error {
	return r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Where("id = ?", id).
		Updates(updates).Error
}

func (r *FavoriteRepository) MarkInvalid(ctx context.Context, targetType string, targetIDs []int64) error {
	if len(targetIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysFavorite{}).
		Where("target_type = ? AND target_id IN ? AND deleted = 0", targetType, targetIDs).
		Update("is_invalid", 1).Error
}

var _ IFavoriteRepository = (*FavoriteRepository)(nil)
