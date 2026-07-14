package dataset

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type DatasetRepository struct {
	db *gorm.DB
}

func NewDatasetRepository(db *gorm.DB) *DatasetRepository {
	return &DatasetRepository{db: db}
}

func (r *DatasetRepository) FindByID(ctx context.Context, id int64) (*model.SysDataset, error) {
	var dataset model.SysDataset
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = ?", id, 0).
		First(&dataset).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dataset, err
}

func (r *DatasetRepository) FindAll(ctx context.Context) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("deleted = ?", 0).
		Order("id ASC").
		Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) FindAllActive(ctx context.Context) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("status = ? AND deleted = ?", 1, 0).
		Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) FindRootPage(ctx context.Context, q *query.DatasetQuery) ([]model.SysDataset, int64, error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("parent_id = ? AND deleted = ?", ROOT_NODE_ID, 0)

	if q != nil && q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if q != nil && q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q != nil && q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var datasets []model.SysDataset
	offset := (pageNum - 1) * pageSize
	err := db.Order("id ASC").Offset(offset).Limit(pageSize).Find(&datasets).Error
	if err != nil {
		return nil, 0, err
	}

	return datasets, total, nil
}

func (r *DatasetRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("parent_id = ? AND deleted = ?", parentID, 0).
		Order("id ASC").
		Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) FindByParentIDs(ctx context.Context, parentIDs []int64) ([]model.SysDataset, error) {
	if len(parentIDs) == 0 {
		return nil, nil
	}
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("parent_id IN ? AND deleted = ?", parentIDs, 0).
		Order("id ASC").
		Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) CountHasChildren(ctx context.Context, parentIDs []int64) (map[int64]bool, error) {
	result := make(map[int64]bool)
	if len(parentIDs) == 0 {
		return result, nil
	}

	var counts []CountByDatasetResult
	err := r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Select("parent_id AS dataset_id, COUNT(*) AS cnt").
		Where("parent_id IN ? AND deleted = ?", parentIDs, 0).
		Group("parent_id").
		Scan(&counts).Error
	if err != nil {
		return nil, err
	}

	for _, c := range counts {
		result[c.DatasetID] = c.Cnt > 0
	}
	return result, nil
}

func (r *DatasetRepository) ExistsByParentIDAndName(ctx context.Context, parentID int64, name string, excludeID int64) (bool, error) {
	var count int64
	db := r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("parent_id = ? AND name = ? AND deleted = ?", parentID, name, 0)
	if excludeID > 0 {
		db = db.Where("id != ?", excludeID)
	}
	err := db.Count(&count).Error
	return count > 0, err
}

func (r *DatasetRepository) Create(ctx context.Context, dataset *model.SysDataset) error {
	return r.db.WithContext(ctx).Create(dataset).Error
}

func (r *DatasetRepository) Update(ctx context.Context, dataset *model.SysDataset) error {
	return r.db.WithContext(ctx).Model(dataset).
		Select("parent_id", "type", "name", "description", "path", "status", "update_time", "update_by").
		Updates(dataset).Error
}

func (r *DatasetRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{"deleted": 1}).Error
}

func (r *DatasetRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

func (r *DatasetRepository) GetFormData(ctx context.Context, datasetID int64) (*model.SysDataset, error) {
	var dataset model.SysDataset
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = ?", datasetID, 0).
		First(&dataset).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &dataset, nil
}

func (r *DatasetRepository) Transaction(ctx context.Context, fn func(txRepo IDatasetRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRepo := NewDatasetRepository(tx)
		return fn(txRepo)
	})
}

var _ IDatasetRepository = (*DatasetRepository)(nil)
