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
		Where("id = ?", id).
		First(&dataset).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dataset, err
}

func (r *DatasetRepository) FindAll(ctx context.Context) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Order("id ASC").
		Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) FindAllActive(ctx context.Context) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("status = ?", 1).
		Find(&datasets).Error
	return datasets, err
}

// FindDatasetsWithClearGT 查询含清晰图 GT（type=clear）的启用数据集，用于算法评估测试集选项。
// 数据集自身存在 type=clear 的 item_file 即视为含 GT；taskType 非空时按数据集 type 过滤。
func (r *DatasetRepository) FindDatasetsWithClearGT(ctx context.Context, taskType string) ([]model.SysDataset, error) {
	db := r.db.WithContext(ctx).
		Distinct("sys_dataset.*").
		Joins("JOIN sys_dataset_item ON sys_dataset_item.dataset_id = sys_dataset.id").
		Joins("JOIN sys_item_file ON sys_item_file.item_id = sys_dataset_item.id").
		Where("sys_dataset.deleted = 0 AND sys_dataset.status = 1 AND sys_item_file.type = ?", "clear")
	if taskType != "" {
		db = db.Where("sys_dataset.type = ?", taskType)
	}
	var datasets []model.SysDataset
	err := db.Order("sys_dataset.id ASC").Find(&datasets).Error
	return datasets, err
}

func (r *DatasetRepository) FindRootPage(ctx context.Context, q *query.DatasetQuery) ([]model.SysDataset, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("parent_id = ?", ROOT_NODE_ID)

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var datasets []model.SysDataset
	offset := (q.PageNum - 1) * q.PageSize
	err := db.Order("id ASC").Offset(offset).Limit(q.PageSize).Find(&datasets).Error
	if err != nil {
		return nil, 0, err
	}

	return datasets, total, nil
}

func (r *DatasetRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysDataset, error) {
	var datasets []model.SysDataset
	err := r.db.WithContext(ctx).
		Where("parent_id = ?", parentID).
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
		Where("parent_id IN ?", parentIDs).
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
		Where("parent_id IN ?", parentIDs).
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
		Where("parent_id = ? AND name = ?", parentID, name)
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

func (r *DatasetRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

func (r *DatasetRepository) GetFormData(ctx context.Context, datasetID int64) (*model.SysDataset, error) {
	var dataset model.SysDataset
	err := r.db.WithContext(ctx).
		Where("id = ?", datasetID).
		First(&dataset).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}
	return &dataset, nil
}

func (r *DatasetRepository) ExistsByID(ctx context.Context, id int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDataset{}).
		Where("id = ? AND deleted = 0", id).
		Count(&count).Error
	return count > 0, err
}

func (r *DatasetRepository) Transaction(ctx context.Context, fn func(txRepo IDatasetRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRepo := NewDatasetRepository(tx)
		return fn(txRepo)
	})
}

var _ IDatasetRepository = (*DatasetRepository)(nil)
